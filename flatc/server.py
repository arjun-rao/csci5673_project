import socket
import threading
import time
import pickle
import numpy as np
import os
from threading import Thread, Lock
from model import Model
from message import Message, CLIENT_WEIGHT_UPDATE, REQUEST_GLOBAL_MSG, REPLY_GLOBAL_MSG

import tqdm

mutex = Lock()

SEPARATOR = "<SEP>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step



class Server:

    """Defines the server"""

    MAX_WAITING_CONNECTIONS = 5


    def __init__(self, host, port, modelid = 0):
        """
        Initializes a new Server
        :param host: Host on which server bounds
        :param port: port on which server bound
        """

        #threading.Thread.__init__(self)
        self.modelid = modelid
        self.host = host
        self.port = port

    def bind_socket(self):
        """
        Creates the server socket and binds to given host and port
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        print('Waiting for a Connection here...')
        self.server_socket.listen(self.MAX_WAITING_CONNECTIONS)

    def send(self, connection, msg):
        """
        :param sock: incoming socket
        :param msg: the message/weights to send
        """
        connection.send(msg)


    def process_weights(self, weights, instance_counts):
        """Algorith to process weights"""
        return Model.compute_weighted_average(weights, instance_counts)

    def HandleClientRequest(self, connection):

        received = connection.recv(50)
        received = received.decode()
        msg = received.split(SEPARATOR)
        if msg[0] == CLIENT_WEIGHT_UPDATE:
            filename, filesize = os.path.basename(msg[1]), int(msg[2])
            progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
            os.makedirs('server_data', exist_ok=True)
            with open(os.path.join('server_data', filename), "wb") as f:
                for _ in progress:
                    # read 1024 bytes from the socket (receive)
                    bytes_read = connection.recv(BUFFER_SIZE)
                    if not bytes_read:
                        # nothing is received
                        # file transmitting is done
                        break
                    # write to the file the bytes we just received
                    f.write(bytes_read)
                    # update the progress bar
                    progress.update(len(bytes_read))
            print("msg received received!")
            msg = pickle.load(open(os.path.join('server_data', filename), 'rb'))
            if msg.msg_type==CLIENT_WEIGHT_UPDATE:
                self.WriteClientWeights(msg.weights, msg.clientno, msg.roundno, msg.instance_count)
        elif msg[0] == REQUEST_GLOBAL_MSG:
            self.ReturnUpdatedWeights(connection, msg[1])

    def ReturnUpdatedWeights(self, connection, modelid):
        if modelid !=-1 and self.modelid==modelid:
            #no model update
            connection.send(b'')
            return
        mutex.acquire()
        try:
            client_weight_files = [item for item in os.listdir("server_data") if item.endswith('.npy')]
            if len(client_weight_files) == 0 and os.path.isfile("server_data/GlobalModel.npy.pkl"):
                self.send_global_weights(connection, 'server_data/GlobalModel.npy.pkl', str(self.modelid))
            elif len(client_weight_files) != 0:
                weights_list = []
                instance_counts = []
                for filename in os.listdir("server_data"):
                    if filename.endswith(".npy"):
                        weights = np.load(os.path.join("server_data", filename), allow_pickle=True)
                        instance_counts.append(int(filename.split("_")[2].split('.')[0]))
                        weights_list.append(weights)
                updatedWeights = self.process_weights(weights_list, instance_counts)
                if os.path.isfile("server_data/GlobalModel.npy.pkl"):
                    os.remove("server_data/GlobalModel.npy.pkl")
                self.modelid += 1
                np.save(open("server_data/GlobalModel.npy.pkl", 'wb'), updatedWeights)

                for filename in client_weight_files:
                    os.remove(os.path.join("server_data", filename))
                self.send_global_weights(connection, 'server_data/GlobalModel.npy.pkl', str(self.modelid))
        finally:
            mutex.release()

    def send_global_weights(self, connection, filepath, data='-1'):
        filesize = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        connection.send(f"{REPLY_GLOBAL_MSG}{SEPARATOR}{filename}{SEPARATOR}{filesize}{SEPARATOR}{data}".encode())
        time.sleep(1)
        print('Preparing to send file...')
        # start sending the file
        progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
        with open(filepath, "rb") as f:
            for _ in progress:
                # read the bytes from the file
                bytes_read = f.read(BUFFER_SIZE)
                if not bytes_read:
                    # file transmitting is done
                    break
                # we use sendall to assure transimission in
                # busy networks
                connection.sendall(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))
        # close the socket
        connection.close()

    def WriteClientWeights(self, weights, clientno, roundno, instance_count):
        # print("Weights : ", weights)
        print("Client no : ", clientno)
        print("Round no : ", roundno)
        print("Instance count: ", instance_count)
        outfile = "server_data" + "/" + str(clientno)+"_" + str(roundno)+"_" + str(instance_count) + '.npy'
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        mutex.acquire()
        try:
            np.save(outfile, weights)
        finally:
            mutex.release()



    def accept(self):
        """
        Runs the server
        """
        connection, client_address = self.server_socket.accept()
        print("Client %s, %s connected" % client_address)
        return connection, client_address

    def run(self):
        """
        Given  host and port, bounds the socket and runs the server
        """
        self.bind_socket()
        while True:
            connection, client_address = self.accept()
            print("Accepted connection!")
            t = threading.Thread(target=self.HandleClientRequest, args=(connection,))
            t.start()



def main():
    """
    Main function,creates a new server
    """
    server = Server('127.0.0.1', 10000, -1)
    server.run()


if __name__ == '__main__':
    """Entry point"""

    main()

