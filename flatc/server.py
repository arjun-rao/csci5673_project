import socket
import threading
import pickle
import numpy as np
import os
from threading import Thread, Lock
from model import Model

mutex = Lock()

class Message:
    def __init__(self, id, modelid, weights, clientno, roundno, instance_count=0):
        self.id = id
        self.modelid = modelid
        self.weights = weights
        self.clientno = clientno
        self.roundno = roundno
        self.instance_count = instance_count

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
        data = []
        while True:
            packet = connection.recv(10000)
            print('receiving...')
            if not packet: break
            data.append(packet)
        print("Weights received!")
        msg = pickle.loads(b"".join(data))
        if msg.id==0:
            self.WriteClientWeights(msg.weights, msg.clientno, msg.roundno, msg.instance_count)
        else:
            self.ReturnUpdatedWeights(connection, msg.modelid)

    def ReturnUpdatedWeights(self, connection, modelid):
        if self.modelid==-1 or self.modelid==modelid:
            #no model update
            connection.send('')
            return
        mutex.acquire()
        try:
            if len(os.listdir("server_data")) == 0:
                globalWeights = np.load("GlobalModel.npy")
                globalWeightsPkl = pickle.dumps(globalWeights)
                connection.send(globalWeightsPkl)
            else:
                weights_list = []
                instance_counts = []
                for filename in os.listdir("server_data"):
                    if filename.endswith(".npy"):
                        weights = np.load(filename)
                        instance_counts.append(int(filename.split("_")[2]))
                        weights_list.append(weights)
                updatedWeights = process_weights(weights_list, instance_counts)
                updatedWeightsPkl = pickle.dumps(updatedWeights)
                connection.send(updatedWeightsPkl)
                for filename in os.listdir("server_data"):
                    os.remove(filename)
                os.remove("GlobalModel.npy")
                self.modelid += 1
                np.save("GlobalModel.npy", updatedWeights)
        finally:
            mutex.release()

    def WriteClientWeights(self, weights, clientno, roundno, instance_count):
        print("Weights : ", weights)
        print("Client no : ", clientno)
        print("Round no : ", roundno)
        print("Instance count: ", instance_count)
        outfile = "server_data" + "/" + str(clientno)+"_" + str(roundno)+"_" + str(instance_count)
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

