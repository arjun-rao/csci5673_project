import os
import socket
import time
import argparse
import pickle
import numpy as np
from model import *
from dataset_final import get_round_data, get_data
from message import Message, CLIENT_WEIGHT_UPDATE, REQUEST_GLOBAL_MSG, REPLY_GLOBAL_MSG
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import tqdm

SEPARATOR = "<SEP>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step

class Client:

    """Defines client"""

    def __init__(self, host, port, client_no, data_dir):
        """
        Initializes a new Client
        :param host: Server's host address
        :param port: Server's port
        """

        self.host = host
        self.port = port
        self.client_no = client_no
        self.round_no = -1
        self.model = None
        self.model_id = -1
        self.data_dir = os.path.join(data_dir, str(client_no))
        os.makedirs(self.data_dir, exist_ok=True)

    def init(self, client_x, client_y):
        self.model = Model(client_x, client_y)
        self.round_no += 1

    def connection(self):
        """
        Connect to server
        """
        self.ClientSocket = socket.socket()
        try:
            self.ClientSocket.connect((self.host,self.port))
        except Exception:
            print('Failed to connect to server...')


    def compute_new_model(self, weights):
        self.model.update_weights_average(weights)

    def get_updatedweights(self, modelid):
        """
        Client receives the updates weights from the server
        """
        self.connect()
        self.ClientSocket.send(f"{REQUEST_GLOBAL_MSG}{SEPARATOR}{modelid}".encode())
        received = self.ClientSocket.recv(100).decode()
        msg = received.split(SEPARATOR)
        if msg[0] == REPLY_GLOBAL_MSG:
            filename, filesize, modelid = os.path.basename(msg[1]), int(msg[2]), int(msg[3])
            self.model_id = modelid
            progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)

            with open(os.path.join(self.data_dir, filename), "wb") as f:
                for _ in progress:
                    # read 1024 bytes from the socket (receive)
                    bytes_read = self.ClientSocket.recv(BUFFER_SIZE)
                    if not bytes_read:
                        # nothing is received
                        # file transmitting is done
                        break
                    # write to the file the bytes we just received
                    f.write(bytes_read)
                    # update the progress bar
                    progress.update(len(bytes_read))
            print("msg received received!")
            self.compute_new_model(np.load(os.path.join(self.data_dir, filename), allow_pickle=True))

    def train_round(self, round_x, round_y):
        self.model.train(round_x, round_y)


    def send_file(self, filepath):
        filesize = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        self.connect()
        self.ClientSocket.send(f"{CLIENT_WEIGHT_UPDATE}{SEPARATOR}{filename}{SEPARATOR}{filesize}".encode())
        print('Preparing to send file...')
        time.sleep(1)
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
                self.ClientSocket.sendall(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))
        # close the socket
        self.ClientSocket.close()

    def _send_weights(self, weights, clientno, roundno, instance_count): #numpy
        """
        Client sends his local weights using this function, we can modify it
        """
        msg = Message(CLIENT_WEIGHT_UPDATE, self.model_id, weights, clientno, roundno, instance_count)
        fname = os.path.join(self.data_dir, '{}_{}.pkl'.format(clientno, roundno))
        with open(fname, 'wb') as f:
            msg_pkl = pickle.dump(msg, f)
        self.send_file(fname)
        # while True:
        # 	self.receive_updatedweights()

    def send_weights(self):
        weigths = self.model.get_weights()
        self._send_weights(weigths, self.client_no, self.round_no, self.model.get_instance_count())

    def connect(self):
        self.connection()


def create_client(id):
    client = Client('127.0.0.1', 10000, id, './client_data/')
    return client


if __name__ == '__main__':
    """Entry point"""
    # client = main(0)
    test_x, test_y = get_data('./data/test.csv')

    def evaluate(model_i):
    # print(classification_report(test_y, model_i.predict(test_x)))
    # print('F1, ROC: {}, {}:'.format(f1_score(test_y, model_i.predict(test_x)),
    #     roc_auc_score(test_y, model_i.predict_proba(test_x))))
        return {
                    'f1': f1_score(test_y, model_i.predict(test_x)),
                    'roc': roc_auc_score(test_y, model_i.predict_proba(test_x))
        }

    client = create_client(1)
    client_x, client_y = get_round_data('./data/1', 0)
    client.init(client_x, client_y)
    evaluate(client.model)







