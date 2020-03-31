import socket
import argparse
import pickle
import numpy as np

class Message:
	def __init__(self, id, modelid, weights, clientno, roundno, instance_count=0):
		self.id = id
		self.modelid = modelid
		self.weights = weights
		self.clientno = clientno
		self.roundno = roundno
		self.instance_count = instance_count

class Client:

	"""Defines client"""

	def __init__(self, host, port):
		"""
		Initializes a new Client
		:param host: Server's host address
		:param port: Server's port
		"""

		self.host = host
		self.port = port

	def connection(self):
		"""
		Connect to server
		"""
		self.ClientSocket = socket.socket()
		self.ClientSocket.connect((self.host,self.port))

	def compute_new_model(self, weights):
		pass

	def get_updatedweights(self, modelid):
		"""
		Client receives the updates weights from the server
		"""
		msg = Message(1, modelid, None, None, None)
		msg_pkl = pickle.dumps(msg)
		self.ClientSocket.send(msg_pkl)
		Response = self.ClientSocket.recv(1024)
		compute_new_model(Response.decode('utf-8'))


	def send_weights(self, weights, clientno, roundno, instance_count): #numpy
		"""
		Client sends his local weights using this function, we can modify it
		"""
		msg = Message(0, -1, weights, clientno, roundno, instance_count)
		msg_pkl = pickle.dumps(msg)
		self.ClientSocket.send(msg_pkl)
		# while True:
		# 	self.receive_updatedweights()


	def run(self):
		self.connection()
		self.send_weights(np.array([1,2,3]), 1, 2, 100)
		self.get_updatedweights(0)


def main():
	client = Client('127.0.0.1', 10000)
	client.run()


if __name__ == '__main__':
    """Entry point"""

    main()






