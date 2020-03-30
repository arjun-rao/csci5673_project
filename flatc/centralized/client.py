import socket
import argparse

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

	def receive_updatedweights(self):
		"""
		Client receives the updates weights from the server
		"""
		Response = self.ClientSocket.recv(1024)
		compute_new_model(Response.decode('utf-8'))


	def send_message(self):
		"""
		Client sends his local weights using this function, we can modify it
		"""
		while True:
			Input = input('Send local weights to server, type something:')
			self.ClientSocket.send(str.encode(Input))
			self.receive_updatedweights()


	def run(self):
		self.connection()
		self.send_message()
		#self.receive_updatedweights()


def main():
	client = Client('127.0.0.1', 10000)
	client.run()


if __name__ == '__main__':
    """Entry point"""

    main()






