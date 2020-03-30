import socket
import threading
import pickle
import numpy as np
import os
from threading import Thread, Lock

mutex = Lock()

class Message:
	def __init__(self, id, modelid, weights, clientno, roundno):
		self.id = id
		self.modelid = modelid
		self.weights = weights
		self.clientno = clientno
		self.roundno = roundno

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


	def process_weights(self, weights):
		"""Algorith to process weights"""
		processed = weights + str.encode(' got processedweights')
		return processed

	def HandleClientRequest(self, connection):
		msg_pkl = connection.recv(1024)
		print("Weights received!")
		msg = pickle.loads(msg_pkl)
		if msg.id==0:
			self.WriteClientWeights(msg.weights, msg.clientno, msg.roundno)
		else:
			self.ReturnUpdatedWeights(connection, msg.modelid)

	def ReturnUpdatedWeights(self, connection, modelid):
		if self.modelid==-1 or self.modelid==modelid:
			msg = Message(-1, -1, None, -1, -1) #no model or no update
			connection.send(msg)
		mutex.acquire()
		try:
			if len(dir) == 0:
				globalWeights = np.load("GlobalModel.npy")
				globalWeightsPkl = pickle.dumps(globalWeights)
				connections.send(globalWeightsPkl)
			else:
				weights_list = []
				for filename in os.listdir("data"):
					if filename.endswith(".npy"):
						weights = np.load(filename)
						weights_list.append(weights)
				globalWeights = None
				if modelid!=-1:
					globalWeights = np.load("GlobalModel.npy")
         		#updatedWeights = process_weights(weights_list, globalWeights) if first model, then globalWeights will be None
				updatedWeights = np.array([0,0,0,0,0])
				updatedWeightsPkl = pickle.dumps(updatedWeights)
				connections.send(updatedWeightsPkl)
				for filename in os.listdir("data"):
					os.remove(filename)
				os.remove("GlobalModel.npy")
				np.save("GlobalModel.npy", updatedWeights)
		finally:
			mutex.release()

	def WriteClientWeights(self, weights, clientno, roundno): 		
		print("Weights : ", weights)
		print("Client no : ", clientno)
		print("Round no : ", roundno)
		outfile = "data" + "/" + str(clientno)+"_"+str(roundno)
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

