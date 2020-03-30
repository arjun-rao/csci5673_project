import socket

class Server:

	"""Defines the server"""

	MAX_WAITING_CONNECTIONS = 5
		

	def __init__(self, host, port):
		"""
		Initializes a new Server
		:param host: Host on which server bounds
		:param port: port on which server bound
		"""
		
		#threading.Thread.__init__(self)
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


	def receive(self,connection):
		"""
		Receives incoming message/updated weights from the client
		"""
		msg = 1
		while msg:
			msg = connection.recv(10)
			print("received data: ",msg)
			algo_output = self.process_weights(msg)
			self.send(connection,algo_output)
		return 


	def _run(self):
		"""
		Runs the server
		"""
		connection, client_address = self.server_socket.accept()
		print("Client %s, %s connected" % client_address)
		return connection


	def run(self):
		"""
		Given  host and port, bounds the socket and runs the server
		"""
		self.bind_socket()
		connection = self._run()
		self.receive(connection)
		self.send(connection,'hey tan')

def main():
	"""
	Main function,creates a new server
	"""
	server = Server('127.0.0.1', 10000) 
	server.run()


if __name__ == '__main__':
	"""Entry point"""
	
	main()
	
