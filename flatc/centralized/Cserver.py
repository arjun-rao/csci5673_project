# contains SocketServer,SocketClient classes
import socket

class SocketServer():

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connectToHost(self, host, port):
        self.sock.connect((host, port))
  
    def Listen(self,msg='Accepted Connection from:'):
  
  
    def Receive(self,size=1024):
		'receives the local weights of clients'

    def processWeights(self,client_weights):
		'Receives client weights, processes and then returns json_data'	
		return Json_data
	
    def Sendweights(self,Json_data):
		'server sends the processed weights to the client'






