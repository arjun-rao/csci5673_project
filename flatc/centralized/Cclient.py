# contains SocketServer,SocketClient classes
import socket



class SocketClient():

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))
  		'connect to the server'

    def Send(self,json_data):
        'sends json object having the local weights to the server' 
  
    def Receive(self):
        'receives the processes weights from the server'


