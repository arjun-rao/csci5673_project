import socket
s = socket.socket()
host = socket.gethostname()
port = 8080
s.connect((host,port))
print("Connected to server")
message = s.recv(1024)
message = message.decode()
print("server message:",message)
while 1:
	message = s.recv(1024)
	message = message.decode()
	print("server:", message)
	message = s.recv(1024)
	message = message.decode()
	print("server:", message)
	new_message = input(str(">>"))
	new_message = new_message.encode()
	s.send(new_message)
	print("message sent")

