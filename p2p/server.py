import socket
s=socket.socket()
host= socket.gethostname()
port = 8080
s.bind((host,port))
s.listen(1)
print("Waiting for 2 connections..")
conn,addr = s.accept()
print("Client 1 has connected")
conn.send("Welcome to server".encode())
print("Waiting for 1 connection")
conn1, addr1 = s.accept()
print("Client 2 has connected")
conn1.send("Welcome to the server".encode())

while 1:
	message = input(str(">>"))
	message = message.encode()
	conn.send(message)
	conn1.send(message)
	print("Message sent")
	recv_message = conn.recv(1024)
	print("client :", recv_message.decode())
	conn1.send(recv_message)
	recv_message = conn1.recv(1024)
	print("client:", recv_message.decode())
	conn.send(recv_message)




