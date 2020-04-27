#######################################################################################################################
# AVANS - BLOCKCHAIN - MINOR MAD                                                                                      #
#                                                                                                                     #
# Author: Maurice Snoeren                                                                                             #
# Version: 0.1 beta (use at your own risk)                                                                            #
#                                                                                                                     #
# TcpServerNode implements a peer-to-peer network node based on tcp/ip sockets.                                       #
# TcpServerNode creates a TCP/IP server on the port you have given. It accepts incoming nodes and put these into its  #
# internal datastructure. When nodes disconnect, the nodes are removed. Events are generated when nodes are connected #
# , when nodes leave and when nodes have data. Furthermore, this class is able to connect to other nodes. Sending     #
# data to these nodes is easy as well. The datastructure is up to you and how you implement the protocol to form the  #
# decentralized peer-to-peer network. This class is at you disposal to use within your code to speed up the           #
# development.                                                                                                        #
#######################################################################################################################

import socket
import sys
import json
import time
import threading
import random
import hashlib
import numpy as np
import traceback

import os
import time
import argparse
import numpy as np
from dataset_final import get_round_data, get_data
from message import Message, CLIENT_WEIGHT_UPDATE, REQUEST_PEER_MSG, REPLY_PEER_MSG
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import tqdm
from model import *
import pickle
#######################################################################################################################
# TCPServer Class #####################################################################################################
#######################################################################################################################

SEPARATOR = "<SEP>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step

# Class Node
# Implements a node that is able to connect to other nodes and is able to accept connections from other nodes.
# After instantiation, the node creates a TCP/IP server with the given port.
#
class Node(threading.Thread):

    # Python class constructor
    def __init__(self, host, port, callback, peer_no, data_dir):
        super(Node, self).__init__()

        # When this flag is set, the node will stop and close
        self.terminate_flag = threading.Event()

        # Server details, host (or ip) to bind to and the port
        self.host = host
        self.port = port
        self.peer_no = peer_no
        self.round_no = -1
        self.model = None
        # self.model_id = -1 not required
        self.data_dir = os.path.join(data_dir, str(peer_no))
        os.makedirs(self.data_dir, exist_ok=True)

        # Events are send back to the given callback
        self.callback = callback

        # Nodes that have established a connection with this node
        self.nodesIn = []  # Nodes that are connect with us N->(US)->N

        # Nodes that this nodes is connected to
        self.nodesOut = []  # Nodes that we are connected to (US)->N

        # Create a unique ID for each node.
        id = hashlib.md5()
        t = self.host + str(self.port) + str(random.randint(1, 99999999))
        id.update(t.encode('ascii'))
        self.id = id.hexdigest()

        # Start the TCP/IP server
        self.init_server()

        self.message_count_send = 0;
        self.message_count_recv = 0;

        # Debugging on or off!
        self.debug = False

        # For visuals!!
        self.visuals = False

    #------------------------------------------------------

    def init(self, client_x, client_y):
        self.model = Model(client_x, client_y)
        self.round_no += 1
    
    def compute_new_model(self, weights):
        self.model.update_weights_average(weights)

    def train_round(self, round_x, round_y):
        self.model.train(round_x, round_y)
        self.round_no += 1

    #----------------------------------------------------------------

    def get_message_count_send(self):
        return self.message_count_send

    def get_message_count_recv(self):
        return self.message_count_recv

    def enable_visuals(self):
        self.visuals = True
        self.udp_server = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        self.send_visuals("node-new", { "host": self.host, "port": self.port })

    def send_visuals(self, type, data):
        if ( self.visuals ):
            data["__id"]        = self.get_id()
            data["__host"]      = self.host;
            data["__port"]      = self.port;
            data["__node"]      = type
            data["__timestamp"] = time.time()
            
            message = json.dumps(data, separators=(',', ':'));
            self.dprint("Visuals sending: " + message)
            #self.udp_server.sendto(message, ('92.222.168.248', 15000))
            self.udp_server.sendto(message, ('dev.codingskills.nl', 15000))
            self.udp_server.sendto(message, ('codingskills.nl', 15000))

            del data["__id"]
            del data["__host"]
            del data["__port"]
            del data["__node"]
            del data["__timestamp"]

    def enable_debug(self):
        self.debug = True

    def dprint(self, message):
        if ( self.debug ):
            print("DPRINT: " + message)

    # Creates the TCP/IP socket and bind is to the ip and port
    def init_server(self):
        print("Initialisation of the TcpServer on port: " + str(self.port) + " on node (" + self.id + ")")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.bind((self.host, self.port))
        self.sock.bind(('', self.port))
        self.sock.settimeout(10.0)
        self.sock.listen(1)

    # Print the nodes with this node is connected to. It makes two lists. One for the nodes that have established
    # a connection with this node and one for the node that this node has made connection with.
    def print_connections(self):
        print("Connection status:")
        print("- Total nodes connected with us: %d" % len(self.nodesIn))
        print("- Total nodes connected to     : %d" % len(self.nodesOut))

    def get_inbound_nodes(self):
        return self.nodesIn

    def get_outbound_nodes(self):
        return self.nodesOut

    def get_id(self):
        return self.id

    def get_host(self):
        return self.host

    def get_port(self):
        return self.port

    # Misleading function name, while this function checks whether the connected nodes have been terminated
    # by the other host. If so, clean the array list of the nodes.
    # When a connection is closed, an event is send NODEINBOUNDCLOSED or NODEOUTBOUNDCLOSED
    def delete_closed_connections(self):
        for n in self.nodesIn:
            if n.terminate_flag.is_set():
                self.event_node_inbound_closed(n)

                if ( self.callback != None ):
                    self.callback("NODEINBOUNDCLOSED", self, n, {})

                n.join()
                del self.nodesIn[self.nodesIn.index(n)]

        for n in self.nodesOut:
            if n.terminate_flag.is_set():
                self.event_node_outbound_closed(n)

                if ( self.callback != None ):
                    self.callback("NODEOUTBOUNDCLOSED", self, n, {})

                n.join()
                del self.nodesOut[self.nodesIn.index(n)]

    def create_message(self, data):
        self.message_count_send = self.message_count_send + 1
        data['_mcs'] = self.message_count_send
        data['_mcr'] = self.get_message_count_recv()

        return data;

    #-----------------------------------------------------

    def receive_from_n_random_nodes(self, n):
        rand_indices=np.random.choice(len(self.nodesOut), size=n, replace=False)
        for index in rand_indices:
            self.receive_from_node(self.nodesOut[index]) 

    def receive_from_random_nodes(self):
        num_rand_nodes=random.randint(1,len(self.nodesOut))
        rand_indices=np.random.choice(len(self.nodesOut), size=num_rand_nodes, replace=False)
        for index in rand_indices:
            self.receive_from_node(self.nodesOut[index])       

    def receive_from_node(self, conn):
        """
        Receives weights
        """
        conn.sock.send(f"{REQUEST_PEER_MSG}".encode())
        received = conn.sock.recv(100).decode()
        if received == '':
            return False
        msg = received.split(SEPARATOR)
        if msg[0] == REPLY_PEER_MSG:
            filename, filesize = os.path.basename(msg[1]), int(msg[2])
            progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)

            with open(os.path.join(self.data_dir, filename), "wb") as f:
                for _ in progress:
                    # read 1024 bytes from the socket (receive)
                    bytes_read = conn.sock.recv(BUFFER_SIZE)
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
            return True
        return False

#---------------------------------------------------------------------

    # Send a message to all the nodes that are connected with this node.
    # data is a python variable which is converted to JSON that is send over to the other node.
    # exclude list gives all the nodes to which this data should not be sent.
    def send_to_nodes(self, data, exclude = []):
        for n in self.nodesIn:
            if n in exclude:
                self.dprint("TcpServer.send2nodes: Excluding node in sending the message")
            else:
                self.send_to_node(n, data)

        for n in self.nodesOut:
            if n in exclude:
                self.dprint("TcpServer.send2nodes: Excluding node in sending the message")
            else:
                self.send_to_node(n, data)

    # Send the data to the node n if it exists.
    # data is a python variabele which is converted to JSON that is send over to the other node.
    def send_to_node(self, n, data):
        self.delete_closed_connections()
        if n in self.nodesIn or n in self.nodesOut:
            try:
                n.send(self.create_message( data ))
                
            except Exception as e:
                self.dprint("TcpServer.send2node: Error while sending data to the node (" + str(e) + ")");
        else:
            self.dprint("TcpServer.send2node: Could not send the data, node is not found!")

    # Make a connection with another node that is running on host with port.
    # When the connection is made, an event is triggered CONNECTEDWITHNODE.
    def connect_with_node(self, host, port):
        print("connect_with_node(" + host + ", " + str(port) + ")")
        if ( host == self.host and port == self.port ):
            print("connect_with_node: Cannot connect with yourself!!")
            return;

        # Check if node is already connected with this node!
        for node in self.nodesOut:
            if ( node.get_host() == host and node.get_port() == port ):
                print("connect_with_node: Already connected with this node.")
                return True
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dprint("connecting to %s port %s" % (host, port))
            sock.connect((host, port))

            #thread_client = NodeConnection(self, sock, (host, port), self.callback)
            thread_client = self.create_new_connection(sock, (host, port), self.callback)
            thread_client.start()
            self.nodesOut.append(thread_client)
            self.event_connected_with_node(thread_client)

            if (self.callback != None):
                self.callback("CONNECTEDWITHNODE", self, thread_client, {})

            self.print_connections()

        except Exception as e:
            self.dprint("TcpServer.connect_with_node: Could not connect with node. (" + str(e) + ")")

    # Disconnect with a node. It sends a last message to the node!
    def disconnect_with_node(self, node):
        if node in self.nodesOut:
            node.send(self.create_message( {"type": "message", "message": "Terminate connection"} ))
            node.stop()
            node.join() # When this is here, the application is waiting and waiting
            del self.nodesOut[self.nodesOut.index(node)]

    # When this function is executed, the thread will stop!
    def stop(self):
        self.terminate_flag.set()

    # This method can be overrided when a different nodeconnection is required!
    def create_new_connection(self, connection, client_address, callback):
        return NodeConnection(self, connection, client_address, callback)

    # This method is required for the Thead function and is called when it is started.
    # This function implements the main loop of this thread.
    def run(self):
        while not self.terminate_flag.is_set():  # Check whether the thread needs to be closed
            try:
                self.dprint("TcpServerNode: Wait for incoming connection")
                connection, client_address = self.sock.accept()

                # TODO: Startup first communication in which the details of the node is communicated
                # TODO: 
                
                thread_client = self.create_new_connection(connection, client_address, self.callback)
                thread_client.start()
                self.nodesIn.append(thread_client)

                self.event_node_connected(thread_client)


                if ( self.callback != None ):
                    self.callback("NODECONNECTED", self, thread_client, {})

            except socket.timeout:
                pass

            except:
                raise

            time.sleep(0.01)

        print("TcpServer stopping...")
        for t in self.nodesIn:
            t.stop()

        for t in self.nodesOut:
            t.stop()

        time.sleep(1)

        for t in self.nodesIn:
            t.join()

        for t in self.nodesOut:
            t.join()

        self.sock.close()
        print("TcpServer stopped")

        # For visuals!
        self.send_visuals("node-closed", { "host": self.host, "port": self.port}) 

    # Started to implement the events, so this class can be extended with a better class
    # In the event a callback can be called!

    # node is the node thread that is running to get information and send information to.
    def event_node_connected(self, node):
        self.dprint("event_node_connected: " + node.getName())

        # For visuals!
        self.send_visuals("node-connection-from", { "host": node.host, "port": node.port })
        
    def event_connected_with_node(self, node):
        self.dprint("event_node_connected: " + node.getName())

        # For visuals!
        self.send_visuals("node-connection-to", { "host": node.host, "port": node.port })

    def event_node_inbound_closed(self, node):
        self.dprint("event_node_inbound_closed: " + node.getName())

        # For visuals!
        self.send_visuals("node-connection-from-closed", { "host": node.host, "port": node.port })

    def event_node_outbound_closed(self, node):
        self.dprint("event_node_outbound_closed: " + node.getName())

        # For visuals!
        self.send_visuals("node-connection-to-closed", { "host": node.host, "port": node.port })

    def event_node_message(self, node, data):
        self.dprint("event_node_message: " + node.getName() + ": " + str(data))


#######################################################################################################################
# NodeConnection Class ###############################################################################################
#######################################################################################################################

# Class NodeConnection
# Implements the connection that is made with a node.
# Both inbound and outbound nodes are created with this class.
# Events are send when data is coming from the node
# Messages could be sent to this node.
class NodeConnection(threading.Thread):

    # Python constructor
    def __init__(self, nodeServer, sock, clientAddress, callback):
        super(NodeConnection, self).__init__()

        self.host = clientAddress[0]
        self.port = clientAddress[1]
        self.nodeServer = nodeServer
        self.sock = sock
        self.clientAddress = clientAddress
        self.callback = callback
        self.terminate_flag = threading.Event()

        # Variable for parsing the incoming json messages
        self.buffer = ""

        id = hashlib.md5()
        t = self.host + str(self.port) + str(random.randint(1, 99999999))
        id.update(t.encode('ascii'))
        self.id = id.hexdigest()

        self.nodeServer.dprint("NodeConnection.send: Started with client (" + self.id + ") '" + self.host + ":" + str(self.port) + "'")        

    def get_host(self):
        return self.host

    def get_port(self):
        return self.port
    
    # Send data to the node. The data should be a python variable
    # This data is converted into json and send.
    def send(self, data):
        #data = self.create_message(data) # Call it yourself!!

        try:
            message = json.dumps(data, separators=(',', ':')) + "-TSN";
            self.sock.sendall(message.encode('utf-8'))

            # For visuals!
            self.nodeServer.send_visuals("node-send", data)

        except:
            self.nodeServer.dprint("NodeConnection.send: Unexpected error:", sys.exc_info()[0])
            self.terminate_flag.set()

    def check_message(self, data):
        return True

    def get_id(self):
        return self.id

    # Stop the node client. Please make sure you join the thread.
    def stop(self):
        self.terminate_flag.set()        

    # Required to implement the Thread. This is the main loop of the node client.
    def run(self):

        # Timeout, so the socket can be closed when it is dead!
        self.sock.settimeout(10.0)
        #----------------------------------------------------------
        while not self.terminate_flag.is_set(): # Check whether the thread needs to be closed
            line = ""
            try:
                line = self.sock.recv(50)      
            except socket.timeout:
                pass

            except:
                self.terminate_flag.set()
                self.nodeServer.dprint("NodeConnection: Socket has been terminated (%s)" % line)

            if line != "":
                try:
                    line = line.decode('utf-8')
                except:
                    print("NodeConnection: Decoding line error")
                msg = line.split(SEPARATOR)
                if msg[0]==REQUEST_PEER_MSG:
                    weigths = self.nodeServer.model.get_weights()
                    filepath = os.path.join(self.nodeServer.data_dir, '{}_{}.pkl'.format(self.nodeServer.client_no, self.nodeServer.round_no))
                    msg = Message(REPLY_PEER_MSG, weights, self.nodeServer.peer_no, self.nodeServer.round_no, self.nodeServer.get_instance_count())
                    with open(filepath, 'wb') as f:
                        msg_pkl = pickle.dump(msg, f)
                    filesize = os.path.getsize(filepath)
                    filename = os.path.basename(filepath)
                    self.sock.send(f"{REPLY_PEER_MSG}{SEPARATOR}{filename}{SEPARATOR}{filesize}".encode())
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
                            self.sock.sendall(bytes_read)
                            # update the progress bar
                            progress.update(len(bytes_read))
        #----------------------------------------------------------------------        
                # index = self.buffer.find("-TSN")
                # while ( index > 0 ):
                #     message = self.buffer[0:index]
                #     self.buffer = self.buffer[index+4::]

                #     try:
                #         data = json.loads(message)
                        
                #     except Exception as e:
                #         print("NodeConnection: Data could not be parsed (%s) (%s)" % (line, str(e)) )

                #     if ( self.check_message(data) ):
                #         self.nodeServer.message_count_recv = self.nodeServer.message_count_recv + 1
                #         self.nodeServer.event_node_message(self, data)
                        
                #         if (self.callback != None):
                #             self.callback("NODEMESSAGE", self.nodeServer, self, data)

                #         # For visuals!
                #         self.nodeServer.send_visuals("node-receive", data)


                #     else:
                #         self.nodeServer.send_visuals("node-error", {"error": "failed check"})
                #         self.nodeServer.dprint("-------------------------------------------")
                #         self.nodeServer.dprint("Message is damaged and not correct:\nMESSAGE:")
                #         self.nodeServer.dprint(message)
                #         self.nodeServer.dprint("DATA:")
                #         self.nodeServer.dprint(str(data))
                #         self.nodeServer.dprint("-------------------------------------------")

                #     index = self.buffer.find("-TSN")

            time.sleep(0.01)

        self.sock.settimeout(None)
        self.sock.close()
        self.nodeServer.dprint("NodeConnection: Stopped")

#######################################################################################################################
# Example usage of Node ###############################################################################################
#######################################################################################################################
#
# from TcpServerNode import Node
#
# node = None # global variable
#
# def callbackNodeEvent(event, node, other, data):
#    print("Event Node 1 (" + node.id + "): %s: %s" % (event, data))
#    node.send2nodes({"thank": "you"})
#
# node = Node('localhost', 10000, callbackNodeEvent)
#
# node.start()
#
# node.connect_with_node('12.34.56.78', 20000)
#
# server.terminate_flag.set() # Stopping the thread
#
# node.send2nodes({"type": "message", "message": "test"})
#
# while ( 1 ):
#    time.sleep(1)
#
# node.stop()
#
# node.join()
#
#
# END OF FILE

if __name__ == '__main__':
    """Entry point"""
    # client = main(0)
    test_x, test_y = get_data('data/test.csv')

    def evaluate(model_i):
    # print(classification_report(test_y, model_i.predict(test_x)))
    # print('F1, ROC: {}, {}:'.format(f1_score(test_y, model_i.predict(test_x)),
    #     roc_auc_score(test_y, model_i.predict_proba(test_x))))
        return {
                    'f1': f1_score(test_y, model_i.predict(test_x)),
                    'roc': roc_auc_score(test_y, model_i.predict_proba(test_x))
        }

    # client = create_client(1)
    # client_x, client_y = get_round_data('./data/1', 0)
    # client.init(client_x, client_y)
    # evaluate(client.model)


    parser = argparse.ArgumentParser(description='Start a peer')
    parser.add_argument("--id",type=int, dest="id", default=1,
                        help="ID for peer. Default is 1.")
    parser.add_argument("--host", dest="host", default="0.0.0.0",
                        help="Host IP for peer. Default is '0.0.0.0'.")
    parser.add_argument("--port", type=int, dest="port", default=10000,
                        help="Host Port for peer. Default is '10000'.")
    parser.add_argument("--data_dir", dest="data_dir", default='peer_data/',
                        help="peer Data Dir to store weights. Default is 'peer_data/'.")
    parser.add_argument("--train_data_dir", dest="train_data_dir", default='data/',
                        help="Peer Data Dir to store train data. Default is 'data/'.")
    args = parser.parse_args()
    host = args.host
    port = args.port

    print("Type 'help' to see possible options")
    helpstr= """
    Possible commands:
        * evaluate - run the current model's evaluation
        * train_round <round_no> - train the model for given round number
        * send_weights  - send weights to param server
        * receive_global - receive global update for weights from param server
        * exit - to quit
    """

    peer = Node(host, port, None, args.id, args.data_dir)
    peer_x, peer_y = get_round_data(os.path.join(args.train_data_dir, f'{args.id}'), 0)
    peer.init(peer_x, peer_y)
    peer.start()
    print("Peer created with model for round 0. Type evaluate to see performance.")



    while True:
        msg = input('> ').split(' ')
        if msg[0] == 'exit':
            break
        elif msg[0] == 'help':
            print(helpstr)
        elif msg[0] =='evaluate':
            print(evaluate(peer.model))
        elif msg[0] == 'random_update':
            peer.receive_from_random_nodes()
        elif msg[0] == 'random_update_n':
            peer.receive_from_n_random_nodes(int(msg[1]))
        elif msg[0] == 'connect':
            peer.connect_with_node(msg[1], int(msg[2]))
        elif msg[0] == 'get_weights':
            print(peer.model.get_weights())
        elif msg[0] == 'train_round':
            client_x, client_y = get_round_data(os.path.join(args.train_data_dir, f'{args.id}'), int(msg[1]))
            client.train_round(client_x, client_y)
