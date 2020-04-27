#######################################################################################################################
# AVANS - BLOCKCHAIN - MINOR MAD                                                                                      #
#                                                                                                                     #
# Author: Maurice Snoeren                                                                                             #
# Version: 0.1 beta (use at your own risk)                                                                            #
#                                                                                                                     #
# Example python script to show the working principle of the TcpServerNode Node class.                                #
#######################################################################################################################

import time
import pprint
from TcpServerNode import Node
from MyPeer2PeerNode import MyPeer2PeerNode

node_p2p1 = MyPeer2PeerNode('localhost', 1000, None, 0, "./data/0")
node_p2p2 = MyPeer2PeerNode('localhost', 2000, None, 1, "./data/1")
node_p2p3 = MyPeer2PeerNode('localhost', 3000, None, 2, "./data/2")

node_p2p1.start()
node_p2p2.start()
node_p2p3.start()
node_p2p4.start()

node_p2p1.connect_with_node('localhost', 2000)
node_p2p1.connect_with_node('localhost', 3000)

node_p2p2.connect_with_node('localhost', 1000)
node_p2p2.connect_with_node('localhost', 3000)

node_p2p3.connect_with_node('localhost', 1000)
node_p2p3.connect_with_node('localhost', 2000)

while True:
    node_p2p1.receive_from_random_nodes()
    time.sleep(5)

print("main stopped")

