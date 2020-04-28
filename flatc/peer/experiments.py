from peer import *
import sys
import os
from dataset_final import get_round_data, get_data
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import time


output = 'experiment_6.csv'
train_data_dir = 'data/c5_r5'
peer_data_dir = 'peer_data/'
start_port = 9000
max_peers = 5
rounds = 4 # starts from 0. so 4 => 5 rounds.

test_x, test_y = get_data(f'{train_data_dir}/test.csv')

def evaluate(client, fname='results.csv'):
    data = {
        'client_no': client.peer_no,
        'round_no': client.round_no,
        'instance_count': client.model.get_instance_count(),
        'f1': np.around(f1_score(test_y, client.model.predict(test_x)), 2),
        'roc': np.around(roc_auc_score(test_y, client.model.predict_proba(test_x)), 2),
        'messages_sent': client.message_count_send,
        'messages_recv': client.message_count_recv
    }
    if fname != '':
        df = pd.DataFrame([data], columns=['client_no', 'round_no', 'instance_count', 'f1', 'roc', 'messages_sent', 'messages_recv'])
        with open(fname, 'a') as f:
            df.to_csv(fname,index=False, mode='a', header=f.tell()==0)
    return data

peers = []

def reconnect_peers(peers, max_peers):
    # Connect peers to each other
    for i in range(max_peers):
        print(f'Connecting nodes to peer: {i}')
        for j in range(max_peers):
            if i == j:
                continue
            peers[i].connect_with_node("localhost", start_port + j)
            time.sleep(0.5)
        time.sleep(1)

def init_peer(peer):
    peer_x, peer_y = get_round_data(os.path.join(train_data_dir, f'{peer.peer_no}'), 0)
    peer.init(peer_x, peer_y)
    peer.start()
    print(f"Peer {peer.peer_no} created with model for round 0.")


def exchange_weights(peers, max_peers):
    # For each peer exchange weights
    for i in range(max_peers):
        peer = peers[i]
        print(f'Receiving weights to peer: {i}')
        # Change this for different experiments
        # all_ports = [peers[j].port for j in range(max_peers) if j != i]
        # ports = list(np.random.choice(all_ports, size=max_peers-1, replace=False))

        # For ring topology
        all_ports = [peers[j].port for j in range(max_peers)]
        ports = [all_ports[all_ports.index(peer.port) - 1]]

        while len(ports) > 0:
            port = ports.pop(0)
            print(f'Receiving update from node: {port} for peer: {i}')
            status, port = peer.receive_from_port(port)
            if not status:
                time.sleep(1)
                peer.connect_with_node("localhost", port)
                ports.append(port)
            time.sleep(1)
        time.sleep(2)
        print(evaluate(peer, output))

# initialize peers
for i in range(max_peers):
    peer = Node('localhost', start_port + i, None, i, peer_data_dir)
    init_peer(peer)
    peers.append(peer)


# Connect peers to each other
exchange_weights(peers, max_peers)



for round_no in range(rounds):
    print(f"Performing experiment round {round_no}")
    #For each peer train next round
    for i in range(max_peers):
        peer = peers[i]
        print(f'Training model for round {peer.round_no + 1}')
        client_x, client_y = get_round_data(os.path.join(train_data_dir, f'{peer.peer_no}'), peer.round_no + 1)
        peer.train_round(client_x, client_y)
    round_no += 1

    exchange_weights(peers, max_peers)


print('Experiment complete')
sys.exit()



