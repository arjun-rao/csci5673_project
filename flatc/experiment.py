from client import *

test_x, test_y = get_data('data/test.csv')

def evaluate(client, fname='results.csv'):
    data = {
        'client_no': client.client_no,
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


def init_peer(peer):
    peer_x, peer_y = get_round_data(os.path.join('data/', f'{peer.client_no}'), 0)
    peer.init(peer_x, peer_y)
    print(f"Peer {peer.client_no} created with model for round 0.")
    # print(evaluate(peer, output))


output = '5_clients_centralized.csv'
train_data_dir = 'data/'
max_peers = 5
rounds = 4
peers = []


# initialize peers
for i in range(max_peers):
    peer = Client('localhost', 10000, i, './client_data')
    init_peer(peer)
    peers.append(peer)

# For each peer send weights
for i in range(max_peers):
    peer = peers[i]
    print(f'Send weights from peer: {i}')
    peer.send_weights()
    time.sleep(1)
    # print(evaluate(peer, output))

# For each peer get global model
for i in range(max_peers):
    peer = peers[i]
    print(f'Receiving weights to peer: {i}')
    peer.get_updatedweights(peer.model_id)
    time.sleep(1)
    print(evaluate(peer, output))



for round_no in range(rounds):
    print(f"Performing experiment round {round_no}")
    #For each peer train next round
    for i in range(max_peers):
        peer = peers[i]
        print(f'Training model for round {peer.round_no + 1}')
        client_x, client_y = get_round_data(os.path.join(train_data_dir, f'{peer.client_no}'), peer.round_no + 1)
        peer.train_round(client_x, client_y)
    round_no += 1

    # For each peer send weights
    for i in range(max_peers):
        peer = peers[i]
        print(f'Send weights from peer: {i}')
        peer.send_weights()
        time.sleep(1)
        print(evaluate(peer, output))

    # For each peer get global model
    for i in range(max_peers):
        peer = peers[i]
        print(f'Receiving weights to peer: {i}')
        peer.get_updatedweights(peer.model_id)
        time.sleep(1)
        print(evaluate(peer, output))

print('Experiment complete')
sys.exit()



