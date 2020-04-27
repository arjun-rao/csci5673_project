import json

CLIENT_WEIGHT_UPDATE = "CLIENT_WEIGHT_UPDATE"
REQUEST_PEER_MSG = "REQUEST_PEER_MSG"
REPLY_PEER_MSG = "REPLY_PEER_MSG"

class Message:
    def __init__(self, msg_type, weights, peerno, roundno, instance_count=0):
        self.msg_type = msg_type
        self.weights = weights
        self.peerno = peerno
        self.roundno = roundno
        self.instance_count = instance_count

