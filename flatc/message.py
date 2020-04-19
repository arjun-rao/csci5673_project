import json

CLIENT_WEIGHT_UPDATE = "CLIENT_WEIGHT_UPDATE"
REQUEST_GLOBAL_MSG = "REQUEST_GLOBAL_MSG"
REPLY_GLOBAL_MSG = "REPLY_GLOBAL_MSG"

class Message:
    def __init__(self, msg_type, modelid, weights, clientno, roundno, instance_count=0):
        self.msg_type = msg_type
        self.modelid = modelid
        self.weights = weights
        self.clientno = clientno
        self.roundno = roundno
        self.instance_count = instance_count

