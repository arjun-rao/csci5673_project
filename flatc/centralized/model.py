from flatc.dataset_final import get_round_data

class Model:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        # instantiate feature extractor
        # instantiate keras model


    def train(self, round_x, round_y):
        # TODO: Add round_x and round_y to self.X and self.Y
        # TODO: Fit Feature Extractor on X & Y
        # Retrain the model with this data but don't reinitialize weights

    def update_weights(weights):
        # TODO: Algorithm to average weights
        old_weights = self.model.get_weights()
        # TODO: Replace this line with actual for loop for averaging.
        averaged_weights = (old_weights + weights) / 2
        self.model.set_weights(averaged_weights)


 if __name__ == "__main__":
    round1_x, round1_y = get_round_data('../data/0', 0)
    model = Model(round1_x, round1_y)
    round2_x, round2_y = get_round_data('../data/0', 1)
    model.train(round2_x, round2_y)
