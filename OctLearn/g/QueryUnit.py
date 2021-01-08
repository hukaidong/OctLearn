
class QueryUnit:
    def __init__(self, active_learning_network):
        self.network = active_learning_network

    def sample(self, num_sample):
        return self.network(num_sample)