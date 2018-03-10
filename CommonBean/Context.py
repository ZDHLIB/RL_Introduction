import numpy as np

class Context:
    def __init__(self, *args, **kwargs):
        self.allRewards = kwargs.get('allRewards', np.zeros((2000, 10)) )
        self.pickedMaxAction = kwargs.get('pickedMaxAction', np.zeros((2000, 10)) )
        self.avgReward = kwargs.get('avgReward', np.zeros(1000) )
        self.perOptAction = kwargs.get('perOptAction', np.zeros(1000) )
        self.cumReward = kwargs.get('cumReward', np.zeros(1000) )
        self.cumProb = kwargs.get('cumProb', np.zeros(1000) )