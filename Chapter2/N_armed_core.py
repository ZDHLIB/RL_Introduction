import numpy as np
import random as rd
from CommonBean.Context import Context
import matplotlib.pyplot as plt

class N_armed_core:

    def __init__(self, *args, **kwargs):
        self.nB = kwargs.get('nB', 2000)
        self.nA = kwargs.get('nA', 10)
        self.nP = kwargs.get('nP', 1000)
        self.sigma = kwargs.get('sigma', 1)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.alpha = kwargs.get('alpha',0.1)


    # avg: estimating the action value using simple average AND
    # alpha: estimating the action value using a constant geometric factor alpha
    def starRL(self, strategy):
        allRewards = np.zeros((self.nB, self.nP))
        pickedMaxAction = np.zeros((self.nB, self.nP))
        for bi in range(self.nB):
            qStarMeans = np.ones((self.nB, self.nA))
            # qStarMeans = np.random.normal(size=(self.nB, self.nA))
            qTable = np.zeros(self.nA)
            qNum = np.zeros(self.nA)

            for pi in range(self.nP):
                if rd.random() <= self.epsilon :
                    arm = rd.randrange(0, self.nA)
                else:
                    arm = qTable.argmax()

                bestArm = qStarMeans[bi].argmax()

                if arm == bestArm:
                    pickedMaxAction[bi][pi] = 1;

                reward = qStarMeans[bi][arm] + self.sigma * rd.uniform(-1,1)
                allRewards[bi][pi] = reward

                # update qN, qT
                qNum[arm] += 1
                qTable[arm] = self.__updateQTable(strategy=strategy, qValue=qTable[arm],
                                           reward=reward, qNum=qNum[arm], alpha=self.alpha)


        avgReward = np.mean(allRewards, axis=0)  # mean of each column
        perOptAction = np.mean(pickedMaxAction, axis=0)
        csAR = np.cumsum(allRewards, axis=1)
        cumReward = np.mean(csAR, 0)
        csPA = np.cumsum(pickedMaxAction, axis=1) / np.cumsum(np.ones(pickedMaxAction.shape[1]))
        cumProb = np.mean(csPA, axis=0)

        return Context(allRewards=allRewards, pickedMaxAction=pickedMaxAction,
                       avgReward=avgReward, perOptAction=perOptAction,
                       cumReward=cumReward, cumProb=cumProb)


    # avg: estimating the action value using simple average AND
    # alpha: estimating the action value using a constant geometric factor alpha
    def __updateQTable(self, strategy, qValue, reward,  qNum, alpha ):
        if strategy == 'avg':
            return qValue + (reward - qValue ) / qNum
        elif strategy == 'alpha':
            return qValue + alpha * (reward - qValue )


if __name__ == '__main__':
    params = {'nB':2000, 'nA':10, 'nP':1000, 'sigma':1, 'epsilon':0.1, 'alpha':0.1}
    n_armed_core = N_armed_core(params)
    for ei in [0,0.01,0.1]:
        n_armed_core.epsilon = ei
        context = n_armed_core.starRL(strategy='avg')

        x = np.arange(0, params['nP'], 1)
        # produce the average rewards plot
        plt.subplot()
        plt.plot(x, context.avgReward)

    plt.show()