import numpy as np
import random
import matplotlib.pyplot as plt

nB = 2000  #the number of bandits
nA = 10  #the number of arms
nP = 10000  #the number of plays (times we will pull a arm)
sigma = 1.0

# it should be normal distribution
qStarMeans = np.random.normal(size=(nB,nA))
epsArray = [0, 0.01, 0.1]

avgReward    = np.zeros((len(epsArray),nP))
perOptAction = np.zeros((len(epsArray),nP))
cumReward    = np.zeros((len(epsArray),nP))
cumProb      = np.zeros((len(epsArray),nP))

for ei in range(len(epsArray)):
    tEps = epsArray[ei]

    qT = np.zeros((nB,nA))  #initialize to zero draws per arm (no knowledge)
    qN = np.ones((nB,nA))  #keep track of the number draws on this arm
    qS = qT    #keep track of the SUM of the rewards (qT = qS./qN)

    allRewards = np.zeros((nB,nP))
    pickedMaxAction = np.zeros((nB,nP))

    for bi in range(nB):
        for pi in range(nP):
            if random.random() <= tEps:
                arm = random.randrange(0,nA)
            else:
                arm = qT[bi].argmax()

            bestArm = qStarMeans[bi].argmax()
            if bestArm == arm:
                pickedMaxAction[bi][pi] = 1;

            reward = qStarMeans[bi][arm] + sigma * random.uniform(-1,1)
            allRewards[bi][pi] = reward

            qN[bi][arm] = qN[bi][arm] + 1
            qS[bi][arm] = qS[bi][arm] + reward
            qT[bi][arm] = qS[bi][arm] / qN[bi][arm]

    avgRew = np.mean(allRewards,axis=0)  # mean of each column
    avgReward[ei] = avgRew
    percentOptAction = np.mean(pickedMaxAction,axis=0)
    perOptAction[ei] = percentOptAction
    csAR = np.cumsum(allRewards, axis=1)
    csRew = np.mean(csAR,0)
    cumReward[ei] = csRew
    csPA = np.cumsum(pickedMaxAction,axis=1) / np.cumsum( np.ones(pickedMaxAction.shape[1]))
    csProb = np.mean(csPA, axis=0)
    cumProb[ei] = csProb



#plot figures
x = np.arange(0, nP, 1)

# produce the average rewards plot
plt.figure(1)
plt.subplot()
for ei in range(len(epsArray)):
    label = plt.plot(x, avgReward[ei],label=epsArray[ei])
plt.legend(epsArray)

# produce the percent optimal action plot
plt.figure(2)
plt.subplot()
for ei in range(len(epsArray)):
    label = plt.plot(x, perOptAction[ei],label=epsArray[ei])
plt.legend(epsArray)

# produce the cummulative average rewards plot
plt.figure(3)
plt.subplot()
for ei in range(len(epsArray)):
    label = plt.plot(x, cumReward[ei],label=epsArray[ei])
plt.legend(epsArray)

# produce the cummulative percent optimal action plot
plt.figure(4)
plt.subplot()
for ei in range(len(epsArray)):
    label = plt.plot(x, cumProb[ei],label=epsArray[ei])
plt.legend(epsArray)

plt.show()


