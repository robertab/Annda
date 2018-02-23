from SOMmp import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def readVotes():
    votes = np.zeros((349, 31))
    f = open('mp_data/votes.dat', 'r')
    dirty_votes = f.readlines();
    dirty_votes  = [x.strip() for x in dirty_votes] # remove eg. \n
    dirty_votes = dirty_votes[0]
    dirty_votes = dirty_votes.split(',')
    count = 0;
    for i in range(349):
        for j in range(31):
            votes[i][j] = dirty_votes[count]
            count += 1
    return votes;
def readInfo(path):
    mp_info = np.zeros(349)
    f = open(path, 'r')
    # Coding: Male 0, Female 1
    info = f.readlines();
    info = [x.strip() for x in info]
    for i in range(len(info)):
        mp_info[i] = info[i]

    return mp_info


nrNodesInDim = 10
nrFetures = 31
nrSamples = 349
neighbourhood = 8          #10  
epochs = 30

# (349 x 31) 349 mp:s 31 votes
mp_votes = readVotes()
selfOrgMap = SOMmp(nrNodesInDim, nrFetures, mp_votes, nrSamples)
for i in range(epochs):
    neighb = neighbourhood - (0.4 * i)
    neighb = round(neighb)
    if neighb < 0:
     neighb = 0
    #print(neighb)
    orgMap = selfOrgMap.run(neighb)
    #print(orgMap)
#print(selfOrgMap.weights)
print(orgMap)
mp_gender = readInfo('mp_data/mpsex.dat')

# Plot all weights

for i in range(10):
    for j in range(10):
        # find closest input to weight
        shortestDistance = 1000000000
        voteIndex = -1
        for k in range(349):
            difference = np.subtract(mp_votes[k], np.copy(selfOrgMap.weights[i][j]))
            distanceTemp = np.dot(difference.T, difference)
            if shortestDistance > distanceTemp:
                shortestDistance = distanceTemp
                voteIndex = k

        # Coding: Male 0, Female 1
        if mp_gender[voteIndex] == 0:
            plt.plot(i, j, 'bs')
        else:
            plt.plot(i, j, 'r^')
plt.axis('off')
plt.grid(True)
plt.axis([0, 9, 0, 9])
plt.show()

for i in range(349):
    node=orgMap[i]
    x = int(node / 10)
    y = node - x *10
    if mp_gender[i] == 0:
        plt.plot(x, y, 'bs')
    else:
        plt.plot(x, y, 'r^')
        #plt.text(x, y, r'F')
plt.grid(True)
plt.axis([0, 9, 0, 9])
plt.axis('off')
plt.show()


mp_district = readInfo('mp_data/mpdistrict.dat')

for i in range(10):
    for j in range(10):
        # find closest input to weight
        shortestDistance = 1000000000
        voteIndex = -1
        for k in range(349):
            difference = np.subtract(mp_votes[k], np.copy(selfOrgMap.weights[i][j]))
            distanceTemp = np.dot(difference.T, difference)
            if shortestDistance > distanceTemp:
                shortestDistance = distanceTemp
                voteIndex = k

        plt.text(i, j, mp_district[voteIndex])
        plt.plot(i, j, 'wo')


plt.grid(True)
plt.axis([0, 9, 0, 9])
plt.axis('off')
plt.show()


mp_party = readInfo('mp_data/mpparty.dat')
# Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
# Use some color scheme for these different groups

for i in range(10):
    for j in range(10):
        # find closest input to weight
        shortestDistance = 1000000000
        voteIndex = -1
        for k in range(349):
            difference = np.subtract(mp_votes[k], np.copy(selfOrgMap.weights[i][j]))
            distanceTemp = np.dot(difference.T, difference)
            if shortestDistance > distanceTemp:
                shortestDistance = distanceTemp
                voteIndex = k

        plt.text(i, j, mp_party[voteIndex])
        if mp_party[voteIndex] == 1:
            plt.plot(i, j, color='#000099', marker='o')
        elif mp_party[voteIndex] == 2:
            plt.plot(i, j, color='#66ccff', marker='o', label='fp')
        elif mp_party[voteIndex] == 3:
            plt.plot(i, j, color='#ff0000', marker='o', label='s')
        elif mp_party[voteIndex] == 4:
            plt.plot(i, j, color='#ff3399', marker='o', label='v')
        elif mp_party[voteIndex] == 5:
            plt.plot(i, j, color='#003300', marker='o', label='mp')
        elif mp_party[voteIndex] == 6:
            plt.plot(i, j, color='#336699', marker='o', label='kd')
        elif mp_party[voteIndex] == 7:
            plt.plot(i, j, color='#00cc99', marker='o', label='c')
        else:
            plt.plot(i, j, color='#000000', marker='o', label='no party')


plt.grid(True)
plt.axis([0, 9, 0, 9])
m = mpatches.Patch(color='#000099', label='m')
fp = mpatches.Patch(color='#66ccff', label='fp')
s = mpatches.Patch(color='#ff0000', label='s')
v = mpatches.Patch(color='#ff3399', label='v')
mp = mpatches.Patch(color='#003300', label='mp')
kd = mpatches.Patch(color='#336699', label='kd')
c = mpatches.Patch(color='#00cc99', label='c')
noP = mpatches.Patch(color='#000000', label='no party')
plt.axis('off')
plt.legend(handles=[m, fp, s, v, mp, kd, c, noP],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.show()
