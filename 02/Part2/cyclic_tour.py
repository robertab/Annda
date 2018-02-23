from SOM import *
import numpy as np
import matplotlib.pyplot as plt

# read coordinates from file
def readCityCoordinates(city_coord):
    f = open('cities.dat', 'r')
    cities = f.readlines()
    cities  = [x.strip() for x in cities]
    cityNr = 0
    for coordinate in cities:
        city_coord[cityNr][0], city_coord[cityNr][1] = coordinate.split(', ')
        cityNr += 1

    return city_coord

def plotTraining(city_coord):
    plt.plot([city_coord[:,0]], [city_coord[:,1]], 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cities and traveling sales path')



nrOutputNodes = 15    # nr cities
nrFetures = 2         # city positions(x,y)
cities =  10


city_coord = np.zeros((cities,nrFetures))
city_coord = readCityCoordinates(city_coord)
#plotTraining(city_coord)

# creat self organising map
selfOrgMap = SOM(nrOutputNodes, nrFetures, np.copy(city_coord), nrOutputNodes)

epochs = 30
neighbourhood = 2
for epoch in range(0,epochs):

    # plotTraining(city_coord)
    # plt.plot([selfOrgMap.weights[:,0]], [selfOrgMap.weights[:,1]], 'bo')
    # plt.plot(selfOrgMap.weights[:,0], selfOrgMap.weights[:,1])
    # plt.title("neighbourhood " + str(neighbourhood) + ", epoch:  " + str(epoch))
    # plt.show()
    if epoch == 5:
        neighbourhood = 1
        print(neighbourhood)
    if epoch == 10:
        neighbourhood = 0
        print(neighbourhood)
    if epoch == 15:
        neighbourhood = 0
        print(neighbourhood)
    if epoch == 30:
        neighbourhood = 0
    #print(neighbourhood)

    organizedMap = selfOrgMap.run(neighbourhood, circular=True, nrWinners=cities)
    print(organizedMap)
    # plotTraining(city_coord)
    # plt.plot([selfOrgMap.weights[:,0]], [selfOrgMap.weights[:,1]], 'bo')
    # plt.plot(selfOrgMap.weights[:,0], selfOrgMap.weights[:,1])
    # plt.title("neighbourhood " + str(neighbourhood) + ", epoch:  " + str(epoch))
    # plt.show()
    tour = np.zeros((cities, nrFetures))  # winners
    i = 0
    for i in range(len(organizedMap)):
        tour[i][0] = selfOrgMap.weights[int(organizedMap[i]),0]
        tour[i][1] = selfOrgMap.weights[int(organizedMap[i]),1]
        i += 1
#     print(tour[:,0])
    print(tour)
    print(selfOrgMap.weights)
    plotTraining(city_coord)
    plt.plot([selfOrgMap.weights[:,0]], [selfOrgMap.weights[:,1]], 'bo')
    plt.plot(selfOrgMap.weights[:,0], selfOrgMap.weights[:,1])
    plt.plot(tour[:,0], tour[:,1], 'go')
    plt.show()

print(city_coord)
print(organizedMap)
print(selfOrgMap.weights)
# print("adfafasf")
# print(selfOrgMap.weights)
# plt.plot(selfOrgMap.weights[:,0], selfOrgMap.weights[:,1])
# plt.show()

tour = np.zeros((cities, nrFetures))  # winners
i = 0
for i in range(len(organizedMap)):
    tour[i][0] = selfOrgMap.weights[int(organizedMap[i]),0]
    tour[i][1] = selfOrgMap.weights[int(organizedMap[i]),1]
    i += 1
#print(tour)
print(tour[:,0])
plotTraining(city_coord)
#plt.plot([selfOrgMap.weights[:,0]], [selfOrgMap.weights[:,1]], 'bo')
#plt.plot(selfOrgMap.weights[:,0], selfOrgMap.weights[:,1])
plt.plot(tour[:,0], tour[:,1], 'go')
plt.plot(tour[:,0], tour[:,1])
plt.show()
