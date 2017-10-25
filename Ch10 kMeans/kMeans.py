# kMeans.py

import time
import json
from numpy import *
from urllib import request
from urllib import parse


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    fr.close()
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):  # 构建簇核心
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):  # 寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):  # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()  # 创建一个初始簇
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            pstInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 尝试划分每一簇
            centroidMat, splitClustAss = kMeans(pstInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], i])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 更新簇分配结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1:, ])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'  # 返回类型设置为json
    params['appid'] = 'ppp68N8t'
    params['location'] = '{0:s} {1:s}'.format(stAddress, city)
    url_params = parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)  # 打印出url
    c = request.urlopen(yahooApi)
    return json.loads(c.read())


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        reDict = geoGrab(lineArr[1], lineArr[2])
        if reDict['ResultSet']['Error'] == 0:
            lat = float(reDict['ResultSet']['Results'][0]['latitude'])
            lng = float(reDict['ResultSet']['Results'][0]['longitude'])
            print("{0:s}\t{1:f}\t{2:f}".format(lineArr[0], lat, lng))
            fw.write("{0:s}\t{1:f}\t{2:f}\n".format(line, lat, lng))
        else:
            print("error fetching")
            time.sleep(1)
    fw.close()


if __name__ == '__main__':
    s = time.clock()

    # datMat3 = mat(loadDataSet('testSet2.txt'))
    # centList, myNewAssments = biKmeans(datMat3, 3)
    #
    # print("\ncentList: ", centList)

    geoResults = geoGrab('1 VA Center', 'Augusta, ME')
    print(geoResults)

    print("\n-------------------")
    print("time cost: {0:.3f}s".format(time.clock() - s))
