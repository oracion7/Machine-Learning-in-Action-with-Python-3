# pca.py

import matplotlib
import matplotlib.pyplot as plt
from time import clock
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName, 'r')
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = []
    for line in stringArr:
        datArr.append([float(item) for item in line])
    # python 2: datArr = [list(map(float, line) for line in stringArr)]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 去平均值
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 从小到大对N个值排序
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects  # 将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # 计算所有非NaN的平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # 将所有的NaN置为平均值
    return datMat


if __name__ == '__main__':
    s = clock()

    # dataMat = loadDataSet('testSet.txt')
    dataMat = replaceNanWithMean()
    lowDMat, reconMat = pca(dataMat, 1)
    print(lowDMat)
    print(reconMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90, label='dataMat')
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, label='reconMat')
    plt.legend()
    plt.show()

    print("\n----------------")
    print("time cost:{0:.3f}s".format(clock() - s))
