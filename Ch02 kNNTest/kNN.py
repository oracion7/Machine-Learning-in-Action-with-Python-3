# kNN.py

import operator
import time
from numpy import *
from os import listdir
import matplotlib as mpl
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):  # 选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序
    return sortedClassCount[0][0]


def fileMatrix(filename):
    f = open(filename, "r")
    arrayOLines = f.readlines()  # 获取文件行数
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))  # 创建返回的NumPy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:  # 解析文件数据
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    f.close()
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = fileMatrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {0}, the real answer is: {1}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("range is {0}".format(numTestVecs))
    print("the total error rate is: {0}".format(errorCount / numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    datingDataMat, datingLabels = fileMatrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: {0}".format(resultList[classifierResult - 1]))


def imgVector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename, 'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):  # 从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = imgVector('trainingDigits/{0}'.format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = imgVector('testDigits/{0}'.format(fileNameStr))
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {0}, the real answer is: {0}".format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("\nthe total number of errors is: {0}".format(errorCount))
    print("\nthe total error rate is: {0}".format(errorCount / mTest))


if __name__ == '__main__':
    st = time.clock()

    # group, labels = createDataSet()
    # print(classify([0, 0], group, labels, 3))

    # datingDataMat, datingLabels = fileMatrix('datingTestSet.txt')
    # print(datingDataMat)
    # print(datingLabels[0:20])
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat, ranges, minVals)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()

    # datingClassTest()

    # classifyPerson()

    # testVector = imgVector('testDigits/0_14.txt')
    # print(testVector[0,0:31])

    handwritingClassTest()

    print("Time cost:{0:.2f}s".format(time.clock() - st))
