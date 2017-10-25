# apriori.py

from time import clock


def loadDataSet():
    return [[1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]


def createC1(dataSet):
    C1 = []
    for trasaction in dataSet:
        for item in trasaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # print(C1)
    # return [frozenset(item) for item in C1]
    return list(map(frozenset, C1))  # 对C1中的每个项构建一个不变集合


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 计算所有项集的支持度
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 前k-2个项相同时，将两个集合合并
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # 扫描数据集，从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            print(freqSet)
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                caclConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def caclConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmpl = aprioriGen(H, m + 1)
        Hmpl = caclConf(freqSet, Hmpl, supportData, brl, minConf)
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)


if __name__ == '__main__':
    s = clock()

    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # D = list(map(set, dataSet))
    # L1, supportData0 = scanD(D, C1, 0.5)
    # print(L1)

    L, supportData = apriori(dataSet, 0.5)
    rules = generateRules(L, supportData, 0.5)
    # print(L)
    print(rules)

    print("\n----------------")
    print("time cost: {0:.3f}s".format(clock() - s))
