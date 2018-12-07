from numpy import *
from os import listdir
import operator
def createDataSet():
    """
    以四个点(1,1.1),(1,1),(0,0),(0,0.1)为例，其标签分分别是'A','B','C','D'
    """
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    """
    输入参数为inX（待分类的数据特征向量），dataSet(训练数据多个特征向量构成的矩阵)，labels(训练数据特征向量对应的标签向量)，k（k值）
    返回为inX的预估的标签
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet  # 新数据与训练数据作差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)  # 对差平方矩阵进行每行求和
    distances = sqDistances**0.5       # 开平方根，为新数据向量与训练数据向量的距离
    sortedDistanceIndices = distances.argsort()  #按照距离从小到大排序，返回下标向量
    classCount = {}
    for i in range(k):
        # 统计前k个点对应的标签，插入classCount字典中
        voteIlabel = labels[sortedDistanceIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)  # 利用字典中的value值进行从大到小的排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    """
    根据文件名读取数据，返回特征数据矩阵和标签向量
    """
    dic = {'didntLike':1,'smallDoses':2,'largeDoses':3}
    fr = open(filename)
    arrayOLines = fr.readlines() # 以列表形式存储文本数据，列表元素为字符串
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3)) # 返回的特征数据矩阵
    classLabelVector = []     # 返回的标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip() # remove leading and trailing whitespace
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(dic[listFromLine[-1]])
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """
    参数为特征数据矩阵
    返回归一化后的特征数据矩阵，每一列最大最小值差值向量，每一列最小值向量
    """
    minVals = dataSet.min(0) # 对每一列求最小值，得到一个有三个元素的向量
    maxVals = dataSet.max(0) # 对每一列求最大值，得到一个有三个元素的向量
    ranges = maxVals-minVals # 获得每一列最大值与最小值的差值
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]  #数据集有多少行
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    """
    命令行输入特征数据，给出结果标签
    """
    hoRatio = 0.10
    datingDataMat,datingDataLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errCoount = 0    # 预测错误向量个数
    for i in range(numTestVecs):
        # 遍历所有的测试向量
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print("the classifier came back with:%d, the real answer is:%d"%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errCoount += 1
    print("the total error rate is: %f"%(errCoount/float(numTestVecs)))

def img2vector(filename):
    """
    参数为一个图像文本文件的名字
    返回1x1024的向量
    """
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def classifyDigitTest():
    # 首先列出trainingDigits下的所有文件名，文件数为m
    # 根据trainingDigits下的所有文件构造m*1024的特征矩阵和，m*1的标签向量
    # 列出testDigits下的所有文件名，文件数为testN
    # 对testDigits文件夹下的进行循环判断，以获得算法的手写数字识别错误率
        # 调用classify0获得digitVect的结果标签
        # 有文件名名可获得digitVect的真实标签，对比，若不等errorCount++
    # 获得手写识别错误率
    trainingFiles = listdir('trainingDigits')
    m = len(trainingFiles)
    trainingDataMat = np.zeros((m,1024))
    trainingLabelsVect = []
    for i in range(m):
        trainingFileName = trainingFiles[i]
        realLabel = int(trainingFileName.split('.')[0].split('_')[0])
        trainingDataMat[i,:] = img2vector('trainingDigits/'+trainingFileName) # 构造特征矩阵第i行
        trainingLabelsVect.append(realLabel)
    testFiles = listdir('testDigits')
    testN = len(testFiles)
    errCount = 0.0
    for i in range(testN):
        testFileName = testFiles[i]
        testDataVect = img2vector('testDigits/'+testFileName)  # 获得测试文件的特征向量
        testRealLabel = int(testFileName.split('.')[0].split('_')[0]) # 获得测试文件名中的数据标签
        testLabel = classify0(testDataVect,trainingDataMat,trainingLabelsVect,3) # 采用k近邻算法估计测试文件的标签
        # print("test data file:%s,predicted label:%s,real label:%s"%(testFileName,testLabel,testRealLabel))
        if testLabel!=testRealLabel:
            errCount+=1
    print("model accuracy:%.5f"%(1-errCount/testN))