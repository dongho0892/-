'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']           
    #change to discrete values
    return dataSet, labels      
    # feature 이름

def calcShannonEnt(dataSet):    # 위의 labels와 Label은 다름!
    numEntries = len(dataSet)
    labelCounts = {}                 # 사전 준비   ( for 돌리면 {'yes' : 1 ,  
    for featVec in dataSet: #the the number of unique elements and their occurance           # 데이터 셋 가져옴.
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0               # 그 안에 없다면
        labelCounts[currentLabel] += 1               # yes 면 2, no 면 1                       # 몇번 나왔는지 보는 이유는 확률을 구하기 위함임!
    #       labelCounts[currentLabel] = labelCunt.get(currentLabel, 0 )  + 1     - 파이썬 3버전
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries               
        shannonEnt -= prob * log(prob,2) #log base 2             # 정보량의 기대값은 확률에 로그확률을 곱합
    return shannonEnt                                           # 엔트로비를 구하는 방법
    
    
      - dataSet: 분할하고자 하는 데이터 집합
      - axis: 특징의 인덱스
      - value: 특징의 값
    
                # 맞추어서 써야함.# 0번째 feature로 감.
def splitDataSet(dataSet, axis, value):                     #     
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:            첫번째 로우의 밸류값이 
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])     => 위 두라인을 통과하고 나면   해당 feature만 뺌   [ 0,0 ] =>        
            retDataSet.append(reducedFeatVec)        # 0에 대해서 시행하고 / 1에 대해서 시행하고, 각각 분리되서 나오게 됨.
    return retDataSet                              # 리턴데이터셋에다가 넣어줌. 
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels         
                                           # [1, 1, 'yes'],   이렇게 생긴 거에서 마지막 label만 제거 하면 피처 2개만 나옴
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1     # 최적인 정보 값을 구하기 위해서 우선 0으로 둠.
        피처번호 (0,1) - 노서피싱 / 지느러미
    for i in range(numFeatures):                # iterate over all the features   # 피처 갯수만큼 반복을 함.
           # 0번째 피처에 해당하는                  
        featList = [example[i] for example in dataSet]   # create a list of all the examples of this feature
        uniqueVals = set(featList)   # 유니크한 값만 나오게해서 나중에 밸류값으로 쓰기 위해서           # get a set of unique values
        newEntropy = 0.0
                      # 유니크한  피처를 뽑아오겠다. 
        for value in uniqueVals:        # 0번째 피처에 0값을 가지는 것
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))    확률을 구해서
            newEntropy += prob * calcShannonEnt(subDataSet)        # 정보량을 구한 다음에, 각각 자기의 확률을 다시 곱해서  합해줌.
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy
                   
        if (infoGain > bestInfoGain):       # compare this to the best gain so far
            bestInfoGain = infoGain         # if better than current best, set to best
            bestFeature = i                    # 정보이득을 판단해서 좋은 걸로 선택하게끔 결과치를 리턴시켜줌.
                                               # 그래서 좋은 것이면 아래 리턴 할때 그걸로 되게끔 함.
    return bestFeature                      #returns an integer    

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
