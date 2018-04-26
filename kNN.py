'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

# Python3 버전
def classify0(inX, dataSet, labels, k):
    diffMat = inX - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount, key=classCount.get, reverse=True)
    return sortedClassCount[0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 수정본
#   - 파일을 한 번만 연다
#   - 즉, open(), readlines()를 한 번만 호출
def file2matrix(filename):
    fr = open(filename)
    index = 0                     # 반환할 행렬의 로우 인덱스
    classLabelVec = []            # 클래스 레이블 변수 준비
    for line in fr.readlines():
        # 문자열 양 옆 공백 제거 후 탭('\t')을 기준으로 단어로 분리한 리스트
        lineList = line.strip().split('\t')
        
        # lineList의 마지막 컬럼값(레이블)을 클래스 레이블 변수에 추가
        classLabelVec.append(lineList[-1])
        
        # List Comprehension in Python
        feature = [float(value) for value in lineList[0:3]]
        
        # 처음 읽은 라인이면 returnMat 변수 초기화, 아니면 returnMat 변수에 행 추가
        # 아래 if else 문을 한 문장으로 표현
        returnMat = vstack((returnMat, feature)) \
                        if index != 0 else array(feature)
        #if (index == 0):
        #    returnMat = np.array(feature)
        #else:
        #    returnMat = np.vstack((returnMat, feature))
        index += 1
    return returnMat, classLabelVec

def autoNorm(dataSet):
    # 각 특징별 최솟값, 최댓값, 범위를 구함
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    
    # 정규화된 배열을 구함
    normDataSet = (dataSet - minVals) / ranges
    
    # ranges와 minVals은 테스트 값에 대해 정규화할 필요가 있으므로 반환해야 함
    return normDataSet, ranges, minVals

  
"""
file: 테스트 데이터 파일 이름
ratio: 테스트 데이터 비율
k: kNN의 k 값

입력 데이터 중 [0, 총데이터*테스트비율) 범위의 데이터가 테스트 데이터로 사용된다.
"""
def datingClassTest(file, ratio = 0.1, k = 3):
    # 각종 변수 준비
    datingDataMat, datingLabels = file2matrix(file)
    normMat, ranges, minVals = autoNorm(datingDataMat)
     # 가지고 와서 보관한 다음에     # 정규화된 데이터를 넘겨줌.
    numTotal = normMat.shape[0]            # 데이터 총 개수, 로우 수
    numTestVecs = int(numTotal * ratio)    # 테스트 데이터 개수
    errorCount = 0
        
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                         datingLabels[numTestVecs:m], k)        
        # 교재에서는 %s 대신 %d, 교재대로 하려면 datingTestSet2.txt 파일을 사용
        print("the classifier came back with: %s, the real answer is: %s"\
                % (classifierResult, datingLabels[i]))# 분류기의 결과와 실제 값이 화면에 출력됨
        if (classifierResult != datingLabels[i]):
            errorCount += 1
            print("!!!NOT MATCHED!!!")
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))