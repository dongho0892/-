'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)   - 기존의 데이터셋과 비교하는 벡터
            dataSet: size m data set of known vectors (NxM)    - NxM 크기의 알려진 벡터의 데이터셋
            labels: data set labels (1xM vector)               - 데이터셋의 레이블
            k: number of neighbors to use for comparison (should be an odd number)      - 비교를 위해 사용되는 숫자
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

# Python3 버전
def classify0(inX, dataSet, labels, k):  # 4개의 인자가 전달됨
    diffMat = inX - dataSet              #사용자가 입력한 값과 알려진 dataset을 뺌   ( 특정 지정된 값 )
    sqDiffMat = diffMat**2               # 그 뺸 값들을 각각 제곱한 뒤
    sqDistances = sqDiffMat.sum(axis=1)  # row 방향으로 그 값들을 더한다.
    distances = sqDistances**0.5          # 그 더한 값들을 제곱근한다 (루트를 씌운다.)
    sortedDistIndices = distances.argsort()     # 계산된 distances를 작은 값부터 순서대로 index(0부터 숫자를 매긴다)  
                                                #거리 : [2,3,1,4]  =>  그 거리의 인덱스 값[1,2,0,3] 
    classCount = {}             # 사전을 만듬.
    
    # 반복    range(k) : k개 만큼의 숫자 리스트를 만듬.
    for i in range(k):                                 # 0부터 k까지 한개씩 넣어줌
        voteIlabel = labels[sortedDistIndices[i]]     # sortedDistIndices의 값이 값이 0인 곳의 값을 찾아서 그 값에 해당하는 레이블을 voteIlabel에 저장한다.
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1              # {'B': 2, 'A': 2}  이런 식으로 값이 나옴.
         # # classCount 딕셔너리의 키 값에 없는 레이블(처음 나타난 레이블)이면 get 함수가 0을 리턴
        
    sortedClassCount = sorted(classCount, key=classCount.get, reverse=True)               #  내림차순으로 정렬
    return sortedClassCount[0]         # sortedClassCount 는 ['B', 'A'] 이렇게 리턴되는데, 그중 [0] 을 도출함.

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 수정본
#   - 파일을 한 번만 연다
#   - 즉, open(), readlines()를 한 번만 호출
def file2matrix(filename):        # 3가지 기능 있음
                                  # 파일을 열어서 해당 파일 내용중에서 공백 제거 / 레이블 값을 따로 저장 / feature값을 따로 저장
    fr = open(filename)          
    index = 0                     # 반환할 행렬의 로우 인덱스
    classLabelVec = []            # 클래스 레이블 변수 준비

    for line in fr.readlines():       # strip() 함수 : 양쪽 공백 제거 / split('\t') 탭 기준으로 단어 분리 
                  # 문자열 양 옆 공백 제거 후 탭('\t')을 기준으로 단어로 분리한 리스트
        lineList = line.strip().split('\t')
        
        # lineList의 마지막 컬럼값(레이블)을 클래스 레이블 변수에 추가
        classLabelVec.append(lineList[-1])
        
        # List Comprehension in Python
        feature = [float(value) for value in lineList[0:3]]
                  # 실수타입으로 리스트로 저장
            
        # 처음 읽은 라인이면 returnMat 변수 초기화, 아니면 returnMat 변수에 행 추가
        # 아래 if else 문을 한 문장으로 표현              # 세로로 쌓는다.   stack vertical
        returnMat = vstack((returnMat, feature)) \ 
                        if index != 0 else array(feature)
        #if (index == 0):
        #    returnMat = np.array(feature)
        #else:
        #    returnMat = np.vstack((returnMat, feature))
        index += 1
    return returnMat, classLabelVec

(또다른 버전)--------------------------------------
    def file2matrix(filename):
    fr = open(filename)
    
    # 훈련 데이터(returnMat)와 클래스 레이블(classLabelVector) 변수 준비
    numberOfLines = len(fr.readlines())      # 텍스트 파일의 라인 개수             
    returnMat = np.zeros((numberOfLines, 3)) # 텍스트_라인수 X 3 크기의 영행렬 준비  # 1000 x 3 짜리 행렬을 만들어놓음 (공간 미리 확보)
    classLabelVector = []                    # 클래스 레이블 변수 준비          # 클래스를 받은 빈 벡터를 만들어놓음.
    
    # readlines() 는 파일 끝까지 읽은 뒤 포인터를 파일의 처음으로 돌려놓지 않으므로
    # 파일 내용을 다시 읽으려면 open() 을 사용하여 파일을 다시 읽어야 한다.
    fr = open(filename)        # 다시 오픈해서 읽어들이겠다.
    index = 0            # 반환할 행렬의 로우 인덱스          
    
    for line in fr.readlines():       # 1000건의 데이터를 읽어들임.
        # 문자열 양 옆 공백 제거
        line = line.strip()           
        
        # 문자열을 탭('\t')을 기준으로 단어로 분리하여 리스트로 반환
        listFromLine = line.split('\t') 
        
        # returnMat의 인덱스 로우에 위 리스트의 첫 번째부터 세 번째 값을 저장
        returnMat[index, :] = listFromLine[0:3]          
               # 1000행 3열짜리      / 4개의 값이 들어있는 리스트 중에서 0,1,2   3개만 취하겠다는 의미. 
               #그 값을 returnMat 첫번째 row에 값을 대입하겠다.
                
        # classLabelVector 리스트에 listFromLine 리스트의 마지막 값을 저장
        classLabelVector.append(listFromLine[-1]) # 마지막 3, 4번째 값을 넣어줌.
        index += 1       # 자료를 다 넣어주고, 인덱스 값을 1 더해줌 ( 인덱스 값을 바꿈 )
        
    return returnMat, classLabelVector

    --------------------------------------


def autoNorm(dataSet):
  # 각 특징별 최솟값, 최댓값, 범위를 구함
    minVals = dataSet.min(0)    # dataSet 배열의 열에 대한 최솟값, 0이 '열'을 의미
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals 

    # 범위가  - 마일리지, 아이스크림 소배, 비디오게임
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
def datingClassTest():
                # 각종 변수 준비
    hoRatio = 0.10                    # 테스트 데이터 비율 (10%)
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    m = normMat.shape[0]              # 데이터 총 개수, 로우 수
          # 1000 
    numTestVecs = int(m * hoRatio)    # 테스트 데이터 개수
        # 1000건의 10% => 100개짜리 테스트용 데이터 
    errorCount = 0.0
        
    for i in range(numTestVecs):      # kNN 알고리즘에 들어있음 def 함수 / 
        classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                          # inX        , dataSet
                                         
                                           # 정규화된                         1000 들어오면
                                         datingLabels[numTestVecs:m], 3)        # 3개의 점을 찾아서     # kNN 수행: k 값은 3
                                           # labels                 , k 
        # classifierResult의 값 예시'largeDoses'
        
        # 교재에서는 %s 대신 %d, 교재대로 하려면 datingTestSet2.txt 파일을 사용
        print("the classifier came back with: %s, the real answer is: %s"\
                % (classifierResult, datingLabels[i]))
        
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            print("!!!NOT MATCHED!!!")
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))   # 함수가 종료되면 에러율을 보여줌. 
    
    
    
    
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