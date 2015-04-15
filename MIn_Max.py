import BpUseOnline as BP
from numpy import *
import random
import matplotlib as pyplt
def LoadTrainData():
    try:
        fr = open('train.txt')
    except Exception,e:
        print "open error"
    matofclasszero = []
    matofclassone = []
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        if(int(linearr[2]) == 1):
            matofclassone.append([float(linearr[0]),float(linearr[1]),int(linearr[2])])
        else:
            matofclasszero.append([float(linearr[0]),float(linearr[1]),int (linearr[2])])
    numofzero = shape(matofclasszero)[0]
    numofone = shape(matofclassone)[0]
    return numofzero,matofclasszero,numofone,matofclassone

def RandomChoose():
    numofzero,matofclasszero,numofone,matofclassone = LoadTrainData()

    nzero = int(numofzero*3/4)
    none = int(numofone*3/4)
    #print nzero ,none
    zeromat= []
    onemat = []
    for i in range(4):
        arrayzero = [k for k in range(numofzero)]
        arrayone = [k for k in range(numofone)]
        random.shuffle(arrayzero)
        random.shuffle(arrayone)
        tmpzeromat = []
        for j in range(nzero):
            zeromat.append(matofclasszero[arrayzero[j]])
        for j in range(none):
            onemat.append(matofclassone[arrayone[j]])
    resultmat=[]
    submodular = []
    for i in range(4):
        for j in range(4):
            trainmat = []
            for k in range(nzero):
                trainmat.append(zeromat[k])
            for k in range(none):
                trainmat.append(onemat[k])
            n = BP.NN(2,10,1)
            print '-----submodel_'+str(4*i+j+1),'is building------'
            n.SubModelTrain(trainmat,2001)
            print '------------building completed------------\n'
            submodular.append(n)#save every submodel
    Min_Max_model(submodular)
def Min_Max_model(submodel):
    
    resultmat1=[]
    for i in range(len(submodel)):
        string = 'WeightOutputFile/submodel_'+str(i)
        submodel[i].outputweights(string)
        error,retmat = submodel[i].test()
        resultmat1.append(retmat)
    resultmat = numpy.mat(resultmat1)
    res1 = resultmat[0:4,:]
    res2 = resultmat[4:8,:]
    res3 = resultmat[8:12,:]
    res4 = resultmat[12:16,:]
    maxmat=[]
    res1min = amin(res1,0)
    res2min = amin(res2,0)
    res3min = amin(res3,0)
    res4min = amin(res4,0)
    maxmat.append(res1min)
    maxmat.append(res2min)
    maxmat.append(res3min)
    maxmat.append(res4min)
    outcome=numpy.amax(maxmat,0)
   
    n,m = numpy.shape(outcome)
    resultmat = numpy.zeros((n,m))
    
    errcount = 0
    for i in range(n):
        for j in range(m):
            if(outcome[i][j]>0.5):
                resultmat[i][j] = 1
            else:
                resultmat[i][j] = 0
            if((j%2==0 and resultmat[i][j] !=1) or (j%2==1 and resultmat[i][j] !=0)):
                errcount = errcount + 1
  
    print 'the total error rate of Min_Max_Model is %f' % float((errcount*1.0)/(n*m))
    
if __name__ == '__main__':
    RandomChoose()
        










        
