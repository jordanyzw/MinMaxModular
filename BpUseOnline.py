# -*- coding: cp936 -*-
# Back-Propagation Neural Networks
# 

from numpy import *
import random
import string
import  matplotlib
import matplotlib.pyplot as plt
random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function
def sigmoid(x):
    return 1.0/(1+exp(-x))

# derivative of our sigmoid function
# 在求取输出层和隐藏侧的误差项的时候会用到
def dsigmoid(y):
    return y*(1-y)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        # 输入层，隐藏层，输出层的数量，三层网络
        self.ni = ni
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        #生成权重矩阵，每一个输入层节点和隐藏层节点都连接
        #每一个隐藏层节点和输出层节点链接
        #大小：self.ni*self.nh
        self.ui = makeMatrix(self.ni, self.nh)
        self.vi = makeMatrix(self.ni, self.nh)
        self.bh = makeMatrix(self.nh, 1)
        #大小：self.ni*self.nh
        self.uo = makeMatrix(self.nh, self.no)
        self.vo = makeMatrix(self.nh, self.no)
        self.bo = makeMatrix(self.no,1)
        # set them to random vaules
        #生成权重，在-0.2-0.2之间
        for i in range(self.ni):
            for j in range(self.nh):
                self.ui[i][j] = rand(-1.0, 1.0)
                self.vi[i][j] = rand(-1.0, 1.0)
               
        for i in range(self.nh):
            self.bh[i][0] = rand(-1.0,1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.uo[j][k] = rand(-1.0, 1.0)
                self.vo[j][k] = rand(-1.0, 1.0)
        for i in range(self.no):
            self.bo[i][0] = rand(-1.0,1.0)
        # last change in weights for momentum 

    def update(self, inputs):
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        # 输入的激活函数，就是y=x;
        for i in range(self.ni):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        #隐藏层的激活函数,求和然后使用压缩函数
        for j in range(self.nh):
            totalsum = 0.0
            for i in range(self.ni):
                #sum就是《ml》书中的net
                totalsum = totalsum + self.ai[i] * self.ai[i] * self.ui[i][j] + self.ai[i] * self.vi[i][j]
            totalsum = totalsum + self.bh[j][0]
            self.ah[j] = sigmoid(totalsum)

        # output activations
        #输出的激活函数
        for k in range(self.no):
            totalsum = 0.0
            for j in range(self.nh):
                totalsum = totalsum + self.ah[j] * self.ah[j] * self.uo[j][k] + self.ah[j] * self.vo[j][k]
            totalsum  =  totalsum + self.bo[k][0]
            self.ao[k] = sigmoid(totalsum)
        return self.ao[:]

    #反向传播算法 targets是样本的正确的输出
    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')
        # calculate error terms for output
        #计算输出层的误差项 
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            #计算k-o
            error = self.ao[k]- targets[k]
            output_deltas[k] =  dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        #计算隐藏层的误差项
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * ( 2*self.ah[j]*self.uo[j][k] + self.vo[j][k] )
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        # update output weights
        # 更新输出层的权重参数
        # 这里可以看出，本例使用的是带有“增加冲量项”的BPANN
        # 其中，N为学习速率 M为充量项的参数 self.co为冲量项
        # N: learning rate
        # M: momentum factor
        for j in range(self.nh):
            for k in range(self.no):
                self.uo[j][k] = self.uo[j][k] - N*output_deltas[k]*self.ah[j]*self.ah[j]
                self.vo[j][k] = self.vo[j][k] - N*output_deltas[k]*self.ah[j]
        for j in range(self.no):
            self.bo[j][0] = self.bo[j][0] - N*output_deltas[j]
               
        # update input weights
        #更新输入项的权重参数
        for i in range(self.ni):
            for j in range(self.nh):      
                self.ui[i][j] = self.ui[i][j] - N*hidden_deltas[j]*self.ai[i]*self.ai[i]
                self.vi[i][j] = self.vi[i][j] - N*hidden_deltas[j]*self.ai[i]  
        for j in range(self.nh):
            self.bh[j][0] = self.bh[j][0] - N*hidden_deltas[j]

    #测试函数，用于测试训练效果
    def test(self):
        atributemat = []; labelmat = []
        retmat = []
        fr = open('test.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1])])
            labelmat.append([int(linearr[2])])
        n = shape(atributemat)[0]
####        fig = plt.figure()
####        ax = fig.add_subplot(111)
####        plotmat = mat(atributemat)
####        labmat= mat(labelmat)+ones((n,1))
####        ax.scatter(plotmat[:,0],plotmat[:,1],15.0*array(labmat),15.0*array(labmat))
####        plt.show()
        errcount = 0
        for i in range(n):
            temp = self.update(atributemat[i])[0]
            retmat.extend([temp])    
##            print('correct label:',labelmat[i][0],'perdicted value',temp)
            if((labelmat[i][0] == 1 and temp < 0.5) or (labelmat[i][0] == 0 and temp > 0.5)):
##                print('case %d is not classified correctly' % i)
                errcount = errcount + 1
        print('the total errrate is %f' % (errcount/(n*1.0)))
        return (errcount/(n*1.0)),retmat
            

    def train(self,iterations=5001, N=0.111,pyvot = 0.00001):
        # N: learning rate
        # M: momentum factor
        atributemat = []; labelmat = []
        datamat=[]
        fr = open('train.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1])])
            labelmat.append([int(linearr[2])])
            datamat.append([float(linearr[0]),float(linearr[1]),int(linearr[2])])
        n = shape(atributemat)[0]
##        fig = plt.figure()
##        ax = fig.add_subplot(111)
##        plotmat = mat(atributemat)
##        labmat= mat(labelmat)+ones((n,1))
##        ax.scatter(plotmat[:,0],plotmat[:,1],15.0*array(labmat),15.0*array(labmat))
##        plt.show()
        for i in range(iterations):
            
            for j in range(n):
                inputs = atributemat[j]
                targets = labelmat[j]
                self.update(inputs)
                self.backPropagate(targets, N)
            error = self.CalMeanError(datamat)
            if(error < pyvot):
                print "return"
                return
            if(i%1000 == 0 and i != 0):
                N=N/1.1
            if i % 1000 == 0:
                print('after %d iterations ,the mean error %-.5f' % (i,error))
    
    def CalMeanError(self,datamat,outputnum=1):
        error = 0.0
        n,m = shape(datamat)
        target = [] 
        for i in range(n):
            target.append([datamat[i][m-outputnum]])
        for i in range(n):
            result = self.update(datamat[i][0:m-outputnum])
            for k in range(outputnum):
                error = error + 0.5*(target[i][k]-self.ao[k])**2
        return error/n
            
    def SubModelTrain(self,datamat,iterations=5001, N=0.111,pyvot=0.000001):
        n,m= shape(datamat)
        for i in range(iterations):
            for j in range(n):
                inputs = []
                for k in range(m-1):
                    inputs.extend([datamat[j][k]])                  
                targets = [datamat[j][-1]]
                self.update(inputs)
                self.backPropagate(targets, N)
            error = self.CalMeanError(datamat)
            if(error < pyvot):
                print "return"
                return
            if(i%500==0):
                print('after %d iterations ,the mean error %-.5f' % (i,error))
    def outputweights(self,filename):
        try:
            fr = open(filename,'w')
        except Exception,e:
            print "open error"
        string = ""
        for i in range(self.ni):
            for j in range(self.nh):
                string = string + str(self.ui[i][j])+'\t'+str(self.vi[i][j])+'\n'
        for j in range(self.nh):
            for k in range(self.no):
                string = string + str(self.uo[j][k]) +'\t'+ str(self.vo[j][k])+'\n'      
        for j in range(self.nh):
            string = string + str(self.bh[j][0])+'\n'
        for j in range(self.no):
             string = string + str(self.bo[j][0])+'\n'
        fr.write(string)     
def demo():
    # create a network with two input, one hidden,in the hidden there are two nueron, and one output nodes
    n = NN(2, 10, 1)
    n.train()
    n.test()
    n.outputweights("WeightOutputFile/BpWeight.txt")
if __name__ == '__main__':
    demo()
