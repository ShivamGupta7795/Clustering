#!/usr/bin/python3

import pandas as pd
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

class DBSCAN(object):
    import numpy as np
    def __init__(self, eps, minPts, metricOrder):
        self.eps=eps
        self.minPts=minPts
        self.ord=metricOrder
        
    def fit(self, DB):
        self.DB=DB
        self.dataDict={}
        C=0        
        for i in range(len(self.DB)):
            if(i in self.dataDict):
                continue
            neighbours = self.rangeQuery(i)
            if(len(neighbours) < self.minPts):
                self.dataDict[i]=-1
                continue
            C+=1
            self.estimatedClusters=C
            self.dataDict[i]=C
            seedSet = neighbours.difference({i})
            self.expandCluster(seedSet,C)
    
    def expandCluster(self,seedSet,C):        
        for j in seedSet:                
            if(j in self.dataDict):
                if(self.dataDict[j]==-1):
                    self.dataDict[j]=C                        
                continue
            self.dataDict[j]=C
            neighbours_ = self.rangeQuery(j)
            if(len(neighbours_)>=self.minPts):                
                self.expandCluster(neighbours_.difference({j}),C)       
    
    def rangeQuery(self, Q):
        neighbours = set()
        for i in range(len(self.DB)):
            if(np.linalg.norm(self.DB[Q] - self.DB[i], ord=self.ord)<=self.eps):
                neighbours.add(i)        
        return neighbours

class METRICS(object):
    """
    Input: Takes in two matrices of size (1 x n)
           Reshape the input matrices for this class to work.
           trueLabels = input1.reshape(1,input1.size)
    Usage: metrics=METRICS(trueLables = input1.reshape(1,input1.size,generatedLabels=input2.reshape(1, input2.size))
    """
    def __init__(self, trueLabels, generatedLabels):
        self.tL = trueLabels
        self.gL = generatedLabels
                
    def randIndex(self):        
        self.rand=(self.tL == self.tL.T).astype(int) == (self.gL == self.gL.T).astype(int)
        return (self.rand.sum()/self.rand.size)
    
    def jaccardIndex(self):        
        jaccard=((self.tL == self.tL.T).astype(int) & (self.gL == self.gL.T).astype(int)).sum()/((self.tL == self.tL.T).astype(int) | (self.gL == self.gL.T).astype(int)).sum()
        return jaccard

def hexColor():
    return ''.join([random.choice('0123456789ABCDEF') for x in range(6)])

file=sys.argv[1]
try:
    eps=float(sys.argv[2])
    minPts=int(sys.argv[3])
except:
    print("Missing epsilon/minPoints value")
    sys.exit()

try:
    metricOrder=sys.argv[4]
except:
    metricOrder=2

################################################################################
#  Imports the tab delimited file excluding the last column into a numpy matrix
data=np.genfromtxt(file, delimiter="\t")[:,2:]
################################################################################
#  Import the labels into a list, extract using pandas only the last column
true_labels=np.array(list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,1]))
################################################################################
#  Normalizing the data
X=(data - data.mean(0))
################################################################################
#  Compute the covariance matrix
S=(1/(X.shape[0]))*X.T.dot(X)
################################################################################
#  Compute and extract the eigen vectors from the covariance matrix
eigen_vectors=np.linalg.eig(S)[1]
################################################################################
#  Select the first two columns from the eigen vector table as the principal components
#  recompute samples based on principal components
pca_plotData=data.dot(eigen_vectors[:,0:2])

db=DBSCAN(eps=eps, minPts=minPts, metricOrder=metricOrder)
db.fit(DB=data)

labels=np.array(list(db.dataDict.values()))
lb=list(set(db.dataDict.values()))

df_pca = pd.DataFrame(dict(x=list(pca_plotData[:,0]),y=list(pca_plotData[:,1]), labels=np.array(list(db.dataDict.values()))))

metrics=METRICS(true_labels.reshape(1,true_labels.size),labels.reshape(1, labels.size))
print("Jaccard Index: "+str(metrics.jaccardIndex()))
print("Rand Index: "+str(metrics.randIndex()))
print(df_pca['labels'].groupby(df_pca['labels']).describe())
################################################################################
#  Plotting the dataframe with labels
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(len(lb)):
    ax1.scatter(df_pca[df_pca['labels']==lb[i]]['x'], df_pca[df_pca['labels']==lb[i]]['y'], color=("#"+hexColor()), label=lb[i])
plt.legend(loc='upper left')
plt.title("DBSCAN plot on "+file)
plt.savefig('DBSCAN_plot_'+file[:-4]+'.pdf')
plt.show()

