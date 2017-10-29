#!/usr/bin/python3

#<h3>Import necessary packages</h3>
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.spatial.distance as sd
import sys
from pandas import Series
# from numpy.random import randn


#<h3>Supply dataset file</h3>
file = sys.argv[1]
#<h3>Read data from the raw file into a numpy ndarray</h3>
data=np.genfromtxt(file, delimiter="\t")[:,2:]

#<h3>Get the labels</h3>
true_labels=list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,1])

#<h3>Storing the observation IDs</h3>
obv_id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,0])

#<h3>Normalising the data</h3>
X = (data - data.mean(0))

try:
    k = int(sys.argv[2])
except:
    k=len(set(true_labels))

# <h3>Calculating Proximity Matrix</h3>
# <h4>For this I'm using scipy.spatial.distance.cdist package
# https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.spatial.distance.cdist.html</h4>
# <h4>Calculating the actual distance matrix using Euclidean Distance as the metric</h4>

def distance_matrix_calc(data):
    # Create a blank numpy square matrix to the size of data observations
    dist_mat = np.zeros((data.shape[0],data.shape[0]))
    
    # Calculating the actual distance matrix using Euclidean Distance as the metric
    dist_mat = sd.cdist(data, data, metric = 'euclid')
    
    #print("Shape of resultant Distance Matrix is: ",dist_mat.shape)
    return dist_mat

# <h3>Calling the function distance_matrix_calc</h3>
dist_mat=distance_matrix_calc(data)

# <h4>Initialised the minimum distance to the 1st non-zero value in dist_mat</h4>
min_dist = [dist_mat[1][0],1,0]

# <h3>Loop to find the minimum value in the dist_mat and saving the same in a list min_dist</h3>
# <h4> (considering the lower triangle only since distance matrix is a diagonal matrix)</h4>
def find_min(dist_mat, min_dist):
    min_dist = [dist_mat[1][0],1,0]
    for i in range(0,len(dist_mat)):
        #print("i =",i)
        for j in range(0, i):
            #print("j =",j)
            if(dist_mat[i][j] == 0):
                #print("value 0 reached at i,j =",i,j)
                break;
            elif(dist_mat[i][j] < min_dist[0]):
                min_dist = [dist_mat[i][j], i, j]
    return min_dist

# <h4>Updating the Distance Matrix given the minimum distance observations merge to form a single cluster.
# We update the row and column values of observation 1 to reflect the updated values.</h4>
# <h4>Also updating the list of observation labels/ids to reflect the same</h4>
def update_mat(updist_mat, min_dist, upid):

    for i in range(0, updist_mat.shape[0]):
        updist_mat[min_dist[1]][i] = min (updist_mat[min_dist[1]][i], updist_mat[min_dist[2]][i]) 
        updist_mat[i][min_dist[1]] = min (updist_mat[i][min_dist[1]], updist_mat[i][min_dist[2]])

    merge_id = (upid[min_dist[1]], upid[min_dist[2]])
    upid[min_dist[1]] = list(merge_id)
    
    updist_mat = np.delete(updist_mat, min_dist[2], axis=0)
    updist_mat = np.delete(updist_mat, np.s_[min_dist[2]], axis =1)
    
    del upid[min_dist[2]]
    
    return updist_mat, upid

# <h3>Initializing values</h3>
upid = list(obv_id)
updist_mat = dist_mat

# <h3>Calling function update_mat</h3>
while(len(upid)> k):
    min_dist = find_min(updist_mat, min_dist)
    updist_mat, upid = update_mat(updist_mat, min_dist, upid)
    
# <h3>Flatten the list upid</h3>
def listflatten(l):
    a=[]
    if(type(l)==list):
        for i in l:
            a+=listflatten(i)
    else:
        a.append(l)
    return a

# <h4>Generate a flattened structure of upid</h4>
a=[]
for i in upid:
    a.append(listflatten(i))

# <h3>Duplicating data</h3>
data2=data.copy()

# <h4>Adding a column to new data2 that stores respective labels</h4>
data2=np.concatenate((np.random.rand(data2.shape[0],1), data2), axis=1)
for i in range(len(a)):
    for j in a[i]:
        data2[j-1,0]=i
        
# <h3>Metric Class</h3>
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
    
# <h4>Calling the metrics class</h4>
mt = METRICS(data2[:,0].reshape(data2.shape[0],1), np.array(true_labels).reshape(len(true_labels),1))

# <h4>Computing the Covariance Matrix for PCA plot</h4>
S=(1/(X.shape[0]))*X.T.dot(X)

# <h4>Get the eigen vectors and then values</h4>
eigen_vectors=np.linalg.eig(S)[1]

# <h4>Applying PCA and retaining only the top 2 attributes that contribute the most</h4>
pca_plotData=data.dot(eigen_vectors[:,0:2])

# <h3>Storing generated labels as a list</h3>
lb = data2[:,0].reshape(data2.shape[0],1)
lb = lb.reshape(len(lb))

# <h3>The final dataframe to be used for plotting</h3>
df_pca = pd.DataFrame(dict(x=list(pca_plotData[:,0]),y=list(pca_plotData[:,1]), labels=lb.reshape(len(lb))))

# <h3>Plot coloring function</h3>
def hexColor():
    return ''.join([random.choice('0123456789ABCDEF') for x in range(6)])

print("Jaccard Index: "+str(mt.jaccardIndex()))
print("Rand Index: "+str(mt.randIndex()))
print(df_pca['labels'].groupby(df_pca['labels']).describe()['count'])

# <h3>The plotting function</h3>
################################################################################
#  Plotting the dataframe with labels
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(len(upid)):
    ax1.scatter(df_pca[df_pca['labels']==i]['x'], df_pca[df_pca['labels']==i]['y'], s=10, color=("#"+hexColor()), label=i+1)
plt.legend(loc='upper left')
plt.title("HAC plot on "+file)
plt.savefig('HAC_plots_'+file[:-4]+'.pdf')
plt.show()
