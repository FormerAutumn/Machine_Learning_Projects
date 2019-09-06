
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


#load the dataset from the local url

'''
    RET:
        initdata: the initial data without any operations
        dataset: the image information set
        set_k: the k we choose
        label: the answer of cluster
'''

def loaddata( dataurl, __gets, __result ):
    initdata = sio.loadmat( dataurl )
    keys = list( initdata.keys() )
    values = list( initdata.values() )
    dataset = initdata[__gets]
    label = initdata[__result]
    set_k = np.max( label )
    return initdata, dataset, set_k, label


# In[3]:


#calculate the dist btw matrix_A and matrix_B

'''
    RET:
        dist between Mat_A and Mat_B
'''

def calcdist( Mat_A, Mat_B ):
    return np.sqrt( np.sum( np.power( Mat_A-Mat_B, 2 ) ) )


# In[4]:


#choose the initial k cluster_centers

'''
    RET:
        initcent: the k initial centers
'''

def randcents( dataset, k, label ):
    _M = []
    initcent = []
    n = dataset.shape[0]
    for i in range(n):
        nw = label[i]
        flg = 0
        for j in range(len(_M)):
            if _M[j] == nw:
                flg = 1
        if flg == 0:
            initcent.append(dataset[i])
            _M.append(nw)
    return initcent


# In[5]:


#the function for kmeans

'''
    RET:
        cents: the last centers we choose
        result_mat: the cluster information of each sample
        init_cents: the initial clusters we choose
'''

def K_Means( dataset, k, result_label, mx_it_tme = 27 ):
    
    #get the init_cents and copy it for loss it
    cents = randcents( dataset, k, result_label )
    init_cents = cents.copy()
    
    '''
    print("The init cents")
    print(type(cents))
    for i in cents:
        print(i)
    print("End of check")
    '''
    #get the rows of dataset
    rows, cols = dataset.shape
    #modify means the centers is changed or not
    modify = True
    #iter time
    it_count = 0
    #initial the result_mat
    #result_mat[i,0] means the cluster of i sample, result_mat[i,1] means the dist from i to it's center
    result_mat = np.mat( np.zeros((rows,2)) )
    result_mat = np.full( result_mat.shape, -1 )
    
    while modify and it_count < mx_it_tme:
        #count the times of iterations
        it_count += 1
        #for every loop, wo suppose the centers is fixed
        modify = False
        
        for i in range(rows):
            
            #mndis means the dist from i to it's center
            mndis = np.inf
            #mnind means the cluster which i belongs
            mnind = 0
            
            #calculate the dist for i and each cluster_center( changing )
            for j in range(k):
                dis = calcdist( dataset[i], cents[j] )
                #renew the mndis and dis
                if dis < mndis:
                    mnind = j
                    mndis = dis
                    
            #cahnge the i's cluster to a more rational choice
            if result_mat[i,0] != mnind:
                modify = True #wo need to change the i's cluster boviously
            #renew the result_mat[i]
            result_mat[i, :] = mnind, mndis**2
        
        #renew the k clusters
        for i in range(k):
            #sm means the sum of dist
            sm = np.full( dataset[0].shape, 0 )
            flg = 0
            for j in range(rows):
                if result_mat[j,0] == i:
                    sm = sm + dataset[j]
                    flg += 1
            #if the cluster is no empty, renew the centsdata
            if flg != 0:
                cents[i] = sm / flg
        
    return cents, result_mat, init_cents


# In[6]:


#use the api above to complete the K-Mwans

'''
    RET:
        cents: the last cluster we choose
        result_mat: the cluster information for each sample
        init_cents: the initial cluster we choose
'''

def main( filename, __gets, __result ):
    
    initdata, dataset, set_k, label = loaddata( filename, __gets, __result )
    
    rows, cols = dataset.shape
    cents, result_mat, init_cents = K_Means( dataset, set_k, label )
    
    clusters = []
    '''
    clusters.append(-1)
    for i in range(rows):
        bel = result_mat[i,0]
        flg = 1
        for j in clusters:
            if bel == j:
                flg = 0
        if flg:
            clusters.append(bel)
            img = dataset[i].reshape(32,32).T
            plt.subplot( 4, 5, len(clusters)-1 )
            plt.imshow(img)
            plt.axis('off')
    plt.show()
    print(len(clusters)-1)
    for i in clusters:
        print(i, end=" ")
    print()
    '''
    return cents, result_mat, init_cents


# In[13]:


#result test
dataURL = ['data/COIL20.mat', 'data/Yale_32x32.mat', 'data/data_batch_1.mat']
_gets = [ 'fea', 'data' ]
_result = [ 'gnd', 'labels' ]
#change the index of _gets and _result for different dataset
#the data_batch_1 cluster is on another program for it is a colorful image
initdata, dataset, set_k, label = loaddata( dataURL[0], _gets[0], _result[0] )
cents, result_mat, init_cents = main(dataURL[0], _gets[0], _result[0] )


# In[14]:


#the RI_estimate
def Estimate_RI( init_labels, lst_labels ):
    TP = TN = FP = FN = 0
    n = len(init_labels)
    for i in range(n):
        for j in range(i+1,n):
            if init_labels[i]==init_labels[j] and lst_labels[i,0]==lst_labels[j,0]:
                TP += 1
            if init_labels[i]!=init_labels[j] and lst_labels[i,0]!=lst_labels[j,0]:
                TN += 1
            if init_labels[i]!=init_labels[j] and lst_labels[i,0]==lst_labels[j,0]:
                FP += 1
            if init_labels[i]==init_labels[j] and lst_labels[i,0]!=lst_labels[j,0]:
                FN += 1
        
    ratio = ( TP + TN ) / ( TP + TN + FP + FN )
    return ratio


# In[15]:


#RI_Estimate
precent = Estimate_RI( label, result_mat )
print("The RI estimate:", precent)


# In[16]:


#the ARI estimate
label_true = label.ravel()
label_pred = []
for i in range(len(result_mat)):
    label_pred.append( result_mat[i,0] )
ratio = metrics.adjusted_rand_score( label_true, label_pred )
print("The ARI estimate:", ratio)


# In[11]:


#the accuracy
def brute_force_match( label_true, label_pred ):
    n = len(label_true)
    ct = 0
    for i in range(n):
        if label_true[i] == label_pred[i]+1:
            ct += 1
    ratio = ct / n
    return ratio


# In[17]:


ratio = brute_force_match( label_true, label_pred )
print("The accuracy:", ratio)

