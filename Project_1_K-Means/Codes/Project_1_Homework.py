
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[4]:


#load the dataset from the local url
def loaddata( dataurl ):
    initdata = sio.loadmat( dataurl )
    keys = list( initdata.keys() )
    values = list( initdata.values() )
    dataset = initdata['fea']
    label = initdata['gnd']
    set_k = np.max( label )
    return initdata, dataset, set_k, label


# In[5]:


#calculate the dist btw matrix_A and matrix_B
def calcdist( Mat_A, Mat_B ):
    return np.sqrt( np.sum( np.power( Mat_A-Mat_B, 2 ) ) )


# In[6]:


def randcents( dataset, k ):
    initcent = []
    n = dataset.shape[0]
    step = n / k
    for i in range(n):
        if i % step == 0:
            initcent.append(dataset[i])
    return initcent


# In[7]:


def K_Means( dataset, k, mx_it_tme = 15 ):
    cents = randcents( dataset, k )
    init_cents = cents.copy()
    
    '''
    print("The init cents")
    print(type(cents))
    for i in cents:
        print(i)
    print("End of check")
    '''
    
    rows, cols = dataset.shape
    modify = True
    #iter time
    it_count = 0
    #the init result [0...][0...][...]..[]
    result_mat = np.mat( np.zeros((rows,2)) )
    result_mat = np.full( result_mat.shape, -1 )
    
    while modify and it_count < mx_it_tme:
        it_count += 1
        modify = False
        
        for i in range(rows):
            
            mndis = np.inf
            mnind = 0
            
            for j in range(k):
                dis = calcdist( dataset[i], cents[j] )
                if dis < mndis:
                    mnind = j
                    mndis = dis
            
            if result_mat[i,0] != mnind:
                modify = True
            result_mat[i, :] = mnind, mndis**2
            
        for i in range(k):
            sm = np.full( dataset[0].shape, 0 )
            flg = 0
            for j in range(rows):
                if result_mat[j,0] == i:
                    sm = sm + dataset[j]
                    flg += 1
            if flg != 0:
                cents[i] = sm / flg;
    return cents, result_mat, init_cents


# In[39]:


def main( filename ):
    
    initdata, dataset, set_k, label = loaddata( filename )
    
    rows, cols = dataset.shape
    print(rows)
    cents, result_mat, init_cents = K_Means( dataset, set_k )
    
    #sbhow the result of i_th cluster
    #draw the total clusters
    clusters = []
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
    return cents, result_mat, clusters, init_cents


# In[40]:


dataURL = ['data/COIL20.mat', 'data/Yale_32x32.mat']
initdata, dataset, set_k, label = loaddata( dataURL[1] )
cents, result_mat, clusters, init_cents = main(dataURL[1])


# In[38]:


#test for my stupid
print("The Standard Answer")
ct = 0
n = len(label)
delta1 = delta2 = 0
step = n // set_k
ee = np.sqrt(set_k)
ee = int(ee)
for i in range(ee,1,-1):
    if n % i == 0:
        delta1 = i
        delta2 = set_k / i
        break

for i in range(n):
    if i % step == 0:
        ct += 1
        nw = dataset[i]
        nw = nw.reshape(32,32).T
        plt.subplot(delta1,delta2,ct)
        plt.axis('off')
        plt.imshow(nw)
plt.show()


# In[12]:


def E_value( init_labels, lst_labels ):
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


# In[36]:


precent = E_value( label, result_mat )
print(precent)


# In[37]:


label_true = label.ravel()
label_pred = []
for i in range(len(result_mat)):
    label_pred.append( result_mat[i,0] )
ratio = metrics.adjusted_rand_score( label_true, label_pred )
print(ratio)


# In[10]:


print("The Predict Answer:")
ct = 0
for i in init_cents:
    ct += 1
    img = i.reshape(32,32)
    plt.subplot( 4, 5, ct )
    plt.imshow(img)
    plt.axis('off')
plt.show()

