
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


#the cmp of two-keys-sort
def by_dis(t):
    return t[0]


# In[3]:


'''
    RET:
        dataset: the initial data (1024,1)
        label_true: the true label
'''

def LoadData( url, feas, labels ):
    
    _dataset = sio.loadmat( url )
    dataset = _dataset[feas]
    label_true = _dataset[labels]
    return dataset, label_true


# In[4]:


'''
    RET:
        dis: the distance between vec_a and vec_b
'''

def CalcDist( vec_a, vec_b ):
    
    dis = np.sqrt( np.sum( (vec_a-vec_b)**2 ) )
    return dis


# In[5]:


'''
    fisher for 2 datasets
    dataset_class: dictionary of dataset (key=label, val=dataset)
    u_class: dictionary of means (key=label, val=u)
'''

def LDA_2_AimD( dataset, label_true, aim_d ):
    
    n = len(dataset)
    #print("dataset shape:", dataset.shape)
    
    #get the label for classify
    label = []
    for i in label_true:
        label.append( i[0] )
    label = np.array( label )
    #print("label shape:", label.shape)
    label = list( set(label) )
    
    #merge the same_label samples, store them into a dictionary
    dataset_class = {}
    for i in label:
        nw_data = np.array( [ dataset[j] for j in range(n) if label_true[j] == i ] )
        dataset_class[i] = nw_data
    
    u = np.mean( dataset, axis = 0 )
    u_class = {}
    
    for i in label:
        u_class[i] = np.mean( dataset_class[i], axis = 0 )
    
    St = np.zeros( (len(u),len(u)) )
    St = np.dot( ( dataset - u ).T, (dataset - u) )
    #print("St shape:", St.shape)
    
    Sw = np.zeros( (len(u),len(u)) )
    for i in label:
        Sw += np.dot( (dataset_class[i]-u_class[i]).T, (dataset_class[i]-u_class[i]) )
    #print("Sw shape:", Sw.shape)
    
    Sb = np.zeros( (len(u),len(u)) )
    Sb = St - Sw
    #print("Sb shape:",Sb.shape)
    
    eig_vals, eig_vecs = np.linalg.eig( np.dot( np.linalg.pinv(Sw), Sb ) )

    sorted_indices = np.argsort( eig_vals )
    W = eig_vecs[ :, sorted_indices[:-aim_d-1:-1] ]
    
    feaset = []
    for i in range(n):
        feaset.append( np.dot(dataset[i], W) )
    return feaset, Sb, Sw, W, dataset_class, u_class
    


# In[6]:


'''
    aim_d: the top aim_d featrue
    RET:
        feaset: the featrue for each sample after lda
'''

def GetFea( dataset, label_true, aim_d, model = 1 ):
    
    if model == 1:
        X = dataset
        y = np.array( [ i[0] for i in label_true ] )
        #lda
        '''
        lda = LinearDiscriminantAnalysis( n_components = aim_d )
        lda.fit( X, y )
        feaset = lda.transform( X )
        '''
        pca = PCA( n_components = aim_d )
        pca.fit( X )
        feaset = pca.fit_transform( X )
    elif model == 0:
        feaset, Sb, Sw, W, dataset_class, u_class = LDA_2_AimD( dataset, label_true, aim_d )

    return feaset


# In[7]:


'''
    k = 7 means the k(17)nn
    The KNN function
    RET:
        
'''

def KNN( feaset, label_true, k = 7 ):
    
    n = len(feaset)
    label_pred = np.zeros(n)
    st_dis = []
    st_clu = []
    
    for i in range(n):
        
        i2each_dis = []
        
        for j in range(n):
            
            if i == j:
                continue
            nwdis = CalcDist( feaset[i], feaset[j] )
            i2each_dis.append( nwdis )
        
        i2each_dis = np.array( i2each_dis )
        i2each_dis_ind = np.argsort( i2each_dis )
        
        sm_dis = []
        sm_clu = []
        vote_result = {}

        for j in range(k):
            sm_dis.append( i2each_dis[ i2each_dis_ind[j] ])
            tp_label = label_true[ i2each_dis_ind[j] ]
            sm_clu.append( tp_label )
            vote_result[tp_label[0]] = vote_result.get( tp_label[0], 0 ) + 1
        
        st_dis.append( sm_dis )
        st_clu.append( sm_clu )
        label_pred[i] = max( vote_result, key=vote_result.get )
        
    return label_pred, st_dis, st_clu


# In[8]:


def ARI( label_true, label_pred ):
    ratio = metrics.adjusted_rand_score( label_true.ravel(), label_pred.ravel() )
    return ratio


# In[9]:


def Accuracy( _label_true, _label_pred ):
    fm = len(_label_true)
    fz = 0
    for i in range(len(_label_true)):
        if _label_true[i] == _label_pred[i]:
            fz += 1
    ratio = fz / fm
    return ratio


# In[10]:


def main( url, feas, labels ):
    
    aim_d = 32
    dataset, label_true = LoadData( url, feas, labels )
    
    for mode in range(2):
        tp_y = []
        for aim_d in X_axis:
            feaset = GetFea( dataset, label_true, aim_d, mode )
            label_pred, st_dis, st_clu = KNN( feaset, label_true )
            ratio = Accuracy( label_true, label_pred )
            tp_y.append(ratio)
        Y_axis[mode] = tp_y
        
    return dataset, label_true, label_pred, st_dis, st_clu


# In[15]:


urls = [ 'data/COIL20.mat', 'data/Yale_32x32.mat' ]
feas = [ 'fea' ]
labels = [ 'gnd' ]

_data, _label = LoadData( urls[0], feas[0], labels[0] )
mxc = max( _label )[0]

Y_axis = {}
X_axis = [ i for i in range(1,mxc)]
print(X_axis)

dataset, label_true, label_pred, st_dis, st_clu = main( urls[0], feas[0], labels[0] )


# In[12]:


print(X_axis)
print(Y_axis[0])
print(Y_axis[1])


# In[16]:


def Draw_result( X_axis, Y_axis ):
    plt.plot( Y_axis[0], c='red', label='LDA' )
    plt.plot( Y_axis[1], c='blue', label='PCA' )
    plt.xlabel('n-component for LDA/PCA')
    plt.xticks(range(-1,len(X_axis)+5))
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[17]:


print("The {} result.".format( urls[0] ) )
Draw_result( X_axis, Y_axis )


# In[11]:


nwct = 0
for i in range(lower_b, upper_b):
    pas = i - lower_b
    if label_true[i] != label_pred[i]:
        nwct += 1
    print(pas, label_true[pas], label_pred[pas])
print(nwct)


# In[5]:


_a = list(np.arange(1,10)) + list(np.arange(10,20,5))
print(_a)

