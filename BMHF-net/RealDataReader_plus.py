


"""
Created on Mon Jul  9 12:13:53 2018
这个是真的会有一点点难，要测的东西很多
@author: XieQi
"""
import numpy as np
import scipy.io as sio  
import MyLib as ML
import random 
#import cv2

def all_train_data_in():
    
    data = sio.loadmat('RealData/VS')
    V = data['V']
    S = data['S']
    
    data = sio.loadmat('RealData/trainX')
    allDataX = []
    allDataX.append(np.tensordot(data['X1'],V, [2,0]))
    allDataX.append(np.tensordot(data['X2'],V, [2,0]))
    
    data = sio.loadmat('RealData/SRFinfo')
    mmC  = data['mmC']
    mmW  = data['mmW']
    mmC[:,0] = mmC[:,0]-10
    mmC[:,1] = mmC[:,1]+10
    mmW[:,0] = mmW[:,0]-10
    mmW[:,1] = mmW[:,1]+10
    
    Wave = np.linspace(364, 1046, 144)
    
    
    return allDataX, V, S, Wave, mmC, mmW


def train_data_in(allX, V, S, Wave, mmC, mmW, sizeI,  batch_size, Rankup=12):

    H1,W1,T   = allX[0].shape    
    H2,W2,T   = allX[1].shape   
    batch_X = np.zeros((batch_size, sizeI, sizeI, T),'f') # 不是最终的大小，还要SVD一下
    batch_Z = np.zeros((batch_size, int(sizeI/8), int(sizeI/8), T),'f')
    
    R = randR(Wave, mmC,mmW) # 这个的随机生成有一定的困难
    R = np.matmul(V.T,R)
    
    C = ML.gauss(12, 1 + np.matlib.rand(1)*9)
    
    for i in range(batch_size):
        # 从两个训练图里选batch
        dataChose = random.randint(0, 2)
        if  dataChose ==0:
            px = random.randint(0,H2-sizeI)
            py = random.randint(0,W2-sizeI)
            subX = allX[1][px:px+sizeI:1,py:py+sizeI:1,:]
        else:
            px = random.randint(0,H1-sizeI)
            py = random.randint(0,W1-sizeI)
            subX = allX[0][px:px+sizeI:1,py:py+sizeI:1,:]
        # 可以加个随机的扰动    
        # 随机转
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        
        for j in range(rotTimes):
            subX = np.rot90(subX)

        for j in range(vFlip):
            subX = subX[:,::-1,:]

        for j in range(hFlip):
            subX = subX[::-1,:,:]

        batch_X[i,:,:,:] = subX      
   
    batch_Y = np.tensordot(batch_X, R, [3,0])
        
    tempX = ML.mypadding(batch_X, 2)
    
    for j in range(12):
        for k in range(12):
            batch_Z = batch_Z + tempX[:,j:j+sizeI:8,k:k+sizeI:8,:]*C[j,k]
             
    uX      = np.reshape(batch_X, [batch_size*sizeI*sizeI, T])
    uY      = np.reshape(batch_Y, [batch_size*sizeI*sizeI, 4])
    YTY     = uY.T.dot(uY)
    A       = np.linalg.inv(YTY).dot(uY.T.dot(uX))
    
    E       = uX - uY.dot(A)
    
    u,s,vh  = np.linalg.svd(E, full_matrices=False)
    u       = u[:,0:Rankup:1]
    s       = s[0:Rankup:1]
    vh      = vh[0:Rankup:1,:]
    signB   = np.sign(np.sum(vh,1))
#    print(signB.shape)
    B       = (vh.T*signB).T
    u       = u*signB
    batch_Yh      = u*s
#    print(batch_Yh.shape)
    batch_Yh      = np.reshape(batch_Yh, [batch_size, sizeI, sizeI, Rankup])    
             
    C = np.expand_dims(C[::-1,::-1], axis = 2)
    C = np.expand_dims(C, axis = 3)
    return batch_X, batch_Y, batch_Z, batch_Yh, A, B, C




def randR(Wave, mmC, mmW):
    sigma = 50
    
    R = np.zeros([144,4])
    for i in range(4):
        center = mmC[i,0] + (mmC[i,1]-mmC[i,0])*np.matlib.rand(1)
        width  = mmW[i,0] + (mmW[i,1]-mmW[i,0])*np.matlib.rand(1)   
#        width  = width/2
        
        ratio = 0.1 + 0.2*np.matlib.rand(1)    
        
        temp = np.array(np.maximum((np.abs( Wave-center)-ratio * width), 0 ).T)[:,0]
        
        thro = np.exp(-( ((1-ratio) * width)**2/2/(sigma**2)))
        
        R[:,i] = np.maximum(np.exp(-( temp**2/2/(sigma**2)))-thro,0)/(1-thro)
        R[:,i] = R[:,i]/np.sum(R[:,i])
        
    
    return R


#allX, V, S, Wave, mmC, mmW = all_train_data_in()
#random.seed( 1 )
#for i in range(10):   
#    R = randR(Wave, mmC, mmW)
#    ML.plot(R)
###    
#  
#batch_X, batch_Y, batch_Z, batch_Yh, A, B, C = train_data_in(allX, V, S, Wave, mmC, mmW, 200,  10)
#
#for nframe in range(1) :
##    nframe = 18
#    print(nframe)
#    X = batch_X[nframe,:,:,:]
#    Y = batch_Y[nframe,:,:,:]
#    Z = batch_Z[nframe,:,:,:]
##    X = np.tensordot(X,np.linalg.inv(S),(2,0))    
#    toshow = np.hstack((ML.normalized(X[:,:,[0,1,2]]),ML.normalized(Y[:,:,[2,1,0]])))
#    ML.imshow2(toshow)
#    ML.imshow2(ML.normalized(Z[:,:,[0,1,2]]))
#    
#print(Z.shape)
#print(A.shape)
#print(B.shape)
#print(C.shape)
