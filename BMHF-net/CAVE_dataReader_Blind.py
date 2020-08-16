"""
Created on Mon Jul  9 12:13:53 2018

@author: XieQi
"""
import os
import numpy as np
import scipy.io as sio
import MyLib as ML
import random
import cv2

def all_train_data_in():
    allDataX = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] 
    for root, dirs, files in os.walk('CAVEdata/X/'):
           for j in range(20):
#                print(Ind[0,j])
                i = Ind[0,j]-1
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
                allDataX.append(inX)
 
           data  = sio.loadmat('CAVEdata/AllR')
           allR  = data['R']
           data  = sio.loadmat('CAVEdata/AllC') 
           allC  = data['C']
    return allDataX, allR, allC

def all_test_data_in():
    allDataX = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('CAVEdata/X/'):
           for j in range(12):
                i = Ind[0,j+20]-1
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
                allDataX.append(inX)
    return allDataX

def train_data_in(allX, allR, allC, sizeI, batch_size, channel=31,dataNum = 20):
#    meanfilt = np.ones((32,32))/32/32
    batch_X = np.zeros((batch_size, sizeI, sizeI, channel),'f')
#    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    batch_Z = np.zeros((batch_size, int(sizeI/32), int(sizeI/32), channel),'f')
    sizeR = allR.shape[2]
    coef    = np.matlib.rand(sizeR)
    coef    = np.sort(coef)
    coef[0,sizeR-1] = 1
    coef[:,1:sizeR] = coef[:,1:sizeR:1] - coef[:,0:sizeR-1:1] 

    R       = np.squeeze(np.tensordot(allR, coef, (2,1)),axis = (2)) + np.random.normal(0,0.075,[3,31])
    
    coef    = np.matlib.rand(20)
    coef    = np.sort(coef)
    coef[0,19] = 1
    coef[:,1:20] = coef[:,1:20:1] - coef[:,0:19:1] 

    C       = np.squeeze(np.tensordot(allC, coef, (2,1)),axis = (2)) 
    
    for i in range(batch_size):
        ind = random.randint(0, dataNum-1)
        X = allX[ind]
        px = random.randint(0,512-sizeI)
        py = random.randint(0,512-sizeI)
        subX = X[px:px+sizeI:1,py:py+sizeI:1,:]
#        subY = Y[px:px+sizeI:1,py:py+sizeI:1,:]
                
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
              
    batch_Y = np.tensordot(batch_X, R,[3,1])    
    tempX = mypadding(batch_X)
    
    for j in range(48):
        for k in range(48):
            batch_Z = batch_Z + tempX[:,j:j+sizeI:32,k:k+sizeI:32,:]*C[j,k]
            
    uX      = np.reshape(batch_X, [batch_size*sizeI*sizeI, channel])
    uY      = np.reshape(batch_Y, [batch_size*sizeI*sizeI, 3])
    YTY     = uY.T.dot(uY)
    A       = np.linalg.inv(YTY).dot(uY.T.dot(uX))
    
    E       = uX - uY.dot(A)
    
    u,s,vh  = np.linalg.svd(E, full_matrices=False)
    u       = u[:,0:12:1]
    s       = s[0:12:1]
    vh      = vh[0:12:1,:]
    signB   = np.sign(np.sum(vh,1))
    B       = (vh.T*signB).T
    u       = u*signB
    batch_Yh      = u*s
    batch_Yh      = np.reshape(batch_Yh, [batch_size, sizeI, sizeI, 12])
    
    C = np.expand_dims(C[::-1,::-1], axis = 2)
    C = np.expand_dims(C, axis = 3)
    
    return batch_X, batch_Y, batch_Z, batch_Yh, A, B, C

def eval_data_in(allR, allC, sizeI=96,batch_size=10):
#    用这两行不需要h5文件
    allX = all_test_data_in()
    return train_data_in(allX, allR, allC, sizeI, batch_size)  
                                
def mypadding(X, paddnum = 8):
    
    sizeI = X.shape[1]
    tempX = np.zeros(np.array(X.shape)+[0,paddnum*2,paddnum*2,0])
    
    tempX[:, paddnum:sizeI+paddnum, paddnum:sizeI+paddnum, :] = X
    # 四个角的padding
    temptemp           = X[:,0:paddnum,0:paddnum,:]
    tempX[:,0:paddnum,0:paddnum,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,sizeI-paddnum:sizeI,sizeI-paddnum:sizeI,:]
    tempX[:,paddnum+sizeI:sizeI+paddnum*2,paddnum+sizeI:sizeI+paddnum*2,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,sizeI-paddnum:sizeI,0:paddnum,:]
    tempX[:,paddnum+sizeI:sizeI+paddnum*2,0:paddnum,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,0:paddnum,sizeI-paddnum:sizeI,:]
    tempX[:,0:paddnum,paddnum+sizeI:sizeI+paddnum*2,:] = temptemp[:,::-1,::-1,:]
    
    # 四个边的padding
    temptemp           = X[:,0:paddnum,:,:]
    tempX[:,0:paddnum,paddnum:sizeI+paddnum,:] = temptemp[:,::-1,:,:]
    
    temptemp           = X[:,sizeI-paddnum:sizeI,:,:]
    tempX[:,paddnum+sizeI:sizeI+paddnum*2,paddnum:sizeI+paddnum,:] = temptemp[:,::-1,:,:]
    
    temptemp           = X[:,:,0:paddnum,:]
    tempX[:,paddnum:sizeI+paddnum,0:paddnum,:] = temptemp[:,:,::-1,:]
    
    temptemp           = X[:,:,sizeI-paddnum:sizeI,:]
    tempX[:,paddnum:sizeI+paddnum,paddnum+sizeI:sizeI+paddnum*2,:] = temptemp[:,:,::-1,:]
    
    return tempX

def PrepareDataAndiniValue(R,C,prepare='Yes'):
    DataRoad = 'CAVEdata/'
    if prepare != 'No':
        print('Generating the training and testing data in folder CAVEdata')
        Ind  = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,32,11,1];
        ML.mkdir(DataRoad+'X/')
        n = 0
        for root, dirs, files in os.walk('rowData/CAVEdata/complete_ms_data/'):
            for i in range(32):
                n=n+1
                print('processing '+ dirs[Ind[i]-1])
                X = readImofDir('rowData/CAVEdata/complete_ms_data/'+dirs[Ind[i]-1]+'/'+dirs[Ind[i]-1])/255

                sio.savemat(DataRoad+'X/'+dirs[Ind[i]-1], {'msi': X})     
                if n<=20:
                    if n==1:
                        allX = np.reshape(X,[512*512,31])
                    else:
                        allX = np.vstack((allX,np.reshape(X,[512*512,31]))) 
            break
        allX = np.matrix(allX)
        sio.savemat(DataRoad+'List', {'Ind': Ind}) 
    else:
        print('Using the prepared data and initial values in folder CAVEdata')

def readImofDir(theRoad):
    X = np.zeros([512,512,31])
    for root, dirs, files in os.walk(theRoad):
        for i in range(31):
            if files[0] == 'Thumbs.db':
                j = i+1
            else:
                j = i
            I = cv2.imread(theRoad+'/'+files[j])
            I =  I.astype('Float32')
            X[:,:,i] = np.mean(I,2)
    return X
    
#
#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
##batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#X = batch_X[0,:,:,:]
#Y = batch_Y[0,:,:,:]
#Z = batch_Z[0,:,:,:]
#toshow = np.hstack((ML.normalized(X[:,:,[0,1,2]]),Y[:,:,[0,1,2]]))
#ML.imshow(toshow)
#ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))
#
#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
##batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#X = batch_X[0,:,:,:]
#Y = batch_Y[0,:,:,:]
#Z = batch_Z[0,:,:,:]
#toshow = np.hstack((ML.normalized(X[:,:,[0,1,2]]),Y[:,:,[0,1,2]]))
#ML.imshow(toshow)
#ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))