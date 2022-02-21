# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:13:53 2018
Data Reader
@author: XieQi
"""
import os
import numpy as np
import scipy.io as sio  
#from scipy import misc
import MyLib as ML
import random 
import cv2
def all_train_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('CAVEdata/X/'):
           for j in range(20):
#                print(Ind[0,j])
                i = Ind[0,j]-1
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
#                print(type(inX[1,1,1]))
                allDataX.append(inX)
                data = sio.loadmat("CAVEdata/Y/"+files[i])
                inY  = data['RGB']
                allDataY.append(inY)
                
    return allDataX, allDataY


def all_test_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('CAVEdata/X/'):
           for j in range(12):
#                print(Ind[0,j])
                i = Ind[0,j+20]-1
#                print(i)
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
                allDataX.append(inX)
                data = sio.loadmat("CAVEdata/Y/"+files[i])
                inY  = data['RGB']
                allDataY.append(inY)
    return allDataX, allDataY

def train_data_in(allX, allY, C, sizeI, batch_size, channel=31,dataNum = 20):
#    meanfilt = np.ones((32,32))/32/32
    batch_X = np.zeros((batch_size, sizeI, sizeI, channel),'f')
    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    batch_Z = np.zeros((batch_size, 3, 3, channel),'f')
#    batch_Z = np.zeros((batch_size, sizeI, sizeI, channel))
    for i in range(batch_size):
        ind = random.randint(0, dataNum-1)
        X = allX[ind]
        Y = allY[ind]
        px = random.randint(0,512-sizeI)
        py = random.randint(0,512-sizeI)
        subX = X[px:px+sizeI:1,py:py+sizeI:1,:]
        subY = Y[px:px+sizeI:1,py:py+sizeI:1,:]
                
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        
        # Random rotation
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subY = np.rot90(subY)

        # Random vertical Flip   
        for j in range(vFlip):
            subX = subX[:,::-1,:]
            subY = subY[:,::-1,:]
     
        # Random Horizontal Flip
        for j in range(hFlip):
            subX = subX[::-1,:,:]
            subY = subY[::-1,:,:]

        batch_X[i,:,:,:] = subX
        batch_Y[i,:,:,:] = subY

    for j in range(32):
        for k in range(32):
            batch_Z = batch_Z + batch_X[:,j:512:32,k:512:32,:]*C[k,j]

    return batch_X, batch_Y, batch_Z

def eval_data_in(C, batch_size=20):
    allX, allY = all_test_data_in()
    return train_data_in(allX, allY, C, 96, batch_size, 31, 12)

# Prepare data for training and generate the initial A and upsampling kernals     
def PrepareDataAndiniValue(R,C,prepare='Yes'):
    DataRoad = 'CAVEdata/'
#    folder = os.path.exists(DataRoad+'iniA.mat')
#    if not folder:
    if prepare != 'No':
        print('Generating the training and testing data in folder CAVEdata')
         #random index, firt 20 ind will become traing data, and the others will be testing data
        Ind  = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,32,11,1];

        ML.mkdir(DataRoad+'X/')
        ML.mkdir(DataRoad+'Y/')
        ML.mkdir(DataRoad+'Z/')
        n = 0

        for root, dirs, files in os.walk('rowData/CAVEdata/complete_ms_data/'):
            for i in range(32):
                n=n+1
                Z = np.zeros([16,16,31])
                print('processing '+ dirs[Ind[i]-1])
#                if dirs[Ind[i]-1]=='watercolors_ms':
#                    X = readImofDir('rowData/CAVEdata/complete_ms_data/'+dirs[Ind[i]-1])/255
#                else:
                X = readImofDir('rowData/CAVEdata/complete_ms_data/'+dirs[Ind[i]-1]+'/'+dirs[Ind[i]-1])/255
                Y = np.tensordot(X,R,(2,0))
                for j in range(32):
                    for k in range(32):
                        Z = Z + X[j:512:32,k:512:32,:]*C[k,j]
                sio.savemat(DataRoad+'X/'+dirs[Ind[i]-1], {'msi': X})     
                sio.savemat(DataRoad+'Y/'+dirs[Ind[i]-1], {'RGB': Y})   
                sio.savemat(DataRoad+'Z/'+dirs[Ind[i]-1], {'Zmsi': Z})   
                if n<=20:
                    if n==1:
                        allX = np.reshape(X,[512*512,31])
                        allY = np.reshape(Y,[512*512,3])
                    else:
                        allX = np.vstack((allX,np.reshape(X,[512*512,31])))
                        allY = np.vstack((allY,np.reshape(Y,[512*512,3])))     
            break
        allX = np.matrix(allX)
        allY = np.matrix(allY)
        iniA = (allY.T*allY).I*(allY.T*allX)
        sio.savemat(DataRoad+'iniA', {'iniA': iniA}) 
        
        sio.savemat(DataRoad+'List', {'Ind': Ind}) 
        
        initemp = np.eye(31)
        iniUp1 = np.tile(initemp,[3,3,1,1])
        sio.savemat(DataRoad+'iniUp', {'iniUp1': iniUp1}) 
    else:
        print('Using the prepared data and initial values in folder CAVEdata')
        

def readImofDir(theRoad):
    X = np.zeros([512,512,31])
    for root, dirs, files in os.walk(theRoad):
        files= sorted(files)
        for i in range(31):
            if files[0] == 'Thumbs.db':
                j = i+1
            else:
                j = i
            I = cv2.imread(theRoad+'/'+files[j])
            I =  I.astype('Float32')
            X[:,:,i] = np.mean(I,2)
    return X
        
#X = readImofDir('rowData/CAVEdata/complete_ms_data/'+'watercolors_ms'+'/'+'watercolors_ms')/255
                 

#PrepareDataAndiniValue()    
        

#
#allX, allY = all_train_data_in()
#Ynum = Ynormalize(allY)


#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
#batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#print(type(batch_X[1,1,1,1]))
#for nframe in range(20) :
##    nframe = 18
#    print(nframe)
#    X = batch_X[nframe,:,:,:]
#    Y = batch_Y[nframe,:,:,:]
#    Z = batch_Z[nframe,:,:,:]
#    toshow = np.hstack((ML.normalized(X[:,:,[0,15,30]]),ML.normalized(Y[:,:,[0,1,2]])))
#    ML.imshow(toshow)
#    ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))
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