# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:59:54 2018
使用找到的相机生成R,并用了不同的高斯核来仿真不同的C，这是当前最后的版本
@author: XieQi
"""
#import h5py
import os
import skimage.measure
import numpy as np
import scipy.io as sio    
import re
import CAVE_dataReader_Blind as Crd
import tensorflow as tf
import MyLib as ML
import random 
import BMHFnet as BMHFnet
# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# Mode：train, test, testAll for test all sample
tf.app.flags.DEFINE_string('mode', 'test', 
                           'train or test')
# Prepare Data: if reprepare data samples for training and testing
tf.app.flags.DEFINE_string('Prepare', 'Yes', 
                           'Yes or No')
# output channel number
tf.app.flags.DEFINE_integer('outDim', 31,
                           'output channel number') 
# the rank of Y_hat
tf.app.flags.DEFINE_integer('upRank', 12,
                           'upRank number') 
## alpha
tf.app.flags.DEFINE_float('lam1', 0.1,
                           'lambda') 
# beta
tf.app.flags.DEFINE_float('lam2', 0.1,
                           'lambda') 
# the stage number
tf.app.flags.DEFINE_integer('HSInetL', 20,
                           'layer number of HSInet') 
# the level number of the resnet for proximal operator
tf.app.flags.DEFINE_integer('subnetL', 3,
                           'layer number of subnet') 
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                           'learning_rate') 
# epoch number
tf.app.flags.DEFINE_integer('epoch', 40,
                           'epoch') 
# path of training result
tf.app.flags.DEFINE_string('train_dir', 'temp/CAVE_Exam/',
                           'Directory to keep training outputs.')
# path of the testing result 
tf.app.flags.DEFINE_string('test_dir', 'TestResult/CAVE_Final_10rand/',
                           'Directory to keep eval outputs.')
# the iteration number in each epoch
tf.app.flags.DEFINE_string('load_dir', 'temp/testDiffRDiffC_UD_JustAtrynon/',
                           'Directory to keep inital model.')
# the size of training samples
tf.app.flags.DEFINE_integer('image_size', 96, 
                            'Image side length.')
# the iteration number in each epoch
tf.app.flags.DEFINE_integer('BatchIter', 2000,
                            """number of training h5 files.""")
# the batch size
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
# number of gpus used for training
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
#==============================================================================#
#train
def train():
    
    Crd.PrepareDataAndiniValue(FLAGS.Prepare)    
    random.seed( 1 )        
    ## 变为4D张量 banchsize H W C             
    X       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # supervised label (None,64,64,3)
    Y       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size/32, FLAGS.image_size/32, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    A       = tf.placeholder(tf.float32, shape=(1, 1, 3, FLAGS.outDim))  # supervised data (None,64,64,3)
    B       = tf.placeholder(tf.float32, shape=(1, 1, FLAGS.upRank, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    C       = tf.placeholder(tf.float32, shape=(48, 48, 1, 1)) # supervised detail layer (None,64,64,3)

    
    outX, ListX, YA, E, HY, CX  = BMHFnet.HSInet(Y, Z, A, B, C, FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
    # loss function
    loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.lam1*tf.reduce_mean(tf.square(X - YA))+ FLAGS.lam2*tf.reduce_mean(tf.square(E))  # supervised MSE loss
    for i in range(FLAGS.HSInetL-1):
        loss = loss + FLAGS.lam1*tf.reduce_mean(tf.square(X - ListX[i]))   
    
    lr_ = FLAGS.learning_rate
    lr  = tf.placeholder(tf.float32 ,shape = [])
    g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam # 在这里告诉 TensorFolw 要优化的是谁？
 
    
    # 固定格式
    saver = tf.train.Saver(max_to_keep = 5)
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)    
    save_path = FLAGS.train_dir
    ML.mkdir(save_path)
    epoch = int(FLAGS.epoch)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state(FLAGS.load_dir):   # load previous trained model
            ckpt = tf.train.latest_checkpoint(FLAGS.load_dir)
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d",ckpt)
            if len(ckpt_num)==3:
                start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
            elif len(ckpt_num)==2:
                start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success")
            random.seed( start_point+1 )   
        else:
            print("re-training")
            start_point = 0
        
        allX, allR, allC = Crd.all_train_data_in()
        
        val_h5_X, val_h5_Y, val_h5_Z, val_h5_Yh, val_h5_A, val_h5_B, val_h5_C= Crd.eval_data_in(allR, allC, FLAGS.image_size, FLAGS.batch_size)
        val_h5_A = np.expand_dims(np.expand_dims(val_h5_A, axis = 0), axis = 0)
        val_h5_B = np.expand_dims(np.expand_dims(val_h5_B, axis = 0), axis = 0)
                
        for j in range(start_point,epoch):   # epoch 1 to 120X[:,idx] 

            if j+1 >(4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            Training_Loss = 0                        
            for num in range(FLAGS.BatchIter):    
                batch_X, batch_Y, batch_Z, batch_Yh, batch_A, batch_B, batch_C = Crd.train_data_in(allX, allR, allC, FLAGS.image_size, FLAGS.batch_size)
                batch_A = np.expand_dims(np.expand_dims(batch_A, axis = 0), axis = 0)
                batch_B = np.expand_dims(np.expand_dims(batch_B, axis = 0), axis = 0)

                _,lossvalue = sess.run([g_optim,loss],feed_dict={X:batch_X, Y:batch_Y, Z:batch_Z, A:batch_A, B:batch_B, C:batch_C, lr:lr_})  # 这应该是核心的一步
                
                Training_Loss += lossvalue  # training loss

                _,ifshow = divmod(num+1,200) 
                if ifshow ==1:
                    pred_X,pred_ListX,pred_HY, Pred_YA = sess.run([outX, ListX, HY, YA], feed_dict={Y:batch_Y,Z:batch_Z,A:batch_A,B:batch_B,C:batch_C})
                    psnr = skimage.measure.compare_psnr(batch_X,pred_X)
                    ssim = skimage.measure.compare_ssim(batch_X,pred_X,multichannel=True)
                    CurLoss = Training_Loss/(num+1)
                    model_name = 'model-epoch'   # save model
                    print('...Training with the %d-th banch ....'%(num+1))  
                    print ('.. %d epoch training, learning rate = %.8f, Training_Loss = %.4f, PSNR = %.4f, SSIM = %.4f..'
                           %(j+1, lr_, CurLoss,  psnr, ssim))
                    
                    showX = ML.get3band_of_tensor(batch_X,nbanch=0, nframe=[0,15,30])
                    maxS = np.max(showX)
                    minS = np.min(showX)
                    toshow  = np.hstack((ML.setRange(ML.get3band_of_tensor(Pred_YA,nbanch=0, nframe=[0,15,30]), maxS, minS),
                                         ML.setRange(ML.get3band_of_tensor(pred_ListX[FLAGS.HSInetL-2],nbanch=0, nframe=[0,15,30]), maxS, minS),
                                         ML.setRange(ML.get3band_of_tensor(pred_X,nbanch=0, nframe=[0,15,30]), maxS, minS)))
                    toshow2 = np.hstack((ML.setRange(ML.normalized(ML.get3band_of_tensor(batch_Y,nbanch=0, nframe=[2,1,0]))),
                                         ML.setRange(ML.normalized(ML.get3band_of_tensor(pred_HY))),
                                         ML.setRange(showX, maxS, minS)))
                    toshow  = np.vstack((toshow,toshow2))
                    ML.imshow(toshow)
#                    ML.imwrite(toshow,('tempIm_train/epoch%d_num%d.png'%(j+1,num+1)))
    
            CurLoss = Training_Loss/(num+1)

            model_name = 'model-epoch'   # save model
#            save_path_full = os.path.join(save_path, model_name) #save_path+ model_name
            save_path_full = save_path+ model_name
            saver.save(sess, save_path_full, global_step = j+1)

            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)

            Validation_Loss,pred_val  = sess.run([loss,outX],feed_dict={X:val_h5_X,Y:val_h5_Y,Z:val_h5_Z, A:val_h5_A, B:val_h5_B, C:val_h5_C,lr:lr_})
            psnr_val = skimage.measure.compare_psnr(val_h5_X,pred_val)
            ssim_val = skimage.measure.compare_ssim(val_h5_X,pred_val, multichannel=True)
            toshow = np.hstack(( ML.normalized(ML.get3band_of_tensor(pred_val, nbanch=7, nframe=[0,15,30])),
                                 ML.normalized(ML.get3band_of_tensor(val_h5_X, nbanch=7, nframe=[0,15,30]))))
            
            print ('The %d epoch is finished, learning rate = %.8f, Training_Loss = %.4f, Validation_Loss = %.4f, PSNR = %.4f, SSIM = %.4f, PSNR_Valid = %.4f,SSIM_Valid = %.4f.' %
                  (j+1, lr_, CurLoss, Validation_Loss, psnr, ssim, psnr_val,ssim_val))
            ML.imshow(toshow)
            print('=========================================')     
            print('*****************************************')
                              
#==============================================================================#

def testAll():

    ## 变为4D张量 banchsize H W C

    data = sio.loadmat('rowData/CAVEdata/AllR')
    allR  = data['R']

    data  = sio.loadmat('rowData/CAVEdata/AllC')
    allC  = data['C']
    
    data = sio.loadmat('rowData/CAVEdata/randMatrix')
    randM1  = data['randM1']
    randM2  = data['randM2']
    
    Y       = tf.placeholder(tf.float32, shape=(1, 512, 512, 3))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(1, 512/32, 512/32, FLAGS.outDim))
    A       = tf.placeholder(tf.float32, shape=(1, 1, 3, FLAGS.outDim))  # supervised data (None,64,64,3)
    B       = tf.placeholder(tf.float32, shape=(1, 1, FLAGS.upRank, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    C       = tf.placeholder(tf.float32, shape=(48, 48, 1, 1)) # supervised detail layer (None,64,64,3)

#    R       = np.squeeze(np.tensordot(allR, coef, (2,0)),axis = (2))

    outX, ListX, YA, _, HY, CX  = BMHFnet.HSInet(Y, Z, A, B, C, FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    
    iniC = np.squeeze(np.tensordot(allC, np.ones([1,20])/20, (2,1)),axis = (2)) # for ini the C

    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data

   
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        
        for randNum in range(10):   
            orlC = np.tensordot(allC, randM1[:,randNum], (2,0))
            R    = np.tensordot(allR, randM2[:,randNum], (2,0))
            tempDir = FLAGS.test_dir+('round%s/'%(randNum))
            ML.mkdir(tempDir)
            for root, dirs, files in os.walk('CAVEdata/X/'):
                for jjj in range(12):
                    i = Ind[0,jjj+20]-1   
                    
                    data = sio.loadmat("CAVEdata/X/"+files[i])
                    inX  = data['msi']                
                    inY  = np.tensordot(inX, R, [2,1])
                    inX  = np.expand_dims(inX, axis = 0)
    
                    tempX = Crd.mypadding(inX)
                    inZ   = np.zeros((1, 16, 16, 31),'f')
                    for j in range(48):
                        for k in range(48):
                            inZ = inZ + tempX[:,j:j+512:32,k:k+512:32,:]*orlC[j,k]
                    
                    inC, inR = ML.getC2(inY,np.squeeze(inZ, axis = 0), allC, 48, 32)

                    if i==0:
                        showC =  ML.normalized(np.squeeze(np.squeeze(inC,axis = 3), axis=2))
                        ML.imshow(np.hstack((ML.normalized(iniC), showC, ML.normalized(orlC))))
                    
                    inY  = np.expand_dims(inY, axis = 0)
                    tempY = Crd.mypadding(inY)
                    C_Y = np.zeros((1, 16, 16, 3),'f')
        
                    for j in range(48):
                        for k in range(48):
                            C_Y = C_Y + tempY[:,j:j+512:32,k:k+512:32,:]*inC[j,k,0,0]
                    
                                
                    uX      = np.reshape(inZ, [16*16, 31])
                    uY      = np.reshape(C_Y, [16*16, 3])  
                    YTY     = uY.T.dot(uY)
                    inA       = np.linalg.inv(YTY).dot(uY.T.dot(uX))
                    
                    E       = uX - uY.dot(inA)
                    
                    u,s,vh  = np.linalg.svd(E, full_matrices=False)
                    u       = u[:,0:12:1]
                    s       = s[0:12:1]
                    vh      = vh[0:12:1,:]
                    signB   = np.sign(np.sum(vh,1))

                    inB       = (vh.T*signB).T
                    
                    inA = np.expand_dims(np.expand_dims(inA, axis = 0), axis = 0)
                    inB = np.expand_dims(np.expand_dims(inB, axis = 0), axis = 0)
                    

                    pred_X, outZ, Xk = sess.run([outX, CX, ListX[-1]],feed_dict={Y:inY,Z:inZ,A:inA,B:inB,C:inC})  
                    sio.savemat(tempDir+files[i], {'outX': pred_X,'Xk':Xk, 'inY':inY, 'inZ':inZ, 'outZ':outZ}) 
                    
                    
                    print(('Round %s: '%(randNum)) +files[i] +  ' done!')
                    

if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    with tf.device(dev):
        if FLAGS.mode == 'test': # simple test
            testAll()
        elif FLAGS.mode == 'train': # train
            train()

    
  
