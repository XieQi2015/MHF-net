# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:59:54 2018
Main Code
@author: XieQi
"""
#import h5py
import os
import skimage.measure
import numpy as np
import scipy.io as sio    
import re
import CAVE_dataReader as Crd
import tensorflow as tf
import MyLib as ML
import random 
import MHFnet as MHFnet

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS

# Mode：train, test, testAll for test all sample
tf.app.flags.DEFINE_string('mode', 'test', 
                           'train or test or testAll.')

# output channel number
tf.app.flags.DEFINE_integer('outDim', 31,
                           'output channel number') 
# the rank of Y_hat
tf.app.flags.DEFINE_integer('upRank', 12,
                           'upRank number') 
# alpha
tf.app.flags.DEFINE_float('alpha', 0.1,
                           'lambda') 
# beta
tf.app.flags.DEFINE_float('beta', 0.01,
                           'lambda') 
# the stage number
tf.app.flags.DEFINE_integer('HSInetL', 20,
                           'layer number of HSInet') 
# the level number of the resnet for proximal operator
tf.app.flags.DEFINE_integer('subnetL', 2,
                           'layer number of subnet') 
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                           'learning_rate') 
# epoch number
tf.app.flags.DEFINE_integer('epoch', 31,
                           'epoch') 
# path of testing sample
tf.app.flags.DEFINE_string('test_data_name', 'TestSample',
                           'Filepattern for eval data') 
# path of training result
tf.app.flags.DEFINE_string('train_dir', 'temp/TrainedNet/',
                           'Directory to keep training outputs.')
# path of the testing result 
tf.app.flags.DEFINE_string('test_dir', 'TestResult/Result_1/',
                           'Directory to keep eval outputs.')
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
#test
def test():
    data = sio.loadmat(FLAGS.test_data_name)
    Y    = data['RGB']
    Z    = data['Zmsi']
    X    = data['msi']   
            
    ## banchsize H W C
    inY = np.expand_dims(Y, axis = 0)
    inY = tf.to_float(inY);
    
    inZ = np.expand_dims(Z, axis = 0)
    inZ = tf.to_float(inZ);
    
    inX = np.expand_dims(X, axis = 0)
    inX = tf.to_float(inX);
    
    iniA     = 0
    iniUp3x3 = 0
    
    outX, X1, YA, _, HY = MHFnet.HSInet(inY,inZ, iniUp3x3, iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    
    with tf.Session(config=config) as sess:        
       ckpt = tf.train.latest_checkpoint(save_path)
       saver.restore(sess, ckpt) 
       pred_X,pred_YA,pred_HY,inX = sess.run([outX, YA, HY, inX])     
    
    toshow  = np.hstack((ML.normalized(ML.get3band_of_tensor(pred_HY)),ML.get3band_of_tensor(pred_YA)))
    toshow2 = np.hstack((ML.get3band_of_tensor(pred_X),ML.get3band_of_tensor(inX)))
    toshow  = np.vstack((toshow,toshow2))
    ML.imshow(toshow)
    ML.imwrite(toshow)

#==============================================================================#
#train
def train():
    Crd.PrepareDataAndiniValue()   
    random.seed( 1 )  

   
    ## 变为4D张量 banchsize H W C
    iniData1 = sio.loadmat("CAVEdata/iniA")
    iniA         = iniData1['iniA'] 
    iniData2= sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']
                
    X       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # HrHS (None,96,96,31)
    Y       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))  # HrMS (None,96,96,3)
    Z       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size/32, FLAGS.image_size/32, FLAGS.outDim)) # LrHS (None,3,3,31)

    outX, ListX, YA, E, HY  = MHFnet.HSInet(Y, Z, iniUp3x3,iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
    # loss function
    loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.alpha*tf.reduce_mean(tf.square(X - YA))+ FLAGS.beta*tf.reduce_mean(tf.square(E))  # supervised MSE loss
    for i in range(FLAGS.HSInetL-1):
        loss = loss + FLAGS.alpha*tf.reduce_mean(tf.square(X - ListX[i]))
    
    
    lr_ = FLAGS.learning_rate
    lr  = tf.placeholder(tf.float32 ,shape = [])
    g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam
 
    
    # saver setting
    saver = tf.train.Saver(max_to_keep = 5)
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)    
    save_path = FLAGS.train_dir
    ML.mkdir(save_path)
    epoch = int(FLAGS.epoch)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state('temp/TrainedNet/'):   # load previous trained model
            ckpt = tf.train.latest_checkpoint('temp/TrainedNet/')
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d",ckpt)
            if len(ckpt_num)==3:
                start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
            elif len(ckpt_num)==2:
                start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success")

        else:
            print("re-training")
            start_point = 0
        
        allX, allY = Crd.all_train_data_in()
        
        val_h5_X, val_h5_Y, val_h5_Z= Crd.eval_data_in(20)
                
        for j in range(start_point,epoch):   

            if j+1 >(4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            Training_Loss = 0                        
            for num in range(FLAGS.BatchIter):    
                batch_X, batch_Y,batch_Z = Crd.train_data_in(allX, allY, FLAGS.image_size, FLAGS.batch_size)

                _,lossvalue = sess.run([g_optim,loss], feed_dict={X:batch_X,Y:batch_Y,Z:batch_Z,lr:lr_}) 
                
                Training_Loss += lossvalue  # training loss
                
                # visual output
                _,ifshow = divmod(num+1,200)  
                if ifshow ==1:
                    pred_X,pred_ListX,pred_HY,Pred_YA = sess.run([outX, ListX, HY, YA], feed_dict={Y:batch_Y,Z:batch_Z})
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
            save_path_full = save_path+ model_name
            saver.save(sess, save_path_full, global_step = j+1)

            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)

            Validation_Loss,pred_val  = sess.run([loss,outX],  
                                                            feed_dict={X:val_h5_X,Y:val_h5_Y,Z:val_h5_Z,lr:lr_})
            psnr_val = skimage.measure.compare_psnr(val_h5_X,pred_val)
            ssim_val = skimage.measure.compare_ssim(val_h5_X,pred_val, multichannel=True)
            toshow = np.hstack(( ML.normalized(ML.get3band_of_tensor(pred_val, nbanch=18, nframe=[0,15,30])),
                                 ML.normalized(ML.get3band_of_tensor(val_h5_X, nbanch=18, nframe=[0,15,30]))))

            
            print ('The %d epoch is finished, learning rate = %.8f, Training_Loss = %.4f, Validation_Loss = %.4f, PSNR = %.4f, SSIM = %.4f, PSNR_Valid = %.4f,SSIM_Valid = %.4f.' %
                  (j+1, lr_, CurLoss, Validation_Loss, psnr, ssim, psnr_val,ssim_val))
            ML.imshow(toshow)
            print('=========================================')     
            print('*****************************************')
                              
#==============================================================================#

def testAll():
    ## test all the testing samples
    iniA         = 0 
    iniUp3x3     = 0
    Y       = tf.placeholder(tf.float32, shape=(1, 512, 512, 3))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(1, 512/32, 512/32, FLAGS.outDim))
    outX, X1, YA, _, HY = MHFnet.HSInet(Y, Z, iniUp3x3,iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    ML.mkdir(FLAGS.test_dir)
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        for root, dirs, files in os.walk('CAVEdata/X/'):
            for i in range(32):       
                data = sio.loadmat("CAVEdata/Y/"+files[i])
                inY  = data['RGB']
                inY  = np.expand_dims(inY, axis = 0)
                data = sio.loadmat("CAVEdata/Z/"+files[i])
                inZ  = data['Zmsi']
                inZ  = np.expand_dims(inZ, axis = 0)
                pred_X,ListX,pred_HY,pred_YA = sess.run([outX, X1, HY, YA],feed_dict={Y:inY,Z:inZ})  
                pred_Lr = ListX[FLAGS.HSInetL-2]
                sio.savemat(FLAGS.test_dir+files[i], {'outX': pred_X,'outLR': pred_Lr,'outHY': pred_HY, 'outYA':pred_YA})     
                print(files[i] + ' done!')


if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    with tf.device(dev):
        if FLAGS.mode == 'test': # simple test
            test()
        elif FLAGS.mode == 'testAll': # test all
            testAll()
        elif FLAGS.mode == 'train': # train
            train()

    
  
