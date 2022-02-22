# -*- coding: utf-8 -*-
"""
@author: XieQi
"""

import numpy as np
import scipy.io as sio    
import re
import RealDataReader_plus as Crd
import tensorflow as tf
import MyLib as ML
import random 
import BMHFnet as BMHFnet

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode', 'testPavia', 
                           'train or testAll or testPavia')
# 输出的维数
tf.app.flags.DEFINE_integer('outDim', 30,
                           'output channel number') 
# 输出的维数
tf.app.flags.DEFINE_integer('ratio', 8,
                           'output channel number') 
# 增加的秩
tf.app.flags.DEFINE_integer('upRank', 16,
                           'upRank number') 
# lambda
tf.app.flags.DEFINE_float('lam1', 0.1,
                           'lambda') 
# lambda
tf.app.flags.DEFINE_float('lam2', 0.1,
                           'lambda') 
# HSI网络的层数
tf.app.flags.DEFINE_integer('HSInetL', 20,
                           'layer number of HSInet') 
# 子网络的层数
tf.app.flags.DEFINE_integer('subnetL', 3,
                           'layer number of subnet') 
# 学习率
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                           'learning_rate') 
# epoch
tf.app.flags.DEFINE_integer('epoch', 80,
                           'epoch') 

# 训练过程数据的存放路劲
tf.app.flags.DEFINE_string('train_dir', 'temp/RealDataExam/',
                           'Directory to keep training outputs.')
# 测试过程数据的存放路劲
tf.app.flags.DEFINE_string('test_dir', 'TestResult/RealDataExamFinal_10rand/',
                           'Directory to keep eval outputs.')
# 训练好的初始的存放路劲
tf.app.flags.DEFINE_string('load_dir', 'temp/RealDataExam/',
                           'Directory to keep inital model.')
# 数据参数
tf.app.flags.DEFINE_integer('image_size', 64, 
                            'Image side length.')
tf.app.flags.DEFINE_integer('BatchIter', 2000,
                            """number of training h5 files.""")
#tf.app.flags.DEFINE_integer('num_patches', 200,
#                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
# GPU设备数量（0代表CPU）
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')



#==============================================================================#
#train
def train():
    random.seed( 1 )        
    ## 变为4D张量 banchsize H W C
                
    X       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # supervised label (None,64,64,3)
    Y       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 4))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size/FLAGS.ratio, FLAGS.image_size/FLAGS.ratio, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    A       = tf.placeholder(tf.float32, shape=(1, 1, 4, FLAGS.outDim))  # supervised data (None,64,64,3)
    B       = tf.placeholder(tf.float32, shape=(1, 1, FLAGS.upRank, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    C       = tf.placeholder(tf.float32, shape=(12, 12, 1, 1)) # supervised detail layer (None,64,64,3)

    

    outX, ListX, YA, E, HY, CX  = BMHFnet.HSInet(Y, Z, A, B, C, FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL,FLAGS.ratio,Yfram = [0,4,1])
    
    # loss function
    loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.lam1*tf.reduce_mean(tf.square(X - YA))+ FLAGS.lam2*tf.reduce_mean(tf.square(E))  # supervised MSE loss
    for i in range(FLAGS.HSInetL-1):
        loss = loss + FLAGS.lam1*tf.reduce_mean(tf.square(X - ListX[i]))
#        loss = loss + FLAGS.lam2*tf.reduce_mean(tf.square(ListE[i]))
    
    
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
        
        allX, V, S, Wave, mmC, mmW = Crd.all_train_data_in()
        
#        val_h5_X, val_h5_Y, val_h5_Z, val_h5_Yh, val_h5_A, val_h5_B, val_h5_C= Crd.eval_data_in(allC, FLAGS.batch_size)
#        val_h5_A = np.expand_dims(np.expand_dims(val_h5_A, axis = 0), axis = 0)
#        val_h5_B = np.expand_dims(np.expand_dims(val_h5_B, axis = 0), axis = 0)
                
        for j in range(start_point,epoch):   # epoch 1 to 120X[:,idx] 

#            if j+1 >(epoch/3):  # reduce learning rate
#                lr_ = FLAGS.learning_rate*0.1
            if j+1 >(4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            Training_Loss = 0                        
            for num in range(FLAGS.BatchIter):    # h5 files 每个数据传进来搞一搞，算一下R，但是因为每个数据的batch有很多重叠，所以似乎有很多重复计算
#                batch_X, batch_Y, batch_Z, batch_Yh, batch_A, batch_B, batch_C = Crd.train_data_in(allX, allR, allC, FLAGS.image_size, FLAGS.batch_size)
                batch_X, batch_Y, batch_Z, batch_Yh, batch_A, batch_B, batch_C = Crd.train_data_in(allX, V, S, Wave, mmC, mmW, 
                                                                                                   FLAGS.image_size,  FLAGS.batch_size, FLAGS.upRank)
                batch_A = np.expand_dims(np.expand_dims(batch_A, axis = 0), axis = 0)
                batch_B = np.expand_dims(np.expand_dims(batch_B, axis = 0), axis = 0)

                _,lossvalue = sess.run([g_optim,loss],feed_dict={X:batch_X, Y:batch_Y, Z:batch_Z, A:batch_A, B:batch_B, C:batch_C, lr:lr_})  # 这应该是核心的一步
                
                Training_Loss += lossvalue  # training loss

                _,ifshow = divmod(num+1,200) 
                if ifshow ==1:
                    pred_X,pred_ListX,pred_HY, Pred_YA = sess.run([outX, ListX, HY, YA], feed_dict={Y:batch_Y,Z:batch_Z,A:batch_A,B:batch_B,C:batch_C})
#                    psnr = skimage.measure.compare_psnr(batch_X,pred_X)
#                    ssim = skimage.measure.compare_ssim(batch_X,pred_X,multichannel=True)
                    CurLoss = Training_Loss/(num+1)
                    model_name = 'model-epoch'   # save model
                    print('...Training with the %d-th banch ....'%(num+1))  
                    print ('.. %d epoch training, learning rate = %.8f, Training_Loss = %.4f, ..'
                           %(j+1, lr_, CurLoss))
                    
                    showF = [5,15,25]
                    showX = ML.get3band_of_tensor2(batch_X,nbanch=0, nframe=showF)
#                    maxS = np.max(showX)
#                    minS = np.min(showX)
                    toshow  = np.hstack((ML.get3band_of_tensor2(Pred_YA,nbanch=0, nframe=showF),
                                         ML.get3band_of_tensor2(pred_ListX[FLAGS.HSInetL-2],nbanch=0, nframe=showF),
                                         ML.get3band_of_tensor2(pred_X,nbanch=0, nframe=showF)))
                    toshow2 = np.hstack((ML.setRange((ML.get3band_of_tensor(batch_Y, 0, [2,1,0]))),
                                         ML.setRange(ML.get3band_of_tensor2(pred_HY, 0, [2,1,0])),
                                         showX))
                    toshow  = np.vstack((toshow,toshow2))
                    ML.imshow2(toshow)
#                    ML.imwrite(toshow,('tempIm_train/epoch%d_num%d.png'%(j+1,num+1)))
    
            CurLoss = Training_Loss/(num+1)

            model_name = 'model-epoch'   # save model
#            save_path_full = os.path.join(save_path, model_name) #save_path+ model_name
            save_path_full = save_path+ model_name
            saver.save(sess, save_path_full, global_step = j+1)

            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)

            print('=========================================')     
            print('*****************************************')
                              
#==============================================================================#

def testAll():

    data = sio.loadmat('rowData/CASI_Houston/AllRC')
    allR = data['allR']
    allC = data['allC']
    
    data = sio.loadmat('RealData/VS')
    V = data['V']
#    S = data['S']
    
    data = sio.loadmat('RealData/testX')
            
    orlinX  = data['X']   
    orlinX  = orlinX[0:336,0:880,:]
#    
#    X       = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # supervised label (None,64,64,3)
    Y       = tf.placeholder(tf.float32, shape=(1, 336, 880, 4))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(1, 42, 110, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    A       = tf.placeholder(tf.float32, shape=(1, 1, 4, FLAGS.outDim))  # supervised data (None,64,64,3)
    B       = tf.placeholder(tf.float32, shape=(1, 1, FLAGS.upRank, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    C       = tf.placeholder(tf.float32, shape=(12, 12, 1, 1)) # supervised detail layer (None,64,64,3)

    
    outX, ListX, YA, E, HY, CX  = BMHFnet.HSInet(Y, Z, A, B, C, FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL,FLAGS.ratio,Yfram = [0,4,1])
    
 
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    
#    iniC = np.ones([12,12])/12/12
    iniC = ML.gauss(12, 5)
    ML.mkdir(FLAGS.test_dir)

   
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        
        for randNum in range(10):   
            orlC = allC[randNum]
            R    = allR[randNum]
            tempDir = FLAGS.test_dir+('round%s'%(randNum))
            inY  = np.tensordot(orlinX, R, [2,0])
            inX = np.tensordot(orlinX,V, [2,0])
            [H,W,_] = inX.shape
            inX  = np.expand_dims(inX, axis = 0)
            inY  = np.expand_dims(inY, axis = 0)
            
            tempX = ML.mypadding(inX, 2)
            inZ   = np.zeros((1, int(H/8), int(W/8), 30),'f')
            for j in range(12):
                for k in range(12):
                    inZ = inZ + tempX[:,j:j+H:8,k:k+W:8,:]*orlC[j,k]    
                    
#                    inC = ML.getC2(inY,np.squeeze(inZ, axis = 0), allC, 48, 32)
            inC = ML.getC(np.squeeze(inY, axis = 0),np.squeeze(inZ, axis = 0), iniC, 12, 8)
            showC =  ML.normalized(np.squeeze(np.squeeze(inC,axis = 3), axis=2))
            ML.imshow(np.hstack((ML.normalized(iniC), showC, ML.normalized(orlC))))

            tempY = ML.mypadding(inY,2)
            C_Y  = np.zeros((1, int(H/8), int(W/8), 4),'f')
            for j in range(12):
                for k in range(12):
                    C_Y = C_Y + tempY[:,j:j+H:8,k:k+W:8,:]*inC[j,k,0,0]
            
#            print(inZ.shape)
            uX      = np.reshape(inZ, [int(H/8)*int(W/8), 30])
            uY      = np.reshape(C_Y, [int(H/8)*int(W/8), 4])  
            YTY     = uY.T.dot(uY)
            inA       = np.linalg.inv(YTY).dot(uY.T.dot(uX))
            
            E       = uX - uY.dot(inA)
            
            u,s,vh  = np.linalg.svd(E, full_matrices=False)
            u       = u[:,0:FLAGS.upRank:1]
            s       = s[0:FLAGS.upRank:1]
            vh      = vh[0:FLAGS.upRank:1,:]
            signB   = np.sign(np.sum(vh,1))
        #    print(signB.shape)
            inB       = (vh.T*signB).T
            
            inA = np.expand_dims(np.expand_dims(inA, axis = 0), axis = 0)
            inB = np.expand_dims(np.expand_dims(inB, axis = 0), axis = 0)
            
            pred_X = sess.run(outX,feed_dict={Y:inY,Z:inZ,A:inA,B:inB,C:inC})  
            sio.savemat(tempDir, {'outX': pred_X, 'RGB':inY}) 
            
            print(('Round %s: '%(randNum))  +  ' done!')


def testPavia():

    data = sio.loadmat('RealData/Pavia/XYZVS3')
    inY = data['Y']
    inZ = data['Z']
    V   = data['V']
    [H,W,t] = inY.shape
    [h,w,T] = inZ.shape
    inZ = np.tensordot(inZ,V, [2,0])
    
    ChgP = 185    
    
    Y       = tf.placeholder(tf.float32, shape=(1, H, W, 4))  # supervised data (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(1, h, w, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    A       = tf.placeholder(tf.float32, shape=(1, 1, 4, FLAGS.outDim))  # supervised data (None,64,64,3)
    B       = tf.placeholder(tf.float32, shape=(1, 1, FLAGS.upRank, FLAGS.outDim)) # supervised detail layer (None,64,64,3)
    C       = tf.placeholder(tf.float32, shape=(12, 12, 1, 1)) # supervised detail layer (None,64,64,3)


    outX, ListX, YA, E, HY, CX  = BMHFnet.HSInet(Y, Z, A, B, C, FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL,FLAGS.ratio,Yfram = [0,4,1])
    
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    
    iniC = ML.gauss(12, 5)
    ML.mkdir(FLAGS.test_dir)

    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        
        
        inC = ML.getC(inY,inZ, iniC, 12, 8)
        inZ  = np.expand_dims(inZ, axis = 0)
        inY  = np.expand_dims(inY, axis = 0)
        
        tempY = ML.mypadding(inY,2)
        C_Y  = np.zeros((1, int(H/8), int(W/8), 4),'f')
        for j in range(12):
            for k in range(12):
                C_Y = C_Y + tempY[:,j:j+H:8,k:k+W:8,:]*inC[j,k,0,0]
        
#            print(inZ.shape)
        uX      = np.reshape(inZ, [int(H/8)*int(W/8), 30])
        uY      = np.reshape(C_Y, [int(H/8)*int(W/8), 4])  
        YTY     = uY.T.dot(uY)
        inA       = np.linalg.inv(YTY).dot(uY.T.dot(uX))
        
        E       = uX - uY.dot(inA)
        
        u,s,vh  = np.linalg.svd(E, full_matrices=False)
        u       = u[:,0:FLAGS.upRank:1]
        s       = s[0:FLAGS.upRank:1]
        vh      = vh[0:FLAGS.upRank:1,:]
        signB   = np.sign(np.sum(vh,1))
    #    print(signB.shape)
        inB       = (vh.T*signB).T
        
        inA = np.expand_dims(np.expand_dims(inA, axis = 0), axis = 0)
        inB = np.expand_dims(np.expand_dims(inB, axis = 0), axis = 0)
        
        pred_X, pred_YA = sess.run([outX,YA],feed_dict={Y:inY*ChgP,Z:inZ*ChgP,A:inA,B:inB,C:inC})  
        pred_X = np.squeeze(pred_X, axis = 0)/ChgP
        inY    = np.squeeze(inY, axis = 0)
        inZ    = np.squeeze(inZ, axis = 0)
        pred_X = np.tensordot(pred_X, V, [2,1])
        
        
        pred_YA = np.squeeze(pred_YA, axis = 0)/ChgP
        pred_YA = np.tensordot(pred_YA, V, [2,1])
        ML.mkdir('TestResult/RealDataExamFinal_Pavia3/')
        sio.savemat('TestResult/RealDataExamFinal_Pavia3/result', {'outX': pred_X, 'RGB':inY, 'Zsi':inZ, 'YA': pred_YA}) 
        
        print( ' done!')

def getCY(allR, inY, inZ):
    C_Y = np.zeros((1, 16, 16, 3),'f')
    for j in range(32):
        for k in range(32):
            C_Y = C_Y + inY[:,j:512:32,k:512:32,:]/32/32
    uX      = np.reshape(inZ, [16*16, 31])
    uY      = np.reshape(C_Y, [16*16*3])
    XR      = uX.dot(allR)
    XR      = np.reshape(XR,[16*16*3,84])
    
    XR      = np.hstack((XR, -1*np.ones([16*16*3,1])))
    XR2     = XR.T.dot(XR)
    XRY     = XR.T.dot(uY)
    
    coef    = np.ones([85,1])/50
    
    for i in range(100000):
        coef = np.maximum(  coef - 0.00002/(i+1)*(XR2.dot(coef)-XRY)  ,0)
    
    b = coef[84,0]
    
    a = 1/np.sum(coef[0:84,0])
    
    C_Y = (C_Y+b)*a
    return C_Y, a, b

def getinAB(C_Y, inZ):
    # 估计测试集的AB
#        C_Y = np.zeros((1, 16, 16, 3),'f')
#        for j in range(32):
#            for k in range(32):
#                C_Y = C_Y + inY[:,j:512:32,k:512:32,:]/32/32
                    
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
    #    print(signB.shape)
        inB       = (vh.T*signB).T
        return inA, inB
    

if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    with tf.device(dev):
        if FLAGS.mode == 'testAll': # simple test
            testAll()
        elif FLAGS.mode == 'train': # train
            train()
        elif FLAGS.mode == 'testPavia': # train
            testPavia()

    
  
