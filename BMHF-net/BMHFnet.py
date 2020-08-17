# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:05:10 2018
这个版本里，A,B, C, 都是事先算好输入的,是C一步放大32倍, 一般形式，可以应付不同的倍数
@author: XieQi
"""
import tensorflow as tf
import numpy as np
# main HSI net
def HSInet(Y,Z, A, B, C,upRank,outDim,HSInetL,subnetL,ratio=32,Yfram = [0,30,14]):
    
    tranB = tf.transpose(B,perm = [0,1,3,2])
    # 第一层
    YA = tf.nn.conv2d(Y, A, [1,1,1,1], padding='SAME')
    
    #先验的准备
    priorFram = YA[:,:,:,Yfram[0]:Yfram[1]:Yfram[2]]
    priorNum  = priorFram.get_shape().as_list()[3]
    
    initemp  = np.eye(outDim)
    iniUp3x3 = np.tile(initemp,[3,3,1,1])
    
    Cy = iniUp3x3[:,:,0:priorNum,0:priorNum]
    downY4, downY16, _ = downSamAdj('GetPrior', priorFram, Cy, priorNum, ratio)# getPrior for upsample
    priorNum = priorNum*2
    
    
    downX32 = downSam('CX1',YA, C, outDim, ratio)  # 空间下采样
    E  = downX32-Z   # Z上的残差
    G  = UpSam('E1',E, C, Y, downY4, downY16, iniUp3x3, outDim, ratio, priorNum) # 残差上采样
    G  = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
    HY = -G # YB残差
    HY  = resCNNnet(('Pri%s'%(1)),HY,1,upRank, subnetL)
    ListX = []
    
#     第二到倒二层
    for j in range(HSInetL-2):
        HYB= tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
        ListX.append(YA + HYB)
        downX32  = downSam( ('CX%s'%(j+2)),ListX[int(j)],C,outDim,  ratio)
        E   = downX32-Z
        G   = UpSam( ('E%s'%(j+2)),E, C, Y, downY4, downY16, iniUp3x3, outDim, ratio, priorNum)
        G   = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
        HY  = HY-G
#        HY  = resCNNnet_addF(('Pri%s'%(j+2)),HY,Y/10,j+2,upRank,subnetL)
        HY  = resCNNnet(('Pri%s'%(j+2)),HY,j+2,upRank, subnetL)
    
    #最后一层
    HYB     = tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
    ListX.append(YA + HYB)
    outX    = resCNNnet('FinalAjust',ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    CX  = downSam( ('CX%s'%(HSInetL)),ListX[int(HSInetL-2)], C, outDim,  ratio)
    E  = CX-Z
    return outX, ListX, YA, E, HY, CX
#    return downX32, HY
    
    

# reCNNnet 不确定要用这个还是上面那个
def resCNNnet(name,X,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)                        
        return X    
    
# reCNNnet 不确定要用这个还是上面那个
def resCNNnet_addF(name,X,Y,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel_addF(('resCNN_%s_%s'%(j,i+1)), 3, X, Y, channel,3)                         
        return X     
                    
# 用pooling下采样  
def getCs(name, ratio):
    Clist = []
    with tf.variable_scope(name):
        filter1 = tf.get_variable(
              'Cfilter', 
              [6, 6, 1, 1],
              tf.float32, 
              initializer=tf.constant_initializer(1/36))
        Clist.append(filter1)
        if ratio >4:
            filter2 = tf.get_variable(
                  'Cfilter2', 
                  [6, 6, 1, 1],
                  tf.float32, 
                  initializer=tf.constant_initializer(1/36))     
            Clist.append(filter2)
            if ratio>16:
                  filter3 = tf.get_variable(
                  'Cfilter3', 
                  [3, 3, 1, 1],
                  tf.float32, 
                  initializer=tf.constant_initializer(1/9))     
                  Clist.append(filter3)
    return Clist 

   
    
def downSam(name, X, C, ChDim, ratio):
    k=-1
    with tf.variable_scope(name):
        k      = k+1
        padX   = mypadding(X)
        X      = tf.nn.depthwise_conv2d(padX, tf.tile(C,[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')
        downX = X[:,8+int(ratio/2)-1:-9:ratio,8+int(ratio/2)-1:-9:ratio,:]
        return downX
    
def mypadding(X, paddnum=8):
    
    sizeI   = X.get_shape().as_list()
    H   = sizeI[1]
    W   = sizeI[2]
    
    # Left
    temptemp           = X[:,0:paddnum,0:paddnum,:]
    tempL = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,:,0:paddnum,:]
    tempL = tf.concat([tempL,temptemp[:,:,::-1,:]],1)
    
    temptemp           = X[:,H-paddnum:H,0:paddnum,:]
    tempL = tf.concat([tempL,temptemp[:,::-1,::-1,:]],1)
    
    # Middle
    temptemp           = X[:,0:paddnum,:,:]
    tempM = temptemp[:,::-1,:,:]
    
    tempM = tf.concat([tempM,X],1)
    
    temptemp           = X[:,H-paddnum:H,:,:]
    tempM = tf.concat([tempM,temptemp[:,::-1,:,:]],1)
    

    # Right
    temptemp           = X[:,0:paddnum,W-paddnum:W,:]
    tempR = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,:,W-paddnum:W,:]
    tempR = tf.concat([tempR,temptemp[:,:,::-1,:]],1)
    
    temptemp           = X[:,H-paddnum:H,W-paddnum:W,:]
    tempR = tf.concat([tempR,temptemp[:,::-1,::-1,:]],1)
    
    # all
    padX = tf.concat([tempL, tempM, tempR],2) 
    
    return padX

      
def UpSam(name,X, C, Y, downY4, downY16, iniUp3x3, outDim, ratio, priorNum):
    with tf.variable_scope(name):               
#        X = UpsumLevel('Cfilter32',X, tf.tile(C,  [1,1,outDim,1]), ratio, outDim)# 2 倍上采样 
        
        sizeY = Y.get_shape().as_list()
        X = UpsumLevel2('Cfilter32',X, C, ratio, outDim, sizeY)# 新的反卷积方式更自由一些
        
        #下采样再上采样的调整
        _,_,X = downSamAdj('downAdj',X, iniUp3x3, outDim,ratio)      
        
        X     = UpSamAdj('upAdj',X, downY4, downY16, Y, iniUp3x3, outDim, ratio, sizeY[3], priorNum)
           
        return X  
 
def UpsumLevel2(name,X, filterC, ratio, outDim,sizeY):
    # deepwise 反卷积
    sizeX = X.get_shape().as_list()
    X = tf.transpose(X, [0,3,1,2])
    X = tf.reshape(X,[sizeX[0]*outDim, sizeX[1], sizeX[2], 1])
    outX = tf.nn.conv2d_transpose (X, filterC[::-1,::-1,:,:], [sizeX[0]*outDim, sizeY[1],sizeY[2],1], strides=[1,ratio,ratio,1], padding='SAME')
    outX = tf.reshape(outX, [sizeX[0],outDim, sizeY[1],sizeY[2]])
    outX = tf.transpose(outX, [0,2,3,1])
    return outX   
    
    
def UpsumLevel(name,X, filterC, ratio, outDim):
     # 2 倍上采样     
     # 插入0，这并不简单 OutX[:,int(ratio/2):-1:ratio,int(ratio/2):-1:ratio,:] = X
    sizeX   = X.get_shape().as_list()
    ones_like_X    = tf.ones_like(X, dtype=tf.int32)  
    batch_range = tf.reshape(tf.range(sizeX[0], dtype=tf.int32), shape=[sizeX[0], 1, 1, 1])
    Channel_range = tf.reshape(tf.range(sizeX[3], dtype=tf.int32), shape=[ 1, 1, 1, sizeX[3]])   
    H_range = tf.reshape(tf.range(sizeX[1], dtype=tf.int32), shape=[ 1, sizeX[1], 1, 1])*int(ratio)+int(ratio/2-1)
    W_range = tf.reshape(tf.range(sizeX[2], dtype=tf.int32), shape=[ 1, 1, sizeX[2], 1])*int(ratio)+int(ratio/2-1)  
#    print(ones_like_X)
    x1 = ones_like_X*batch_range # * 可以直接实现bsxfun,这倒是很方便
    x4 = ones_like_X*Channel_range
    x2 = ones_like_X*H_range
    x3 = ones_like_X*W_range
    oral_size = tf.size(X)
    indices = tf.transpose(tf.reshape(tf.stack([x1, x2, x3, x4]), [4, oral_size]))
    values = tf.reshape(X, [oral_size])
    
    outsize = (sizeX[0], sizeX[1]* ratio, sizeX[2]* ratio, sizeX[3])
    outX   = tf.scatter_nd(indices, values, outsize)

    outX = tf.nn.depthwise_conv2d(outX,filterC[::-1,::-1,:,:], strides=[1,1,1,1], padding='SAME')
#    
    filter1 = tf.get_variable(
      name+'Blur', [4, 4, outDim, outDim], tf.float32, initializer=tf.constant_initializer(getBlurKernel(4, outDim)))
    outX = tf.nn.conv2d(outX,filter1,strides=[1,1,1,1],padding='SAME')   

    return outX 



def downSamAdj(name, X, iniUp3x3, ChDim, ratio):
    
#    with tf.variable_scope(name):
        k      = -1
        k      = k+1
        iniC1 = np.stack((iniUp3x3, np.zeros([3,3,ChDim,ChDim])),3)
        iniC2 = np.stack(( np.zeros([3,3,ChDim,ChDim]), iniUp3x3),3)
        iniC  = np.stack(( iniC1, iniC2),2)
        
        filter1 = tf.get_variable(
          name, 
          [3, 3, ChDim, 2*ChDim],
          tf.float32, 
          initializer=tf.constant_initializer(np.stack((iniUp3x3,iniUp3x3),3)/4))
        X      = tf.nn.conv2d(X, filter1, strides=[1,1,1,1],padding='SAME')
        filter12 = tf.get_variable(
          name+'0', 
          [3, 3, 2*ChDim, 2*ChDim],
          tf.float32, 
          initializer=tf.constant_initializer(iniC/4))
        X       = tf.nn.conv2d(X, filter12, strides=[1,1,1,1],padding='SAME')   
        
        
        downX4 = X[:,1:-1:4,1:-1:4,:]
        if ratio ==4:
            downX16 = []
            downX32 = downX4
        else: 
            k       = k+1

            
            filter2 = tf.get_variable(
              name+'1', 
              [3, 3, 2*ChDim, 2*ChDim],
              tf.float32, 
              initializer=tf.constant_initializer(iniC/4))
            X       = tf.nn.conv2d(downX4, filter2, strides=[1,1,1,1],padding='SAME')    
            X       = X[:,0:-1:2,0:-1:2,:]   
            if ratio==8:
                downX32 = X
                downX16 = []
            else:
                filter3 = tf.get_variable(
                  name+'2', 
                  [3, 3, 2*ChDim, 2*ChDim],
                  tf.float32, 
        #          initializer=tf.constant_initializer(0.0))
                  initializer=tf.constant_initializer(iniC/4))
                X       = tf.nn.conv2d(X, filter3, strides=[1,1,1,1],padding='SAME')             
                downX16 = X[:,0:-1:2,0:-1:2,:]   
                if ratio==16:
                    downX32 = downX16
                else:
                    k  = k+1
                    filter4 = tf.get_variable(
                      name+'3', 
                      [3, 3, 2*ChDim, 2*ChDim],
                      tf.float32, 
            #          initializer=tf.constant_initializer(0.0))
                      initializer=tf.constant_initializer(iniC/4))
                    X       = tf.nn.conv2d(downX16, filter4, strides=[1,1,1,1],padding='SAME')    
                    downX32 = X[:,0:-1:2,0:-1:2,:]      

        return downX4,  downX16,  downX32
          
  
def UpSamAdj(name,X, downY4, downY16, Y, iniUp3x3, outDim, ratio, YframNum, priorNum):
#    with tf.variable_scope(name):   
        iniC1 = np.stack((iniUp3x3, np.zeros([3,3,outDim,outDim])),3)
        iniC2 = np.stack(( np.zeros([3,3,outDim,outDim]), iniUp3x3),3)
        iniC  = np.stack(( iniC1, iniC2),2)
            
        if ratio==32:
            X = UpsumLevelAdj('Cfilter1',X,iniC, outDim*2, outDim*2)# 2 倍上采样               
            X = resLevel_addF('Ajust1', 3, X, downY16/10, outDim*2,priorNum)# 两层调整     

        if ratio>=16:
            X = UpsumLevelAdj('Cfilter2',X,iniC, outDim*2, outDim*2)# 2 倍上采样  
#            X = resLevel_addF('Ajust2', 3, X, downY4/10, outDim*2,6)# 两层调整
        if ratio>=8:
            X = UpsumLevelAdj('Cfilter3',X,iniC, outDim*2, outDim*2)# 2 倍上采样   
            X = resLevel_addF('Ajust2', 3, X, downY4/10, outDim*2,priorNum)# 两层调整
        
               
        X = UpsumLevelAdj('Cfilter4',X,np.stack((iniUp3x3, iniUp3x3),2), outDim*2, outDim)# 2 倍上采样  
        X = UpsumLevelAdj('Cfilter5',X,iniUp3x3, outDim, outDim)# 2 倍上采样
        X = resLevel_addF('Ajust3', 3, X, Y/10, outDim,Y.get_shape().as_list()[3])# 两层调整
        filter1 = tf.get_variable(
          'Blur', [4, 4, outDim, 1], tf.float32, initializer=tf.constant_initializer(1/16))
        X = tf.nn.depthwise_conv2d(X,filter1,strides=[1,1,1,1],padding='SAME')        

        return X  


def UpsumLevelAdj(name,X,iniUp2x2, inDim, outDim):
     # 2 倍上采样        
    filter1 = tf.get_variable(
          name, 
          [3, 3, outDim, inDim],
          tf.float32, 
#          initializer=tf.constant_initializer(0.0))
          initializer=tf.constant_initializer(iniUp2x2/4))
    
#    sizeX   = tf.shape(X)*[1,2,2,1]
    sizeX   = X.get_shape().as_list()
    
    X = tf.nn.conv2d_transpose (X,filter1, [sizeX[0],sizeX[1]*2,sizeX[2]*2,outDim], strides=[1,2,2,1], padding='SAME')
    return X

def getBlurKernel(Fsize, outDim):
    initemp = np.tile(np.eye(outDim),[Fsize,Fsize,1,1])
    Upfilter = np.ones([Fsize, Fsize])/(Fsize*Fsize)
    filt = np.expand_dims(np.expand_dims(Upfilter,2),3)
    BlurKernel = initemp*filt
    return BlurKernel       

def resLevel(name, Fsize,X, Channel):
    with tf.variable_scope(name):
        # 两层调整
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
        scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale1'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 加到新的test里
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases3')
        scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale3'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)
        
        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 加到新的test里
        
        
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel], dtype=tf.float32), trainable=True, name='biases2')
        scale = tf.Variable(tf.ones([Channel])/20, trainable=True, name=('scale2'))
        beta = tf.Variable(tf.zeros([Channel]), trainable=True, name=('beta2'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        X = tf.add(X, feature_normal)  #  shortcut  
        return X
    
def resLevel_addF(name, Fsize, X, Y,ChannelX,ChannelY):
    with tf.variable_scope(name):
        # 两层调整
        Channel = ChannelX+ChannelY
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
        scale = tf.Variable(tf.ones([Channel+3])/100, trainable=True, name=('scale1'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

        conv = tf.nn.conv2d(tf.concat([X,Y],3), kernel, [1, 1, 1, 1], padding='SAME')
        
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 我又加了一层
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[ Channel+3], dtype=tf.float32), trainable=True, name='biases2')
        scale = tf.Variable(tf.ones([ Channel+3])/100, trainable=True, name=('scale2'))
        beta = tf.Variable(tf.zeros([ Channel+3]), trainable=True, name=('beta2'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)
        #
        
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, ChannelX])
        biases = tf.Variable(tf.constant(0.0, shape=[ChannelX], dtype=tf.float32), trainable=True, name='biases3')
        scale = tf.Variable(tf.ones([ChannelX])/100, trainable=True, name=('scale3'))
        beta = tf.Variable(tf.zeros([ChannelX]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

#        feature_normal = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_normal)  #  shortcut  
        return X    
    
def ConLevel(name, Fsize, X, inC, outC):
    with tf.variable_scope(name):
        kernel = create_kernel(name=('weights'), shape=[Fsize, Fsize, inC, outC])
        biases = tf.Variable(tf.constant(0.0, shape=[outC], dtype=tf.float32), trainable=True, name=('biases'))
        scale = tf.Variable(tf.ones([outC]), trainable=True, name=('scale'))
        beta = tf.Variable(tf.zeros([outC]), trainable=True, name=('beta'))

        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        X = tf.nn.bias_add(conv, biases)

        mean, var = tf.nn.moments(X,[0, 1, 2])
        X = tf.nn.batch_normalization(X,  mean, var, beta, scale, 1e-5)

        X = tf.nn.relu(X)
        return X
  
    
## A 1X1卷积   
#def MyconvA( name, x, in_filters, out_filters, strides, iniA):
#    with tf.variable_scope(name):
#           
#      
#      kernel = tf.get_variable(
#              'A', 
#              [1, 1, in_filters, out_filters],
#              tf.float32, 
##              initializer=tf.constant_initializer(1))
#              initializer=tf.constant_initializer(iniA))
#      # 计算卷积
#      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
#  
#    
## B 1X1卷积   
#def MyconvB( name, x, in_filters, out_filters, strides):
#    with tf.variable_scope(name):
#      # 获取或新建卷积核，正态随机初始化
#      kernel = tf.get_variable(
#              'B', 
#              [1, 1, in_filters, out_filters],
#              tf.float32, 
#              initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
#      # 计算卷积
#      return tf.nn.conv2d(x, kernel, strides, padding='SAME')    

def create_kernel(name, shape, initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.1)):
#def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables