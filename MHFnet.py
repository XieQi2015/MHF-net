# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:05:10 2018
The MHF-net
@author: XieQi
"""
import tensorflow as tf
     
# main MHF-net net
def HSInet(Y,Z, iniUp3x3,iniA,upRank,outDim,HSInetL,subnetL,ratio=32):
    

    B = tf.get_variable(
              'B', 
              [1, 1, upRank, outDim],
              tf.float32, 
              initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
    tranB = tf.transpose(B,perm = [0,1,3,2])
    CListX = getCs('TheCsX', ratio)#inital the kernel for downsampling
    downY4, downY16, _ = downSam('GetPrior',Y, CListX, 3,ratio)# getPrior for upsample
    
    # fist stage
    YA = MyconvA( 'YA1', Y, 3, outDim, [1,1,1,1], iniA) #caculating YA
    _, _, downX32 = downSam('CX1',YA, CListX, outDim, ratio)  # downsampling 
    E  = downX32-Z   # Z上的残差
    G  = UpSam('E1',E, downY4, downY16, Y, iniUp3x3, outDim, ratio) # unsampling E
    G  = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
    HY = -G #
    HY  = resCNNnet(('Pri%s'%(1)),HY,1,upRank, subnetL) 
    ListX = []
    
    # 2nd to the 19th stage
    for j in range(HSInetL-2):
        HYB= tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
        ListX.append(YA + HYB)
        _, _, downX32  = downSam( ('CX%s'%(j+2)),ListX[int(j)],CListX,outDim,  ratio)
        E   = downX32-Z
        G   = UpSam( ('E%s'%(j+2)),E, downY4, downY16, Y, iniUp3x3, outDim, ratio)
        G   = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
        HY  = HY-G
        HY  = resCNNnet(('Pri%s'%(j+2)),HY,j+2,upRank, subnetL)
    
    #the final stage
    HYB     = tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
    ListX.append(YA + HYB)
    outX    = resCNNnet('FinalAjust',ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    _,_,CX  = downSam( ('CX%s'%(HSInetL)),ListX[int(HSInetL-2)],CListX, outDim,  ratio)
    E  = CX-Z
    return outX, ListX, YA, E, HY
    
# reCNNnet 
def resCNNnet(name,X,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)                        
        return X    
                        
# get the downsampling kernels  
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
    
def downSam(name, X, Clist, ChDim, ratio):
    k=-1
    with tf.variable_scope(name):
        k      = k+1
        X      = tf.nn.depthwise_conv2d(X, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')
        downX4 = X[:,1:-1:4,1:-1:4,:]
        if ratio ==4:
            downX16 = []
            downX32 = downX4
        else: 
            k       = k+1
            X       = tf.nn.depthwise_conv2d(downX4, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')                
            downX16 = X[:,1:-1:4,1:-1:4,:]   
            if ratio==16:
                downX32 = downX16
            else:
                k  = k+1
                X       = tf.nn.depthwise_conv2d(downX16, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')         
                downX32 = X[:,0:-1:2,0:-1:2,:]      

        return downX4,  downX16,  downX32
          
  
def UpSam(name,X, downY4, downY16, Y, iniUp3x3, outDim, ratio):
    with tf.variable_scope(name):               
        if ratio==32:
            X = UpsumLevel2('Cfilter1',X,iniUp3x3, outDim)# 2 timse upsampling
            X = resLevel_addF('Ajust1', 3, X, downY16/10, outDim,3)# adjusting after upsampling     

        if ratio>=16:
            X = UpsumLevel2('Cfilter2',X,iniUp3x3, outDim)# 
            X = UpsumLevel2('Cfilter3',X,iniUp3x3, outDim)#    
            X = resLevel_addF('Ajust2', 3, X, downY4/10, outDim,3)#
                       
        X = UpsumLevel2('Cfilter4',X,iniUp3x3, outDim)# 
        X = UpsumLevel2('Cfilter5',X,iniUp3x3, outDim)# 
        X = resLevel_addF('Ajust3', 3, X, Y/10, outDim,3)# 
        filter1 = tf.get_variable(
          'Blur', [4, 4, outDim, 1], tf.float32, initializer=tf.constant_initializer(1/16))
        X = tf.nn.depthwise_conv2d(X,filter1,strides=[1,1,1,1],padding='SAME')        

        return X  
        

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

        feature_relu = tf.nn.relu(feature_normal)        


        X = tf.add(X, feature_relu)  #  shortcut  
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

        feature_relu = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_relu)  #  shortcut  
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
 
def UpsumLevel2(name,X,iniUp2x2, outDim):
     # 2 倍上采样        
    filter1 = tf.get_variable(
          name, 
          [3, 3, outDim, outDim],
          tf.float32, 
#          initializer=tf.constant_initializer(0.0))
          initializer=tf.constant_initializer(iniUp2x2/4))
    
    sizeX   = tf.shape(X)*[1,2,2,1]
    X = tf.nn.conv2d_transpose (X,filter1, sizeX, strides=[1,2,2,1], padding='SAME')
    return X
        
# A 1X1 convolution for caculating Y*A
def MyconvA( name, x, in_filters, out_filters, strides, iniA):
    with tf.variable_scope(name):
         
      kernel = tf.get_variable(
              'A', 
              [1, 1, in_filters, out_filters],
              tf.float32, 
#              initializer=tf.constant_initializer(1))
              initializer=tf.constant_initializer(iniA))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
  
    
# B 1X1 convolution for caculating Y_hat*B  
def MyconvB( name, x, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      kernel = tf.get_variable(
              'B', 
              [1, 1, in_filters, out_filters],
              tf.float32, 
              initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')    

def create_kernel(name, shape, initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.1)):
#def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables