# Blind MS/HS Fusion Net 
For multispectral and hyperspectral image fusion in mismatch cases of spectral and spatial responses in training and testing
data, and even across different sensors, which is generally considered to be a challenging issue for general supervised MS/HS fusion
methods.

The code has been test on Windows 10 with Tensorflow 1.12.0 and environment of Spyder

Outline:

    Folder structure
    Usage
    Citation
    
Folder structure:

    rowData\    : Original multispectral images (MSI) data 
        |-- CAVEdata\      : CAVE data set
        |       |-- complete_ms_data\    : The images of data (To be downloaded)
        |       |-- AllR.mat             : Bases of spectral response coefficients
        |       |-- AllC.mat             : Bases of spatial response coefficients
        |       |-- randMatrix.mat       : A list of random combination coefficients of AllR and AllC for testing
        |-- CASI_Houston\  : CASI_Houston data set
        |       |-- 2013_IEEE_GRSS_DF_Contest_CASI.tif   : The images of data (To be downloaded)
        |       |-- GetTrainDFTCData.m                   : Matlab code for prepareing the train and test data
        |       |-- AllRC.mat                            : A list of random R and C for testing
        |-- ROSIS_Pavia\   : ROSIS_Pavia data set
        |       |-- original_rosis.tif                   : The images of data (Downloaded)
        |       |-- GetPaviaData.m                       : Matlab code for prepareing the test data
        |       |-- R.mat                                : A matrix of real spectral response coefficient
    RealData\   : Prepared ROSIS_Pavia data for testing 
    temp\       : Trained result (To be downloaded)
        |-- CAVE_Exam\     : Trained result for CAVE data set
        |-- RealDataExam\  : Trained result for RealData    
    CAVE_dataReader.py     : The data reading and preparing code
    CAVEmain.py            : Main code for training and testing 
    MHFnet.py              : Code of MHF-net 
    MyLib.py               : Some used code
    TestSample.mat         : A example testing data

## Usage:

Highly recommended to run CMHF-net first, many usage detail of BMHF-net is similar to CMHF-net.

### Train and Test on CAVE Data Set

To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/, and put the data in the folder ./ rowData/CAVEdata/complete_ms_data/, one can refer to the readme of CMHF-net for more detail.

Then, you can just run CAVEmain.py while setting FLAGS.mode in line 23 as 'train'. There will be 20 samples randomly selected to be training samples, and the remain 12 samples will be used as testing samples.
You can also run CAVEmain.py while setting FLAGS.mode in line 23 as 'test' to test all the 12 testing samples. The trained network parameters need to be download form
https://pan.baidu.com/s/1_qo1a_uF8LzRRqLWg27rLg, Extraction code (提取码): m583.

We randomly generate spectral response R by random combination of real spectral responses obtained from 28 cameras, they are:
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/R.gif)
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/Y.gif)

### Train and Test on CASI_Houston Data Set

To train and test on CASI_Houston data set, you must first download the CAVE data set form https://hyperspectral.ee.uh.edu/?page%20id=459#download, and put 2013_IEEE_GRSS_DF_Contest_CASI.tif into folder ./rowData/CASI_Houston/. Then run the GetTrainDFTCData.m to prepare the train and test data.

Here we set the training data and testing data as:
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/Data.png)
We randomly generate spectral response R and spatial response C for each training sample. For each R we set the center wavelength of the 4 bands in HrMS image as random
numbers in [478; 487], [543; 547], [650; 660] and [816; 883], respectively, while the correlated effective bandwidth is set as random numbers in [73; 115], [80; 154], [70; 120] and [136; 203]. Note that most of the center wavelengths and effective bandwidths of the spectral responses of commonly
used sensors are in this specified range. Examples are:
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/R2.gif)
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/C.gif)

### Test on ROSIS_Pavia Data Set

The testing data ROSIS_Pavia download form http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral, one can run GetPaviaData.m  to prepare the testing data. The trained network parameters need to be download form
https://pan.baidu.com/s/1_qo1a_uF8LzRRqLWg27rLg, Extraction code (提取码): m583.

An illustration of the testing data and response are:

![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/R_real.gif)
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/Real.png)

In the later 2 experimetnts, 
![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/Introduction.png)

Citation:

    Qi Xie, Minghao Zhou, Qian Zhao, Zongben Xu and Deyu Meng* 
    MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion
    IEEE transactions on pattern analysis and machine intelligence, 2020.

    BibTeX:
    
    @article{xie2020MHFnet,
      title={MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion},
      author={Xie, Qi and Minghao, Zhou and Zhao, Qian and Xu, Zongben and Meng, Deyu},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2020},
      publisher={IEEE}
    }

If you encounter any problems, feel free to contact xq.liwu@stu.xjtu.edu.cn
