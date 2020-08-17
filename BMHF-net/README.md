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
        |-- CASI_Houston\  : CASI_Houston data set
        |       |-- 2013_IEEE_GRSS_DF_Contest_CASI.tif   : The images of data (To be downloaded)
        |       |-- GetTrainDFTCData.m                   : Matlab code for prepareing the train and test data
        |-- ROSIS_Pavia\  : ROSIS_Pavia data set
        |       |-- original_rosis.tif                   : The images of data (Downloaded)
        |       |-- GetPaviaData.m                       : Matlab code for prepareing the test data
        |       |-- R.mat                                : A matrix of real spectral response coefficient
    RealData\   : Prepared ROSIS_Pavia data for testing 
    temp\       : Trained result
        |-- TrainedNet\    : A example of trained parameters
    CAVE_dataReader.py     : The data reading and preparing code
    CAVEmain.py            : Main code for training and testing 
    MHFnet.py              : Code of MHF-net 
    MyLib.py               : Some used code
    TestSample.mat         : A example testing data

Usage:
      
To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/, and put the data in the folder ./ rowData/CAVEdata/complete_ms_data/, one can refer to the readme of CMHF-net for more detail.

Then, you can just run CAVEmain.py while setting FLAGS.mode in line 23 as 'train'. There will be 20 samples randomly selected to be training samples, and the remain 12 samples will be used as testing samples.
You can also run CAVEmain.py while setting FLAGS.mode in line 23 as 'testAll' to test all the 12 testing samples


To train and test on CASI_Houston data set, you must first download the CAVE data set form https://hyperspectral.ee.uh.edu/?page%20id=459#download, and put 2013_IEEE_GRSS_DF_Contest_CASI.tif into folder ./ rowData/CASI_Houston/. Then run the GetTrainDFTCData.m to prepare the train and test data.

The testing data ROSIS_Pavia download form http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral, one can run GetPaviaData.m  to prepare the testing data.

Citation:

    Qi Xie, Minghao Zhou, Qian Zhao, Deyu Meng*, Wangmeng Zuo & Zongben Xu
    Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net[C]
    2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2019.

    BibTeX:
    
    @inproceedings{xie2019multispectral,
      title={Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net},
      author={Xie, Qi and Minghao, Zhou and Zhao, Qian and Meng, Deyu and Zuo, Wangmeng and Xu, Zongben },
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019} 
    }

If you encounter any problems, feel free to contact xq.liwu@stu.xjtu.edu.cn
