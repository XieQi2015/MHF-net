# Consistent MS/HS Fusion Net 
For multispectral and hyperspectral image fusion in the case that spectral and spatial responses of training and testing data are consistent

The code has been test on Windows 10 with Tensorflow 1.12.0 and environment of Spyder

Outline:

    Folder structure
    Usage
    Citation
    
Folder structure:

    rowData\    : Original multispectral images (MSI) data 
        |-- CAVEdata\      : CAVE data set
        |       |-- complete_ms_data\       : The images of data
        |       |-- response coefficient.mat: A matrix of exploited response coefficient
    temp\       : Trained result
        |-- TrainedNet\    : A example of trained parameters
    CAVE_dataReader.py     : The data reading and preparing code
    CAVEmain.py            : Main code for training and testing 
    MHFnet.py              : Code of MHF-net 
    MyLib.py               : Some used code
    TestSample.mat         : A example testing data

Usage:

To run testing with the example data "TestSample.mat ", you can just run CAVEmain.py while setting FLAGS.mode in line 23 as 'test'
      
To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/, and put the data in the folder ./ rowData/CAVEdata/complete_ms_data/ just like:

![We should have a image here](https://github.com/XieQi2015/ImageFolder/raw/master/MHFnet/example.png)

Then, you can just run CAVEmain.py while setting FLAGS.mode in line 23 as 'train'. There will be 20 samples randomly selected to be training samples, and the remain 12 samples will be used as testing samples.
You can also run CAVEmain.py while setting FLAGS.mode in line 23 as 'testAll' to test all the 12 testing samples

!!! If you want to use another spectral response function or data, you should reproduce the parameter 'iniA' with the method in scetion 4 of the supplementary material of our paper, this is important to achieve good performance !!!

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
