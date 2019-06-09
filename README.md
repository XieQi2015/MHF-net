# KBR-Denoising
Outline:
    Folder structure
    Usage
    Citation


Folder structure:

    data\           : example multispectral images (MSI) data 
        |-- testMSI_1.mat      : example MSI data 1
        |-- testMSI_2.mat      : example MSI data 2
        |-- testMSI_3.mat      : example MSI data 3
        |-- testMSI_4.mat      : example MSI data 4
    lib\            : library of code
        |-- KBRreg\            : helper functions of the ITSReg method
        |-- tensor_toolbox\    : toolbox for tensor operations [1]
        |-- my_tensor_toolbox\ : another toolbox for tensor operations 
        |-- quality_assess\    : functions of quality assessment indices
        |       |-- ssim_index.m            : SSIM [2]
        |       |-- FeatureSIM.m            : FSIM [3]
        |       |-- ErrRelGlobAdimSyn.m     : ERGAS
        |       |-- SpectAngMapper.m        : SAM
        |       |-- MSIQA.m                 : interface for calculating PSNR and the four indices above
        |-- compete_methods : competing methods (down loaded or implemented based on reference papers)
        |       |-- ksvdbox\                : K-SVD toolbox [4]
        |       |-- ompbox\                 : dependency of toolbox ksvdbox [5]
        |       |-- naonlm3d\               : 3D NLM toolbox [6]
        |       |-- tensor_SVD\             : tSVD toolbox [10]
        |       |-- BM3D\                   : BM3D toolbox [7]
        |       |-- BM4D\                   : BM4D toolbox [8]
        |       |-- tensor_dl\              : TDL toolbox [9]
        |       |-- ui_utils\               : scripts used in GUI
        |       |-- NLM3D.m                 : 3D NLM using toolbox naonlm3d 
        |       |-- LRTA.m                  : LRTA method [9]
        |       |-- PARAFAC.m               : PARAFAC method [9]
        |       |-- LRTVdenoising.m         : Trace-TV method 
        |       |-- tSVD_Denoising.m        : t-SVD method[10] 
        |-- togetGif.m         : function for getting GIF result
        |-- showMSIResult.m    : function for showing MSI result
    Demo.m          : scripts that applies the methods and calculates the QA indices
    KBR_DeNoising.m : core function of the KBRreg-based MSI denoising method 
    KBRreg.m        : function for sloving the intrinsic tensor sparsity regular model 


Usage:

    For MSI with Gaussian noise, you can simply follow these steps:
        1.Re-arrange the MSI into [0, 1].
        2.Add the folder 'lib'into path, and use the function KBR_DeNoising as follows:
            [ clean_img ] = KBR_DeNoising( noisy_img, noise_variance )
    Please type 'help KBR_DeNoising ' to get more information.

    You may find example codes in file Demo.m

    Also, you can use the demo to see some comparison. You can:
      1. Type 'Demo' to to run various methods and see the pre-computed results.
      2. Use 'help Demo' for more information.
      3. Change test MSI by simply modifying variable 'filename' in Demo.m (NOTE: make sure your MSI
         meets the format requirements).
      4. Change noise level by modifying variables  'sigma_ratio' in Demo.m
      5. Select competing methods by turn on/off the enable-bits in Demo.m


Citation:

    Qi Xie, Minghao Zhou, Qian Zhao, Deyu Meng*, Wangmeng Zuo & Zongben Xu
    Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net[C]
    2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2019.

    BibTeX:
      @article{Qi2017Kronecker,  
      title={Kronecker-Basis-Representation Based Tensor Sparsity and Its Applications to Tensor Recovery},  
      author={Qi, Xie and Qian, Zhao and Meng, Deyu and Xu, Zongben},  
      journal={IEEE Transactions on Pattern Analysis & Machine Intelligence},  
      volume={PP},  
      number={99},  
      pages={1-1},  
      year={2017},
      }

