
# coding: utf-8

# # Demo for running IFCNN to fuse multiple types of images

# Project page of IFCNN is https://github.com/uzeful/IFCNN.
# 
# If you find this code is useful for your research, please consider to cite our paper.
# ```
# @article{zhang2019IFCNN,
#   title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
#   author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
#   journal={Information Fusion},
#   volume={54},
#   pages={99--118},
#   year={2020},
#   publisher={Elsevier}
# }
# ```
# 
# Detailed procedures to use IFCNN are introduced as follows.

# ## 1. Load required libraries

# In[1]:


import os
import cv2
import time
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

from utils.myTransforms import denorm, norms, detransformcv2
import SimpleITK as sitk
from utils.myDatasets import ImagePair

def source_imgs_generate(patient_path):
    PET = []
    CT = []
    
    PET_sitk = sitk.ReadImage(patient_path+'/PET.nii.gz')
    image_PET = sitk.GetArrayFromImage(PET_sitk)
    
    CT_sitk = sitk.ReadImage(patient_path+'/CT.nii.gz')
    image_CT = sitk.GetArrayFromImage(CT_sitk)
    
    for z in range(0, image_PET.shape[0]):
        image_z_PET = image_PET[z,:,:]
        image_z_PET = np.abs(image_z_PET)
        image_z_PET = np.uint8((image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))*255)
        image_PET_z = cv2.cvtColor(image_z_PET, cv2.COLOR_GRAY2RGB)
        image_PET_z = image_PET_z / 255.0
        PET.append(image_PET_z)
        
        image_z_CT = image_CT[z,:,:]
        image_z_CT = np.uint8((image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))*255)
        image_CT_z = cv2.cvtColor(image_z_CT, cv2.COLOR_GRAY2RGB)
        image_CT_z = image_CT_z / 255.0
        CT.append(image_CT_z)
    
    PET = np.array(PET, dtype='float32')
    CT = np.array(CT, dtype='float32')
    
    return PET, CT


# ## 2. Load the well-trained image fusion model (IFCNN-MAX)

# In[2]:


# we use fuse_scheme to choose the corresponding model, 
# choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
for i in range(0,3):
    fuse_scheme = i
    if fuse_scheme == 0:
        model_name = 'IFCNN-MAX'
    elif fuse_scheme == 1:
        model_name = 'IFCNN-SUM'
    elif fuse_scheme == 2:
        model_name = 'IFCNN-MEAN'
    else:
        model_name = 'IFCNN-MAX'
    # load pretrained model
    model = myIFCNN(fuse_scheme=fuse_scheme)
    model.load_state_dict(torch.load('snapshots/'+ model_name + '.pth'))
    model.eval()
    model = model.cuda()


    # ## 3. Use IFCNN to respectively fuse CMF, IV and MD datasets
    # Fusion images are saved in the 'results' folder under your current folder.

    # In[3]:

    center = ['gulou', 'huaxi', 'shengrenmin']
    savepath = '../../../compare_fusion_result/'
    mean=[0, 0, 0]         # normalization parameters
    std=[1, 1, 1]
    for c in range(0, len(center)):
        patient_name = os.listdir('E:/lvpao/fusion_result2/'+center[c]+'/')
        for p in range(0, len(patient_name)):
            patient_path = 'E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p]
            PET, CT = source_imgs_generate(patient_path)
            print(PET.shape)
            FUSION_IMG = np.zeros((PET.shape[0],180,180),dtype='float32')
            for i in range(0, PET.shape[0]):
                PET_img = PET[i,:,:,:]
                CT_img = CT[i,:,:,:]
                # load source images
                pair_loader = ImagePair(impath1=PET_img, impath2=CT_img, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)
                                        ]))
                PET_img, CT_img = pair_loader.get_pair()
                PET_img.unsqueeze_(0)
                CT_img.unsqueeze_(0)
                # perform image fusion
                with torch.no_grad():
                    res = model(Variable(PET_img.cuda()), Variable(CT_img.cuda()))
                    res = denorm(mean, std, res[0]).clamp(0, 1) * 255
                    res_img = res.cpu().data.numpy().astype('uint8')
                    img_ = res_img.transpose([1,2,0])
                    img = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
                    FUSION_IMG[i,:,:] = img
                
                if not os.path.exists(savepath+center[c]+'/'+patient_name[p]):
                    os.makedirs(savepath+center[c]+'/'+patient_name[p])
                
                FUSION_z_sitk = sitk.GetImageFromArray(FUSION_IMG)
                sitk.WriteImage(FUSION_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/'+model_name+'_Fusion.nii.gz')
                print('E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p])