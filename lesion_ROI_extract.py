import SimpleITK as sitk
import numpy as np
from skimage import transform
import os

# CT to PET resample
root_path = '../data/'
save_path = '../slice/'
center = os.listdir(root_path)
for c in range(0,len(center)):
    patient_name = os.listdir(root_path+center[c]+'/')
    for i in range(0,len(patient_name)):
        PET_sitk = sitk.ReadImage(root_path+center[c]+'/'+patient_name[i]+'/PET.nii.gz')
        image_PET = sitk.GetArrayFromImage(PET_sitk)

        CT_sitk = sitk.ReadImage(root_path+center[c]+'/'+patient_name[i]+'/resampled_CT.nii.gz')
        image_CT = sitk.GetArrayFromImage(CT_sitk)

        GT_sitk = sitk.ReadImage(root_path+center[c]+'/'+patient_name[i]+'/Segmentation.nii.gz')
        image_GT = sitk.GetArrayFromImage(GT_sitk)

        z_index, _, _ = np.where(image_GT>0)
        z_index = np.unique(z_index)
        print(z_index)

        if not os.path.exists(save_path+center[c]+'/'+patient_name[i]+'/'):
            os.makedirs(save_path+center[c]+'/'+patient_name[i]+'/')

        for z in range(0, len(z_index)):
            image_z_PET = image_PET[z_index[z],:,:]
            image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            image_z_PET_sitk = sitk.GetImageFromArray(image_z_PET)
            sitk.WriteImage(image_z_PET_sitk, save_path+center[c]+'/'+patient_name[i]+'/PET_'+str(z_index[z])+'.nii.gz')

            image_z_CT = image_CT[z_index[z],:,:]
            image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            image_z_CT_sitk = sitk.GetImageFromArray(image_z_CT)
            sitk.WriteImage(image_z_CT_sitk, save_path+center[c]+'/'+patient_name[i]+'/CT_'+str(z_index[z])+'.nii.gz')

            image_z_GT = image_GT[z_index[z],:,:]
            image_z_GT = transform.resize(image_z_GT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            image_z_GT[image_z_GT>0] = 1
            image_z_GT_sitk = sitk.GetImageFromArray(image_z_GT)
            sitk.WriteImage(image_z_GT_sitk, save_path+center[c]+'/'+patient_name[i]+'/GT_'+str(z_index[z])+'.nii.gz')