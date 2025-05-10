import nibabel as nib
import skimage
import os


# CT to PET resample
root_path = '../data/'
center = os.listdir(root_path)
for c in range(0,len(center)):
    patient_name = os.listdir(root_path+center[c]+'/')
    for i in range(0,len(patient_name)):
        pet_image = nib.load(root_path+center[c]+'/'+patient_name[i]+'/PET.nii.gz')
        pet_image_affine = pet_image.affine
        pet_width, pet_height, pet_channel = pet_image.dataobj.shape
        new_space = [pet_width, pet_height, pet_channel]

        ct_image = nib.load(root_path+center[c]+'/'+patient_name[i]+'/CT.nii.gz')
        ct_image_affine = ct_image.affine
        ct_data = ct_image.get_fdata()

        ct_resampled = skimage.transform.resize(ct_data, new_space, order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        nib.Nifti1Image(ct_resampled,pet_image_affine).to_filename(root_path+center[c]+'/'+patient_name[i]+'/resampled_CT.nii.gz')

        if os.path.exists(root_path+center[c]+'/'+patient_name[i]+'/resampled_PET.nii.gz'):
            os.remove(root_path+center[c]+'/'+patient_name[i]+'/resampled_PET.nii.gz')
        if os.path.exists(root_path+center[c]+'/'+patient_name[i]+'/resampled_Segmentation.nii.gz'):
            os.remove(root_path+center[c]+'/'+patient_name[i]+'/resampled_Segmentation.nii.gz')
        print(root_path+center[c]+'/'+patient_name[i]+'/resampled_CT.nii.gz')
'''
# PET and label to CT resample
root_path = '../data/'
center = os.listdir(root_path)
for c in range(0,len(center)):
    patient_name = os.listdir(root_path+center[c]+'/')
    for i in range(0,len(patient_name)):
        ct_image = nib.load(root_path+center[c]+'/'+patient_name[i]+'/CT.nii.gz')
        #print(ct_image)
        ct_image_affine = ct_image.affine
        ct_width, ct_height, ct_channel = ct_image.dataobj.shape
        new_space = [ct_width, ct_height, ct_channel]

        pet_image = nib.load(root_path+center[c]+'/'+patient_name[i]+'/PET.nii.gz')
        pet_image_affine = pet_image.affine
        pet_data = pet_image.get_fdata()
        
        pet_resampled = skimage.transform.resize(pet_data, new_space, order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        nib.Nifti1Image(pet_resampled,ct_image_affine).to_filename(root_path+center[c]+'/'+patient_name[i]+'/resampled_PET.nii.gz')
        print(root_path+center[c]+'/'+patient_name[i]+'/resampled_PET.nii.gz')

        pet_label = nib.load(root_path+center[c]+'/'+patient_name[i]+'/Segmentation.nii.gz')
        pet_label_affine = pet_label.affine
        pet_label_data = pet_label.get_fdata()
        pet_label_resampled = skimage.transform.resize(pet_label_data, new_space, order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        pet_label_resampled[pet_label_resampled<0] = 0
        pet_label_resampled[pet_label_resampled>0] = 1
        nib.Nifti1Image(pet_label_resampled,ct_image_affine).to_filename(root_path+center[c]+'/'+patient_name[i]+'/resampled_Segmentation.nii.gz')
        print(root_path+center[c]+'/'+patient_name[i]+'/resampled_Segmentation.nii.gz')
'''