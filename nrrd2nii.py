import os
import SimpleITK as sitk

# CT to PET resample
patient_name_path = '../data2/'
save_path = '../data/'
center = os.listdir(patient_name_path)
for i in range(0,len(center)):
    patient_name = os.listdir(patient_name_path+center[i]+'/')
    for j in range(0,len(patient_name)):
        if not os.path.exists(save_path+center[i]+'/'+patient_name[j]+'/'):
            os.makedirs(save_path+center[i]+'/'+patient_name[j]+'/')
        image = sitk.ReadImage(patient_name_path+center[i]+'/'+patient_name[j]+'/'+'CT.nrrd')
        sitk.WriteImage(image, save_path+center[i]+'/'+patient_name[j]+'/'+'CT.nii.gz')

        image = sitk.ReadImage(patient_name_path+center[i]+'/'+patient_name[j]+'/'+'PET.nrrd')
        sitk.WriteImage(image, save_path+center[i]+'/'+patient_name[j]+'/'+'PET.nii.gz')

        image = sitk.ReadImage(patient_name_path+center[i]+'/'+patient_name[j]+'/'+'Segmentation.nrrd')
        sitk.WriteImage(image, save_path+center[i]+'/'+patient_name[j]+'/'+'Segmentation.nii.gz')