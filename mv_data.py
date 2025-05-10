import os
import shutil

data_path = '../supplement/'
target_path = '../data1/'

center = os.listdir(data_path)
for i in range(0, len(center)):
    patient_name = os.listdir(data_path+center[i]+'/')
    for j in range(0, len(patient_name)):
        if not os.path.exists(target_path+center[i]+'/'+patient_name[j]+'/'):
            os.makedirs(target_path+center[i]+'/'+patient_name[j]+'/')
        shutil.copy(data_path+center[i]+'/'+patient_name[j]+'/PET.nrrd', target_path+center[i]+'/'+patient_name[j]+'/PET.nrrd')
        shutil.copy(data_path+center[i]+'/'+patient_name[j]+'/CT.nrrd', target_path+center[i]+'/'+patient_name[j]+'/CT.nrrd')
        shutil.copy(data_path+center[i]+'/'+patient_name[j]+'/Segmentation.nrrd', target_path+center[i]+'/'+patient_name[j]+'/Segmentation.nrrd')