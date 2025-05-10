import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

center = ['gulou', 'huaxi', 'shengrenmin']
savepath = '../fusion_result/'
for c in range(0, len(center)):
    patient_name = os.listdir(savepath+center[c]+'/')
    for p in range(0, len(patient_name)):
        PET_sitk = sitk.ReadImage(savepath+center[c]+'/'+patient_name[p]+'/FUSION.nii.gz')
        image_PET = sitk.GetArrayFromImage(PET_sitk)

        GT_sitk = sitk.ReadImage(savepath+center[c]+'/'+patient_name[p]+'/GT.nii.gz')
        image_GT = sitk.GetArrayFromImage(GT_sitk)

        Original_PET_sitk = sitk.ReadImage('../data/'+center[c]+'/'+patient_name[p]+'/PET.nii.gz')
        Original_image_PET = sitk.GetArrayFromImage(Original_PET_sitk)

        Original_GT_sitk = sitk.ReadImage('../data/'+center[c]+'/'+patient_name[p]+'/Segmentation.nii.gz')
        Original_image_GT = sitk.GetArrayFromImage(Original_GT_sitk)
        z_index, _, _ = np.where(Original_image_GT>0)
        z_index = np.unique(z_index)
        PET = []
        GT = []
        for z in range(0, len(z_index)):
            image_z_PET = Original_image_PET[z_index[z],:,:]
            image_z_PET = np.abs(image_z_PET)
            PET.append(image_z_PET)
            GT.append(Original_image_GT[z_index[z],:,:])
        PET = np.array(PET, dtype='float32')
        GT = np.array(GT, dtype='float32')

        for i in range(0, image_PET.shape[0]):
            PET_img = image_PET[i,:,:]
            GT_img = image_GT[i,:,:]
            Original_PET = PET[i,:,:]
            Original_GT = GT[i,:,:]

            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.set_title("PET")
            ax1.imshow(PET_img, cmap=plt.cm.seismic)
            
            ax2 = fig.add_subplot(2,2,2)
            ax2.set_title("GT")
            ax2.imshow(GT_img, cmap=plt.cm.seismic)
            
            ax3 = fig.add_subplot(2,2,3)
            ax3.set_title("Original PET")
            ax3.imshow(Original_PET, cmap=plt.cm.seismic)
            
            ax4 = fig.add_subplot(2,2,4)
            ax4.set_title("Original GT")
            ax4.imshow(Original_GT, cmap=plt.cm.seismic)
            
            fig.tight_layout()
            plt.show()

