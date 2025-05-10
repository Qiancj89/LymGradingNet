from fusion import Fusion
import SimpleITK as sitk
import numpy as np
import os

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
		PET.append(image_z_PET)

		image_z_CT = image_CT[z,:,:]
		image_z_CT = np.uint8((image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))*255)
		CT.append(image_z_CT)
	
	PET = np.array(PET, dtype='float32')
	CT = np.array(CT, dtype='float32')

	return PET, CT

if __name__ == '__main__':
	center = ['gulou', 'huaxi', 'shengrenmin']
	savepath = '../../compare_fusion_result/'
	for c in range(0, len(center)):
		patient_name = os.listdir('E:/lvpao/fusion_result2/'+center[c]+'/')
		for p in range(0, len(patient_name)):
			patient_path = 'E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p]
			PET, CT = source_imgs_generate(patient_path)
			print(PET.shape)
			FUSION_IMG = np.zeros((PET.shape[0],180,180),dtype='float32')
			for i in range(0, PET.shape[0]):
				input_images = []
				PET_img = PET[i,:,:]
				input_images.append(PET_img)
				CT_img = CT[i,:,:]
				input_images.append(CT_img)
				FU = Fusion(input_images)
				fusion_img = FU.fuse()
				FUSION_IMG[i,:,:] = fusion_img

				
			if not os.path.exists(savepath+center[c]+'/'+patient_name[p]):
				os.makedirs(savepath+center[c]+'/'+patient_name[p])
				
			FUSION_z_sitk = sitk.GetImageFromArray(FUSION_IMG)
			sitk.WriteImage(FUSION_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/Zero_Learning_Fusion.nii.gz')
				
			print('E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p])