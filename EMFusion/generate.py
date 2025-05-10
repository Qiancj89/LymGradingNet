# Use a trained DenseFuse Net to generate fused images

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from datetime import datetime
from os import listdir
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
import SimpleITK as sitk
from skimage import transform
import time
import os
import cv2


def generate(pet, mri, model_path):
	ir_img = pet / 255.0
	vis_img = mri / 255.0
	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	ir_dimension.insert(0, 1)
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:

		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_ir = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')

		G = Generator('Generator')
		output_image = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
		output = output[0, :, :, 0]

		return output


def source_imgs_generate(patient_path):
	PET = []
	CT = []

	PET_sitk = sitk.ReadImage(patient_path+'/PET.nii.gz')
	image_PET = sitk.GetArrayFromImage(PET_sitk)
		
	CT_sitk = sitk.ReadImage(patient_path+'/CT.nii.gz')
	image_CT = sitk.GetArrayFromImage(CT_sitk)
		
	GT_sitk = sitk.ReadImage(patient_path+'/GT.nii.gz')
	image_GT = sitk.GetArrayFromImage(GT_sitk)
		
	for z in range(0, image_PET.shape[0]):
		image_z_PET = image_PET[z,:,:]
		image_z_PET = np.abs(image_z_PET)
		image_z_PET = transform.resize(image_z_PET, (45, 45), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
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
	model_path = 'model/model.ckpt'
	for c in range(0, len(center)):
		patient_name = os.listdir('E:/lvpao/fusion_result2/'+center[c]+'/')
		for p in range(0, len(patient_name)):
			patient_path = 'E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p]
			PET, CT = source_imgs_generate(patient_path)
			print(PET.shape)
			FUSION_IMG = np.zeros((PET.shape[0],180,180),dtype='float32')
			for i in range(0, PET.shape[0]):
				PET_img = PET[i,:,:]
				CT_img = CT[i,:,:]
				fusion = generate(PET_img, CT_img, model_path)
				FUSION_IMG[i,:,:] = fusion

				
			if not os.path.exists(savepath+center[c]+'/'+patient_name[p]):
				os.makedirs(savepath+center[c]+'/'+patient_name[p])
				
			FUSION_z_sitk = sitk.GetImageFromArray(FUSION_IMG)
			sitk.WriteImage(FUSION_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/DDcGAN_Fusion.nii.gz')
				
			print('E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p])


