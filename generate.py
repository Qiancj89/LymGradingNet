# Use a trained DenseFuse Net to generate fused images

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from skimage import transform
from skimage.transform import radon, iradon
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import os

def generate(CT, PET, model_path, output_path = None):
	print(CT.shape)
	CT_dimension = list(CT.shape)
	PET_dimension = list(PET.shape)
	CT_dimension.insert(0, 1)
	#CT_dimension.append(1)
	PET_dimension.insert(0, 1)
	#PET_dimension.append(1)
	CT = CT.reshape(CT_dimension)
	PET = PET.reshape(PET_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_PET = tf.placeholder(tf.float32, shape = PET_dimension, name = 'SOURCE_PET')
		SOURCE_PET = rgb2ycbcr(SOURCE_PET)
		SOURCE_CT = tf.placeholder(tf.float32, shape = CT_dimension, name = 'SOURCE_CT')
		SOURCE_CT = rgb2ycbcr(SOURCE_CT)
		# source_field = tf.placeholder(tf.float32, shape = source_shape, name = 'source_imgs')

		G = Generator('Generator')
		output_image = G.transform(vis = SOURCE_PET, ir = SOURCE_CT)
		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_PET: PET, SOURCE_CT: CT})
		output = output[0, :, :, 0]
	return output

def source_imgs_generate(patient_path):
	PET = []
	CT = []
	GT = []

	PET_sitk = sitk.ReadImage(patient_path+'/PET.nii.gz')
	image_PET = sitk.GetArrayFromImage(PET_sitk)
		
	CT_sitk = sitk.ReadImage(patient_path+'/resampled_CT.nii.gz')
	image_CT = sitk.GetArrayFromImage(CT_sitk)
		
	GT_sitk = sitk.ReadImage(patient_path+'/Segmentation.nii.gz')
	image_GT = sitk.GetArrayFromImage(GT_sitk)
		
	z_index, _, _ = np.where(image_GT>0)
	z_index = np.unique(z_index)
		
	for z in range(0, len(z_index)):
		image_z_PET = image_PET[z_index[z],:,:]
		image_z_PET = np.abs(image_z_PET)
		image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		image_z_PET = image_z_PET[60:160,60:160]
		image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		#image_z_PET = (image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))
		image_z_PET = np.uint8((image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))*255)
		image_PET_z = cv2.cvtColor(image_z_PET, cv2.COLOR_GRAY2RGB)
		image_PET_z = image_PET_z/255
		PET.append(image_PET_z)

		image_z_CT = image_CT[z_index[z],:,:]
		image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		image_z_CT = image_z_CT[60:160,60:160]
		image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		image_z_CT = np.uint8((image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))*255)
		image_CT_z = cv2.cvtColor(image_z_CT, cv2.COLOR_GRAY2RGB)
		image_CT_z = image_CT_z/255
		CT.append(image_CT_z)

		image_z_GT = image_GT[z_index[z],:,:]
		image_z_GT = transform.resize(image_z_GT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		image_z_GT = image_z_GT[60:160,60:160]
		image_z_GT = transform.resize(image_z_GT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
		image_z_GT[image_z_GT>0] = 1
		GT.append(image_z_GT)
	
	PET = np.array(PET, dtype='float32')
	CT = np.array(CT, dtype='float32')
	GT = np.array(GT, dtype='float32')

	return PET, CT, GT

def rgb2ycbcr(img_rgb):
	R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr

if __name__ == '__main__':
	center = ['gulou', 'huaxi', 'shengrenmin']
	savepath = '../../fusion_result/'
	for n in range(1,10):
		model_path = 'model/RMSProp_epoch_PETCT_0.'+str(n)+'_0.'+str(10-n)+'/model.ckpt'
		for c in range(2, 3):
			patient_name = os.listdir('../../data/'+center[c]+'/')
			for p in range(147, 148):
				patient_path = '../../data/'+center[c]+'/'+patient_name[p]
				PET, CT, GT = source_imgs_generate(patient_path)
				PET_IMG = np.zeros((PET.shape[0],180,180),dtype='float32')
				CT_IMG = np.zeros((CT.shape[0],180,180),dtype='float32')
				FUSION_IMG = np.zeros((PET.shape[0],180,180),dtype='float32')
				for i in range(0, PET.shape[0]):
					PET_img = PET[i,:,:,:]
					CT_img = CT[i,:,:,:]
					GT_img = GT[i,:,:]
					fusion = generate(CT_img, PET_img, model_path)
					PET_IMG[i,:,:] = PET_img[:,:,0]
					CT_IMG[i,:,:] = CT_img[:,:,0]
					FUSION_IMG[i,:,:] = fusion
					
					'''
					fig = plt.figure()
					ax1 = fig.add_subplot(2,5,1)
					ax1.set_title("Original CT")
					ax1.imshow(CT_img[:,:,0], cmap=plt.cm.Greys_r)

					ax2 = fig.add_subplot(2,5,2)
					ax2.set_title("Original PET")
					ax2.imshow(PET_img[:,:,0], cmap=plt.cm.Greys_r)
						
					ax3 = fig.add_subplot(2,5,3)
					ax3.set_title("Reconstruction\nFiltered back projection")
					ax3.imshow(fusion, cmap=plt.cm.Greys_r)

					ax4 = fig.add_subplot(2,5,4)
					ax4.set_title("Adding")
					ax4.imshow(CT_img[:,:,0]+PET_img[:,:,0], cmap=plt.cm.Greys_r)

					ax5 = fig.add_subplot(2,5,5)
					ax5.set_title("Ground Truth")
					ax5.imshow(GT_img, cmap=plt.cm.Greys_r)

					ax6 = fig.add_subplot(2,5,6)
					ax6.set_title("Original CT")
					ax6.imshow(CT_img[:,:,0], cmap=plt.cm.seismic)

					ax7 = fig.add_subplot(2,5,7)
					ax7.set_title("Original PET")
					ax7.imshow(PET_img[:,:,0], cmap=plt.cm.seismic)
						
					ax8 = fig.add_subplot(2,5,8)
					ax8.set_title("Reconstruction\nFiltered back projection")
					ax8.imshow(fusion, cmap=plt.cm.seismic)

					ax9 = fig.add_subplot(2,5,9)
					ax9.set_title("Adding")
					ax9.imshow(CT_img[:,:,0]+PET_img[:,:,0], cmap=plt.cm.seismic)

					fig.tight_layout()
					plt.show()
					'''
				
				if not os.path.exists(savepath+center[c]+'/'+patient_name[p]):
					os.makedirs(savepath+center[c]+'/'+patient_name[p])
				
				PET_z_sitk = sitk.GetImageFromArray(PET_IMG)
				sitk.WriteImage(PET_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/PET.nii.gz')

				CT_z_sitk = sitk.GetImageFromArray(CT_IMG)
				sitk.WriteImage(CT_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/CT.nii.gz')

				GT_z_sitk = sitk.GetImageFromArray(GT)
				sitk.WriteImage(GT_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/GT.nii.gz')
				
				FUSION_z_sitk = sitk.GetImageFromArray(FUSION_IMG)
				sitk.WriteImage(FUSION_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/FUSION_'+str(n)+'_'+str(10-n)+'.nii.gz')
				
				print('../../data/'+center[c]+'/'+patient_name[p])