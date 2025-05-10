from __future__ import print_function

import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import Model
import SimpleITK as sitk

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
		image_z_PET = (image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))
		PET.append(image_z_PET)

		image_z_CT = image_CT[z,:,:]
		image_z_CT = (image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))
		CT.append(image_z_CT)
	
	PET = np.array(PET, dtype='float32')
	CT = np.array(CT, dtype='float32')

	return PET, CT


if __name__ == '__main__':
	MODEL_SAVE_PATH = './model/model.ckpt'
	with tf.Graph().as_default(), tf.Session() as sess:
		M = Model(BATCH_SIZE=1, INPUT_H=None, INPUT_W=None, is_training=False)
		# restore the trained model and run the style transferring
		t_list = tf.trainable_variables()
		saver = tf.train.Saver(var_list=t_list)
		model_save_path = MODEL_SAVE_PATH
		print(model_save_path)
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, model_save_path)

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
					PET_img = PET[i,:,:]
					CT_img = CT[i,:,:]

					Shape1 = PET_img.shape
					Shape2 = CT_img.shape
					print("shape1:", Shape1,"shape2:", Shape2)
					if len(Shape1) > 2:
						PET_img = PET_img[:, :, 0] * 0.3 + PET_img[:, :, 1] * 0.59 + PET_img[:, :, 2] * 0.11
					if len(Shape2) > 2:
						CT_img = CT_img[:, :, 0] * 0.3 + CT_img[:, :, 1] * 0.59 + CT_img[:, :, 2] * 0.11
					h1 = Shape1[0]
					w1 = Shape1[1]
					h2 = Shape2[0]
					w2 = Shape2[1]
					assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
					PET_img = PET_img.reshape([1, h1, w1, 1])
					CT_img = CT_img.reshape([1, h1, w1, 1])

					start = time.time()
					outputs = sess.run(M.generated_img, feed_dict={M.SOURCE1: PET_img, M.SOURCE2: CT_img})
					output = outputs[0, :, :, 0]
					end = time.time()
					print("Testing [%d] success,Testing time is [%f]\n" % (i, end - start))

					FUSION_IMG[i,:,:] = output

				if not os.path.exists(savepath+center[c]+'/'+patient_name[p]):
					os.makedirs(savepath+center[c]+'/'+patient_name[p])
				
				FUSION_z_sitk = sitk.GetImageFromArray(FUSION_IMG)
				sitk.WriteImage(FUSION_z_sitk, savepath+center[c]+'/'+patient_name[p]+'/FusionDN.nii.gz')
				
				print('E:/lvpao/fusion_result2/'+center[c]+'/'+patient_name[p])