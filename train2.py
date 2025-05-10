# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS

from skimage import transform
from skimage.transform import radon
import SimpleITK as sitk
import os
import cv2

def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g

def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, c, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_PET = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'PET')
		SOURCE_PET = rgb2ycbcr(SOURCE_PET)
		SOURCE_CT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'SOURCE_CT')
		SOURCE_CT = rgb2ycbcr(SOURCE_CT)
		W_PET = tf.placeholder(tf.float32, shape = (BATCH_SIZE), name = 'W_PET')
		W_CT = tf.placeholder(tf.float32, shape = (BATCH_SIZE), name = 'W_CT')
		print('SOURCE_PET shape:', SOURCE_PET.shape)

		G = Generator('Generator')
		generated_img = G.transform(vis = SOURCE_PET, ir = SOURCE_CT)
		print('generate:', generated_img.shape)

		D1 = Discriminator1('Discriminator1')
		grad_of_PET = grad(tf.expand_dims(SOURCE_PET[:, :, :, 0], axis = -1))
		D1_real = D1.discrim(SOURCE_PET, reuse = False)
		D1_fake = D1.discrim(generated_img, reuse = True)

		D2 = Discriminator2('Discriminator2')
		D2_real = D2.discrim(SOURCE_CT, reuse = False)
		D2_fake = D2.discrim(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		G_loss_GAN_D1 = -tf.reduce_mean(tf.log(D1_fake + eps))
		G_loss_GAN_D2 = -tf.reduce_mean(tf.log(D2_fake + eps))
		G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2

		# surface loss
		mse1_Y = Fro_LOSS(tf.expand_dims(SOURCE_PET[:, :, :, 0], axis = -1) - tf.expand_dims(generated_img[:, :, :, 0], axis = -1))
		mse_Cb = Fro_LOSS(tf.expand_dims(SOURCE_PET[:, :, :, 1], axis = -1) - tf.expand_dims(generated_img[:, :, :, 1], axis = -1))
		mse_Cr = Fro_LOSS(tf.expand_dims(SOURCE_PET[:, :, :, 2], axis = -1) - tf.expand_dims(generated_img[:, :, :, 2], axis = -1))
		mse1 = mse1_Y
		mse_chro = mse_Cb + mse_Cr
		mse2 = Fro_LOSS(tf.expand_dims(generated_img[:,:,:,0], axis=-1) - tf.expand_dims(SOURCE_CT[:,:,:,0], axis=-1))

		LOSS_MSE = tf.reduce_mean(W_PET * mse1 + W_CT * mse2)

		LOSS_PET = L1_LOSS(grad(tf.expand_dims(generated_img[:, :, :, 0], axis = -1)) - grad_of_PET)
		G_loss = G_loss_GAN + 1.2 * LOSS_PET + 0.1 * LOSS_MSE + 0.9 * tf.reduce_mean(mse_chro)
		
		# Loss for Discriminator1
		D1_loss_real = -tf.reduce_mean(tf.log(D1_real + eps))
		D1_loss_fake = -tf.reduce_mean(tf.log(1. - D1_fake + eps))
		D1_loss = D1_loss_fake + D1_loss_real

		# Loss for Discriminator2
		D2_loss_real = -tf.reduce_mean(tf.log(D2_real + eps))
		D2_loss_fake = -tf.reduce_mean(tf.log(1. - D2_fake + eps))
		D2_loss = D2_loss_fake + D2_loss_real

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		# theta_de = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'deconv_ir')
		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator1')
		theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator2')

		G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		                                                                 var_list = theta_G)
		G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter,
		                                                             var_list = theta_G)
		# G_GAN_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		#                                                                  var_list = theta_G)
		D1_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D1_loss, global_step = current_iter,
		                                                                      var_list = theta_D1)
		D2_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D2_loss, global_step = current_iter,
		                                                                      var_list = theta_D2)

		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
		clip_D2 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D2]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 500)

		tf.summary.scalar('G_Loss_D1', G_loss_GAN_D1)
		tf.summary.scalar('G_Loss_D2', G_loss_GAN_D2)
		tf.summary.scalar('D1_real', tf.reduce_mean(D1_real))
		tf.summary.scalar('D1_fake', tf.reduce_mean(D1_fake))
		tf.summary.scalar('D2_real', tf.reduce_mean(D2_real))
		tf.summary.scalar('D2_fake', tf.reduce_mean(D2_fake))
		tf.summary.image('vis', SOURCE_PET, max_outputs = 3)
		tf.summary.image('ir', SOURCE_CT, max_outputs = 3)
		tf.summary.image('fused_img', generated_img, max_outputs = 3)

		tf.summary.scalar('Learning rate', learning_rate)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		count_loss = 0
		num_imgs = source_imgs.shape[0]

		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				PET_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0:3]
				CT_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 3:6]
				#PET_batch = np.expand_dims(PET_batch, -1)
				#CT_batch = np.expand_dims(CT_batch, -1)
				PET_en = EN(PET_batch)
				CT_en = EN(CT_batch)
				PET_intensity = intensity(PET_batch)
				CT_intensity = intensity(CT_batch)

				s_PET = PET_en + 3 * PET_intensity
				s_CT = CT_en + 3 * CT_intensity
				w_PET = np.exp(s_PET / c) / (np.exp(s_PET / c) + np.exp(s_CT / c))
				w_CT = np.exp(s_CT / c) / (np.exp(s_PET / c) + np.exp(s_CT / c))
				FEED_DICT = {SOURCE_PET: PET_batch, SOURCE_CT: CT_batch, W_PET: w_PET, W_CT: w_CT}

				it_g = 0
				it_d1 = 0
				it_d2 = 0
				# run the training step
				if batch % 2==0:
					sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
					it_d1 += 1
					sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
					it_d2 += 1
				else:
					sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
					it_g += 1
				g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict = FEED_DICT)

				if batch%2==0:
					while d1_loss > 1.7 and it_d1 < 20:
						sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
						it_d1 += 1
					while d2_loss > 1.7 and it_d2 < 20:
						sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
						d2_loss = sess.run(D2_loss, feed_dict = FEED_DICT)
						it_d2 += 1
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
				else:
					while (d1_loss < 1.4 or d2_loss < 1.4) and it_g < 20:
						sess.run([G_GAN_solver, clip_G], feed_dict = FEED_DICT)
						g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict = FEED_DICT)
						it_g += 1
					while (g_loss > 200) and it_g < 20:
						sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
						g_loss = sess.run(G_loss, feed_dict = FEED_DICT)
						it_g += 1
				print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))

				if batch % 10 == 0:
					elapsed_time = datetime.now() - start_time
					lr = sess.run(learning_rate)
					print('G_loss: %s, D1_loss: %s, D2_loss: %s' % (
						g_loss, d1_loss, d2_loss))
					print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

				result = sess.run(merged, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + '/RMSProp_step_PETCT_1.0_0.0/model.ckpt')

				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = datetime.now() - start_time
					lr = sess.run(learning_rate)
					print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
						epoch + 1, EPOCHS, step, lr, elapsed_time))
			
			saver.save(sess, save_path + '/RMSProp_epoch_PETCT_1.0_0.0/model.ckpt')
		
		writer.close()

def source_imgs_generate(data_path):
	source_imgs = []
	patient_name = os.listdir(data_path)
	for i in range(0,len(patient_name)):
		PET_sitk = sitk.ReadImage(data_path+patient_name[i]+'/PET.nii.gz')
		image_PET = sitk.GetArrayFromImage(PET_sitk)
		
		CT_sitk = sitk.ReadImage(data_path+patient_name[i]+'/resampled_CT.nii.gz')
		image_CT = sitk.GetArrayFromImage(CT_sitk)
		
		GT_sitk = sitk.ReadImage(data_path+patient_name[i]+'/Segmentation.nii.gz')
		image_GT = sitk.GetArrayFromImage(GT_sitk)
		
		z_index, _, _ = np.where(image_GT>0)
		z_index = np.unique(z_index)
		
		for z in range(0, len(z_index)):
			image_z_PET = image_PET[z_index[z],:,:]
			image_z_PET = np.abs(image_z_PET)
			image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
			image_z_PET = image_z_PET[60:160,40:140]
			image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
			image_z_PET = np.uint8((image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))*255)
			image_PET_z = cv2.cvtColor(image_z_PET, cv2.COLOR_GRAY2RGB)
			image_PET_z = image_PET_z/255

			image_z_CT = image_CT[z_index[z],:,:]
			image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
			image_z_CT = image_z_CT[60:160,40:140]
			image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
			image_z_CT = np.uint8((image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))*255)
			image_CT_z = cv2.cvtColor(image_z_CT, cv2.COLOR_GRAY2RGB)
			image_CT_z = image_CT_z/255

			source_img = np.zeros((180, 180, 6), dtype='float32')
			source_img[:,:,0:3] = image_PET_z
			source_img[:,:,0:3] = image_CT_z

			source_imgs.append(source_img)
		print(data_path+patient_name[i])

	source_imgs = np.array(source_imgs, dtype='float32')
	return source_imgs

def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len))
	grey_level = 256
	counter = np.zeros(shape = (grey_level))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		for m in range(patch_size):
			for n in range(patch_size):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies

def intensity(inputs):
	len = inputs.shape[0]
	intensities = np.zeros(shape = (len))
	for i in range(len):
		input = inputs[i, :, :, 0]
		logic = input > 0.85
		input = input * logic
		input = input.reshape([-1])
		# exist = (input != 0)
		num = input.sum(axis = 0)
		# den = exist.sum(axis = 0)
		intensities[i] = num/(inputs.shape[1] * inputs.shape[2])# / den
	return intensities

def rgb2ycbcr(img_rgb):
	R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr

def rgb2ycbcr_np(img_rgb):
	R = np.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = np.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = np.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
	img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
	return img_ycbcr

if __name__ == '__main__':
	data_path = '../../data/huaxi/'
	source_imgs = source_imgs_generate(data_path)
	patch_size = 180
	# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

	LEARNING_RATE = 0.00002
	EPSILON = 1e-5
	DECAY_RATE = 0.9
	eps = 1e-8
	save_path = 'model2/'
	EPOCHES_set = 1
	BATCH_SIZE = 16
	c = 1

	train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, c, logging_period = 1)