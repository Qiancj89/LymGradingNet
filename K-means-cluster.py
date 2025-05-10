from __future__ import division
from __future__ import print_function

import time

import xlrd
import os
import SimpleITK as sitk
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed
seed = 2023

readbook = xlrd.open_workbook(r'E:/lvpao/fusion_result2/huaxi.xls')
sheet = readbook.sheet_by_name('Sheet1')
nrows = sheet.nrows#行
ncols = sheet.ncols#列
patient_name_all = np.zeros((nrows-1,ncols),dtype='<U32')
for i in range(1,nrows):
    for j in range(0,ncols):
        patient_name_all[i-1,j] = sheet.cell(i,j).value

datapath = 'E:/lvpao/fusion_result2/huaxi'
patient_name = os.listdir(datapath)
features = []
label = []
for i in range(0, len(patient_name)):
    img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/bayesian_resnet_FUSION_2_8_1.nii.gz')
    image = sitk.GetArrayFromImage(img_sitk)

    pet_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/bayesian_resnet_PET_1.nii.gz')
    pet = sitk.GetArrayFromImage(pet_sitk)

    for j in range(0, patient_name_all.shape[0]):
        if patient_name[i] == patient_name_all[j,0]:
            label_ = int(float(patient_name_all[j,2]))

    for j in range(0, image.shape[0]):
        image1_ = image[j,:,:]
        feature1 = image1_.flatten()
        image2_ = pet[j,:,:]
        feature2 = image2_.flatten()
        feature = np.hstack((feature1, feature2))
        features.append(feature)
        label.append(label_)

features = np.array(features, dtype='float32')
label = np.array(label, dtype='int')

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.5, random_state=seed, shuffle=True)

kmeans = KMeans(n_clusters=4, random_state=seed)
kmeans.fit(X_train)
 
# 预测聚类结果
y_pred = kmeans.predict(X_test)

for i in range(0, len(y_test)):
    print(y_test[i], y_pred[i])