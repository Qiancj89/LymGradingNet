import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import SGD #optimizer
from sklearn.model_selection import train_test_split
from skimage import transform

from resnet_50 import ResNet
from bayesian_resnet_50 import BayesianResNet
from utils import loadData, calculate_metrics, gen_fake_data, print_metrics
import SimpleITK as sitk
import xlrd
import random

def get_models(load_saved_model = False):
    '''Create or load models.'''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 4, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (180, 180, 1), num_classes = 4) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 4) #init bayesian resnet
    return resnet, bayesian_resnet


def train_models(resnet, bayesian_resnet, center, modality):
    '''Train models until num_epochs reached.'''
    #load training and validation datasets 
    image, label = data_load(center, modality)
    train, val, train_labels, val_labels = train_test_split(image, label, shuffle=True)

    opt1 = SGD(learning_rate = 0.00002) #setup optimizer
    opt2 = SGD(learning_rate = 0.001) #setup optimizer
    
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
              validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
               epochs = 100, batch_size = 8, optimizer = opt1, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
                        epochs = 300, batch_size = 8, optimizer = opt2, save = True) #train model


def data_load(center, modality):
    label_total = []
    images_total = []
    for c in range(0, len(center)):
        readbook = xlrd.open_workbook(r'../../../lvpao/fusion_result2/'+center[c]+'.xls')
        sheet = readbook.sheet_by_name('Sheet1')
        nrows = sheet.nrows#行
        ncols = sheet.ncols#列
        patient_name_all = np.zeros((nrows-1,ncols),dtype='<U32')
        for i in range(1,nrows):
            for j in range(0,ncols):
                patient_name_all[i-1,j] = sheet.cell(i,j).value
        
        datapath = '../../../lvpao/fusion_result2/'+center[c]
        patient_name = os.listdir(datapath)
        images = []
        label = []
        for i in range(0, len(patient_name)):
            img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/'+modality+'.nii.gz')
            image = sitk.GetArrayFromImage(img_sitk)

            for j in range(0, patient_name_all.shape[0]):
                if patient_name[i] == patient_name_all[j,0]:
                    label_ = int(float(patient_name_all[j,2])-1)

            for j in range(0, image.shape[0]):
                image_ = image[j,:,:]
                image_ = (image_-np.min(image_))/(np.max(image_)-np.min(image_))
                images.append(image_)
                label.append(label_)

        images = np.array(images, dtype='float32')
        label = np.array(label, dtype='int')
        
        zero = 0
        zero_label = []
        zero_images = []
        one = 0
        one_label = []
        one_images = []
        two = 0
        two_label = []
        two_images = []
        three = 0
        three_label = []
        three_images = []
        for i in range(0, len(label)):
            if label[i] == 0:
                zero_label.append(label[i])
                zero_images.append(images[i,:])
                zero = zero+1
            elif label[i] == 1:
                one_label.append(label[i])
                one_images.append(images[i,:])
                one = one+1
            elif label[i] == 2:
                two_label.append(label[i])
                two_images.append(images[i,:])
                two = two+1
            else:
                three_label.append(label[i])
                three_images.append(images[i,:])
                three = three+1
        
        #random.seed(2023)
        zero_idx = random.sample(range(0, zero), int(zero*0.3))
        one_idx = random.sample(range(0, one), int(one*0.3))
        two_idx = random.sample(range(0, two), int(two*0.3))
        three_idx = random.sample(range(0, three), int(three*0.3))
        
        zero_label_detect = []
        zero_images_detect = []
        one_label_detect = []
        one_images_detect = []
        two_label_detect = []
        two_images_detect = []
        three_label_detect = []
        three_images_detect = []
        for i in range(0, len(zero_idx)):
            zero_label_detect.append(zero_label[zero_idx[i]])
            zero_images_detect.append(zero_images[zero_idx[i]])
        for i in range(0, len(one_idx)):
            one_label_detect.append(one_label[one_idx[i]])
            one_images_detect.append(one_images[one_idx[i]])
        for i in range(0, len(two_idx)):
            two_label_detect.append(two_label[two_idx[i]])
            two_images_detect.append(two_images[two_idx[i]])
        for i in range(0, len(three_idx)):
            three_label_detect.append(three_label[three_idx[i]])
            three_images_detect.append(three_images[three_idx[i]])

        new_label = zero_label_detect+one_label_detect+two_label_detect+three_label_detect
        label_total = label_total+new_label
        new_images = zero_images_detect+one_images_detect+two_images_detect+three_images_detect
        images_total = images_total+new_images
    
    label_total = np.array(label_total, dtype='int')
    images_total = np.array(images_total, dtype='float32')
    
    return images_total, label_total

def main():
    #train model
    center = ['gulou', 'huaxi', 'shengrenmin']
    modality = 'FUSION_1_9'
    resnet, bayesian_resnet = get_models(load_saved_model = False) #load or create the models
    #train_models(resnet, bayesian_resnet, center, modality) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs

if __name__ == '__main__':
    main()