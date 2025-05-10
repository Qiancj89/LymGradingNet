import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import SGD #optimizer
from sklearn.model_selection import train_test_split
from skimage import transform

from resnet import ResNet
from bayesian_resnet import BayesianResNet
from utils import loadData, calculate_metrics, gen_fake_data, print_metrics
import SimpleITK as sitk
import xlrd
import random
np.random.seed(2023)

def get_models(load_saved_model = False):
    '''Create or load models.'''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 4, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (180, 180, 1), num_classes = 4) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 4) #init bayesian resnet
    return resnet, bayesian_resnet


def train_models(resnet, bayesian_resnet, center):
    '''Train models until num_epochs reached.'''
    #load training and validation datasets 
    image, label = data_load(center)
    train, val, train_labels, val_labels = train_test_split(image, label, random_state=2023, shuffle=True)

    opt1 = SGD(learning_rate = 0.00002) #setup optimizer
    opt2 = SGD(learning_rate = 0.002) #setup optimizer
    
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
              validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
               epochs = 100, batch_size = 32, optimizer = opt1, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
                        epochs = 300, batch_size = 32, optimizer = opt2, save = True) #train model


def test_model(model, data, labels, mc_steps = None):
    '''Test models and return probalities, predicted labels, entropy, and metrics.'''
    start_time = time()
    if mc_steps is None:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1))
    else:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1), mc_steps = mc_steps)
    elapsed_time = time() - start_time
    metrics = calculate_metrics(pred_labels, labels) + [elapsed_time]
    
    #print('{} ~ Accuracy: {:.4f} ~ Time: {:.2f} seconds'.format(model.model.name, metrics[0], metrics[4]))
    return pred, pred_labels, entropy, metrics

def data_load(center):
    label_total = []
    images_total = []
    images = []
    label = []
    for c in range(0, len(center)):
        readbook = xlrd.open_workbook(r'E:/lvpao1/fusion_result2/'+center[c]+'.xls')
        sheet = readbook.sheet_by_name('Sheet1')
        nrows = sheet.nrows#行
        ncols = sheet.ncols#列
        patient_name_all = np.zeros((nrows-1,ncols),dtype='<U32')
        for i in range(1,nrows):
            for j in range(0,ncols):
                patient_name_all[i-1,j] = sheet.cell(i,j).value
        
        datapath = 'E:/lvpao1/fusion_result2/'+center[c]
        patient_name = os.listdir(datapath)
        #images = []
        #label = []
        for i in range(0, len(patient_name)):
            img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/CT.nii.gz')
            image = sitk.GetArrayFromImage(img_sitk)

            for j in range(0, patient_name_all.shape[0]):
                if patient_name[i] == patient_name_all[j,0]:
                    label_ = int(float(patient_name_all[j,2])-1)

            for j in range(0, image.shape[0]):
                image_ = image[j,:,:]
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
    for i in range(0, len(label)):
        if label[i] == 0:
            zero_label.append(label[i])
            zero_images.append(images[i,:])
            zero = zero+1
        elif label[i] == 1:
            one_label.append(label[i])
            one_images.append(images[i,:])
            one = one+1
        else:
            two_label.append(label[i])
            two_images.append(images[i,:])
            two = two+1
    print(zero, one, two)
        
    random.seed(2023)
    zero_idx = random.sample(range(0, zero), int(two))
    one_idx = random.sample(range(0, one), int(two))
    two_idx = random.sample(range(0, two), int(two))
        
    zero_label_detect = []
    zero_images_detect = []
    one_label_detect = []
    one_images_detect = []
    two_label_detect = []
    two_images_detect = []
    for i in range(0, len(zero_idx)):
        zero_label_detect.append(zero_label[zero_idx[i]])
        zero_images_detect.append(zero_images[zero_idx[i]])
    for i in range(0, len(one_idx)):
        one_label_detect.append(one_label[one_idx[i]])
        one_images_detect.append(one_images[one_idx[i]])
    for i in range(0, len(two_idx)):
        two_label_detect.append(two_label[two_idx[i]])
        two_images_detect.append(two_images[two_idx[i]])

    new_label = zero_label_detect+one_label_detect+two_label_detect
    label_total = label_total+new_label
    new_images = zero_images_detect+one_images_detect+two_images_detect
    images_total = images_total+new_images
    
    label_total = np.array(label_total, dtype='int')
    images_total = np.array(images_total, dtype='float32')
    
    return images_total, label_total

def main():
    #train model
    #center = ['gulou', 'huaxi', 'shengrenmin']
    #resnet, bayesian_resnet = get_models(load_saved_model = False) #load or create the models
    #train_models(resnet, bayesian_resnet, center) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs
    
    #test models and print metrics
    
    resnet, bayesian_resnet = get_models(load_saved_model = True) #load or create the models
    center = ['gulou', 'huaxi', 'shengrenmin']
    for c in range(0, len(center)):
        readbook = xlrd.open_workbook(r'E:/lvpao1/fusion_result2/'+center[c]+'.xls')
        sheet = readbook.sheet_by_name('Sheet1')
        nrows = sheet.nrows#行
        ncols = sheet.ncols#列
        patient_name_all = np.zeros((nrows-1,ncols),dtype='<U32')
        for i in range(1,nrows):
            for j in range(0,ncols):
                patient_name_all[i-1,j] = sheet.cell(i,j).value

        datapath = 'E:/lvpao1/fusion_result2/'+center[c]
        patient_name = os.listdir(datapath)
        
        labels = []
        rpred_labelsss = []
        bpred_labelsss = []
        #images = []
        #label = []
        for i in range(0, len(patient_name)):
            img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/FUSION_2_8.nii.gz')
            image = sitk.GetArrayFromImage(img_sitk)

            for j in range(0, patient_name_all.shape[0]):
                if patient_name[i] == patient_name_all[j,0]:
                    label_ = int(float(patient_name_all[j,2])-1)

            images = []
            label = []
            for j in range(0, image.shape[0]):
                image_ = image[j,:,:]
                images.append(image_)
                label.append(label_)

            images = np.array(images, dtype='float32')
            label = np.array(label, dtype='int')
            rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, images, label)
            rpred_ = rpred[:,0:3]
            for j in range(0, rpred_.shape[0]):
                rpred_[j,:] = 1/(np.sum(rpred_[j,:]))*rpred_[j,:]
            rpred_prob = np.mean(rpred_, axis=0)
            rpred_label = np.argmax(rpred_prob, axis=0)
                        
            bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, images, label)
            bpred_ = bpred[:,0:3]
            for j in range(0, bpred_.shape[0]):
                bpred_[j,:] = 1/(np.sum(bpred_[j,:]))*bpred_[j,:]
            bpred_prob = np.mean(bpred_, axis=0)
            bpred_label = np.argmax(bpred_prob, axis=0)

            labels.append(label_)
            rpred_labelsss.append(rpred_label)
            bpred_labelsss.append(bpred_label)
        
        labels = np.array(labels, dtype='int')
        rpred_labelsss = np.array(rpred_labelsss, dtype='int')
        bpred_labelsss = np.array(bpred_labelsss, dtype='int')
        
        rmetrics = calculate_metrics(rpred_labelsss, labels)
        bmetrics = calculate_metrics(bpred_labelsss, labels)

        print_metrics('resnet', rmetrics)
        print_metrics('bayesian', bmetrics)
    

if __name__ == '__main__':
    main()