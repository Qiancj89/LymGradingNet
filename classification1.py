import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import numpy as np

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import SGD #optimizer
from sklearn.model_selection import train_test_split

from resnet_50 import ResNet
from bayesian_resnet_50 import BayesianResNet
import SimpleITK as sitk
import xlrd
import random
import csv
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


def train_models(resnet, bayesian_resnet, center, modality):
    '''Train models until num_epochs reached.'''
    #load training and validation datasets 
    image, label = data_load(center, modality)
    train, val, train_labels, val_labels = train_test_split(image, label, random_state=2023, shuffle=True)

    opt1 = SGD(learning_rate = 0.00002) #setup optimizer
    opt2 = SGD(learning_rate = 0.001) #setup optimizer
    
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
               validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
               epochs = 250, batch_size = 8, optimizer = opt1, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
                        epochs = 640, batch_size = 8, optimizer = opt2, save = True) #train model


def data_load(center, modality):
    train_label, train_patient_name, train_patient_path = train_test_group(center, modality)
    label = []
    images = []
    for i in range(0, len(train_label)):
        img_sitk = sitk.ReadImage('../../../lvpao/fusion_result2/'+train_patient_path[i]+'/'+train_patient_name[i]+'/'+modality+'.nii.gz')
        image = sitk.GetArrayFromImage(img_sitk)

        for j in range(0, image.shape[0]):
            image_ = image[j,:,:]
            image_ = (image_-np.min(image_))/(np.max(image_)-np.min(image_))
            images.append(image_)
            label.append(train_label[i])

    images = np.array(images, dtype='float32')
    label = np.array(label, dtype='int')

    return images, label

def train_test_group(center, modality):
    label = []
    patient_name_total = []
    patient_name_path = []
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

        for i in range(0, len(patient_name)):
            for j in range(0, patient_name_all.shape[0]):
                if patient_name[i] == patient_name_all[j,0]:
                    patient_name_total.append(patient_name[i])
                    patient_name_path.append(center[c])
                    label_ = int(float(patient_name_all[j,2])-1)
                    label.append(label_)

    label = np.array(label, dtype='int')
    patient_name_total = np.array(patient_name_total, dtype='<U32')
    patient_name_path = np.array(patient_name_path, dtype='<U32')
        
    zero = 0
    zero_label = []
    zero_patient_name = []
    zero_patient_path = []
    one = 0
    one_label = []
    one_patient_name = []
    one_patient_path = []
    two = 0
    two_label = []
    two_patient_name = []
    two_patient_path = []
    three = 0
    three_label = []
    three_patient_name = []
    three_patient_path = []
    for i in range(0, len(label)):
        if label[i] == 0:
            zero_label.append(label[i])
            zero_patient_name.append(patient_name_total[i])
            zero_patient_path.append(patient_name_path[i])
            zero = zero+1
        elif label[i] == 1:
            one_label.append(label[i])
            one_patient_name.append(patient_name_total[i])
            one_patient_path.append(patient_name_path[i])
            one = one+1
        elif label[i] == 2:
            two_label.append(label[i])
            two_patient_name.append(patient_name_total[i])
            two_patient_path.append(patient_name_path[i])
            two = two+1
        else:
            three_label.append(label[i])
            three_patient_name.append(patient_name_total[i])
            three_patient_path.append(patient_name_path[i])
            three = three+1

    random.seed(2023)    
    zero_idx = random.sample(range(0, zero), zero)
    zero_train_idx = zero_idx[:int(zero*0.3)]
    zero_test_idx = zero_idx[int(zero*0.3):]
    one_idx = random.sample(range(0, one), one)
    one_train_idx = one_idx[:int(one*0.3)]
    one_test_idx = one_idx[int(one*0.3):]
    two_idx = random.sample(range(0, two), two)
    two_train_idx = two_idx[:int(two*0.3)]
    two_test_idx = two_idx[int(two*0.3):]
    three_idx = random.sample(range(0, three), three)
    three_train_idx = three_idx[:int(three*0.3)]
    three_test_idx = three_idx[int(three*0.3):]
        
    zero_train_label = []
    zero_test_label = []
    zero_train_patient_name = []
    zero_train_patient_path = []
    zero_test_patient_name = []
    zero_test_patient_path = []
    one_train_label = []
    one_test_label = []
    one_train_patient_name = []
    one_train_patient_path = []
    one_test_patient_name = []
    one_test_patient_path = []
    two_train_label = []
    two_test_label = []
    two_train_patient_name = []
    two_train_patient_path = []
    two_test_patient_name = []
    two_test_patient_path = []
    three_train_label = []
    three_test_label = []
    three_train_patient_name = []
    three_train_patient_path = []
    three_test_patient_name = []
    three_test_patient_path = []
    for i in range(0, len(zero_train_idx)):
        zero_train_label.append(zero_label[zero_train_idx[i]])
        zero_train_patient_name.append(zero_patient_name[zero_train_idx[i]])
        zero_train_patient_path.append(zero_patient_path[zero_train_idx[i]])
    for i in range(0, len(zero_test_idx)):
        zero_test_label.append(zero_label[zero_test_idx[i]])
        zero_test_patient_name.append(zero_patient_name[zero_test_idx[i]])
        zero_test_patient_path.append(zero_patient_path[zero_test_idx[i]])
    
    for i in range(0, len(one_train_idx)):
        one_train_label.append(one_label[one_train_idx[i]])
        one_train_patient_name.append(one_patient_name[one_train_idx[i]])
        one_train_patient_path.append(one_patient_path[one_train_idx[i]])
    for i in range(0, len(one_test_idx)):
        one_test_label.append(one_label[one_test_idx[i]])
        one_test_patient_name.append(one_patient_name[one_test_idx[i]])
        one_test_patient_path.append(one_patient_path[one_test_idx[i]])

    for i in range(0, len(two_train_idx)):
        two_train_label.append(two_label[two_train_idx[i]])
        two_train_patient_name.append(two_patient_name[two_train_idx[i]])
        two_train_patient_path.append(two_patient_path[two_train_idx[i]])
    for i in range(0, len(two_test_idx)):
        two_test_label.append(two_label[two_test_idx[i]])
        two_test_patient_name.append(two_patient_name[two_test_idx[i]])
        two_test_patient_path.append(two_patient_path[two_test_idx[i]])

    for i in range(0, len(three_train_idx)):
        three_train_label.append(three_label[three_train_idx[i]])
        three_train_patient_name.append(three_patient_name[three_train_idx[i]])
        three_train_patient_path.append(three_patient_path[three_train_idx[i]])
    for i in range(0, len(three_test_idx)):
        three_test_label.append(three_label[three_test_idx[i]])
        three_test_patient_name.append(three_patient_name[three_test_idx[i]])
        three_test_patient_path.append(three_patient_path[three_test_idx[i]])
    
    train_label = zero_train_label+one_train_label+two_train_label+three_train_label
    train_label = np.array(train_label, dtype='int')
    train_patient_name = zero_train_patient_name+one_train_patient_name+two_train_patient_name+three_train_patient_name
    train_patient_name = np.array(train_patient_name, dtype='<U32')
    train_patient_path = zero_train_patient_path+one_train_patient_path+two_train_patient_path+three_train_patient_path
    train_patient_path = np.array(train_patient_path, dtype='<U32')
    test_label = zero_test_label+one_test_label+two_test_label+three_test_label
    test_label = np.array(test_label, dtype='int')
    test_patient_name = zero_test_patient_name+one_test_patient_name+two_test_patient_name+three_test_patient_name
    test_patient_name = np.array(test_patient_name, dtype='<U32')
    test_patient_path = zero_test_patient_path+one_test_patient_path+two_test_patient_path+three_test_patient_path
    test_patient_path = np.array(test_patient_path, dtype='<U32')

    '''
    fname = ['patient_name', 'label', 'patient_path']
    train1 = np.column_stack((train_patient_name, train_label))
    train2 = np.column_stack((train1, train_patient_path))
    with open('E:/lvpao/fusion_result2/train_'+modality+'_subject30_random2023.csv','w',newline='') as fi:
        writer = csv.writer(fi)
        writer.writerow(fname)
        for l in range(0, train2.shape[0]):
            writer.writerow(train2[l,:])
    fi.close()
    test1 = np.column_stack((test_patient_name, test_label))
    test2 = np.column_stack((test1, test_patient_path))
    with open('E:/lvpao/fusion_result2/test_'+modality+'_subject30_random2023.csv','w',newline='') as fi:
        writer = csv.writer(fi)
        writer.writerow(fname)
        for l in range(0, test2.shape[0]):
            writer.writerow(test2[l,:])
    fi.close()
    '''
    
    return train_label, train_patient_name, train_patient_path

def main():
    #train model
    center = ['gulou', 'huaxi', 'shengrenmin']
    modality = 'FUSION_1_9'
    resnet, bayesian_resnet = get_models(load_saved_model = True) #load or create the models
    train_models(resnet, bayesian_resnet, center, modality) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs

if __name__ == '__main__':
    main()