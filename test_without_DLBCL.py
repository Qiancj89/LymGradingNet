import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from resnet_50 import ResNet
from bayesian_resnet_50 import BayesianResNet
from utils import calculate_metrics, print_metrics
import SimpleITK as sitk
import random
import csv
np.random.seed(2023)

def get_models(load_saved_model = False):
    '''Create or load models.'''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 3, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (180, 180, 1), num_classes = 3) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (180, 180, 1), num_classes = 3) #init bayesian resnet
    return resnet, bayesian_resnet

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

def main():
    #test models and print metrics
    
    resnet, bayesian_resnet = get_models(load_saved_model = True) #load or create the models

    modality = 'FUSION_1_9'
    test_patient_name = []
    test_label = []
    test_patient_path = []
    f = open('../../../lvpao/fusion_result2/three_class_test_'+modality+'_subject30_random2023.csv','r')
    reader = csv.reader(f)
    for item in reader:
        if reader.line_num == 1:
            continue
        test_patient_name.append(item[0])
        test_label.append(item[1])
        test_patient_path.append(item[2])

    test_patient_name = np.array(test_patient_name, dtype='<U32')
    test_label = np.array(test_label, dtype='int')
    test_patient_path = np.array(test_patient_path, dtype='<U32')
    f.close()
    
    rpred_labelsss = []
    bpred_labelsss = []
    rpred_ = []
    bpred_ = []
    #images = []
    #label = []
    for i in range(0, len(test_label)):
        img_sitk = sitk.ReadImage('../../../lvpao/fusion_result2/'+test_patient_path[i]+'/'+test_patient_name[i]+'/'+modality+'.nii.gz')
        image = sitk.GetArrayFromImage(img_sitk)
        print(test_patient_path[i]+'/'+test_patient_name[i]+'/'+modality+'.nii.gz')

        images = []
        label = []
        for j in range(0, image.shape[0]):
            image_ = image[j,:,:]
            image_ = (image_-np.min(image_))/(np.max(image_)-np.min(image_))
            images.append(image_)
            label.append(test_label[i])

        images = np.array(images, dtype='float32')
        label = np.array(label, dtype='int')
        rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, images, label)
        rpred_prob = np.mean(rpred, axis=0)
        rpred_label = np.argmax(rpred_prob, axis=0)
                        
        bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, images, label)
        bpred_prob = np.mean(bpred, axis=0)
        bpred_label = np.argmax(bpred_prob, axis=0)

        rpred_.append(rpred_prob)
        rpred_labelsss.append(rpred_label)
        bpred_.append(bpred_prob)
        bpred_labelsss.append(bpred_label)
        
    rpred_ = np.array(rpred_, dtype='float32')
    rpred_labelsss = np.array(rpred_labelsss, dtype='int')
    bpred_ = np.array(bpred_, dtype='float32')
    bpred_labelsss = np.array(bpred_labelsss, dtype='int')
        
    rmetrics = calculate_metrics(rpred_labelsss, test_label)
    bmetrics = calculate_metrics(bpred_labelsss, test_label)

    print_metrics('resnet', rmetrics)
    print_metrics('bayesian', bmetrics)

    fname = ['patient_name', 'label', 'center', 'pred_label', 'class1', 'class2', 'class3']
    t0 = np.column_stack((test_patient_name, test_label))
    t1 = np.column_stack((t0, test_patient_path))
    t2 = np.column_stack((t1, rpred_labelsss))
    t3 = np.hstack((t2, rpred_))
    t4 = np.column_stack((t1, bpred_labelsss))
    t5 = np.hstack((t4, bpred_))
    with open('three_class/classification_model_ResNet50/three_class_test_subject30_'+modality+'_rprediction_macro.csv','w',newline='') as fi:
        writer = csv.writer(fi)
        writer.writerow(fname)
        for l in range(0, len(test_label)):
            writer.writerow(t3[l,:])
    fi.close()
    with open('three_class/classification_model_ResNet50/three_class_test_subject30_'+modality+'_bprediction_macro.csv','w',newline='') as fi:
        writer = csv.writer(fi)
        writer.writerow(fname)
        for l in range(0, len(test_label)):
            writer.writerow(t5[l,:])
    fi.close()

if __name__ == '__main__':
    main()