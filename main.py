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
np.random.seed(2023)

def get_models(load_saved_model = False):
    '''Create or load models.'''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (50, 50, 1), num_classes = 2, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (50, 50, 1), num_classes = 2) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (50, 50, 1), num_classes = 2) #init bayesian resnet
    return resnet, bayesian_resnet


def train_models(resnet, bayesian_resnet, datapath, num_epochs = 10):
    '''Train models until num_epochs reached.'''
    #load training and validation datasets 
    image, label = load_data(datapath)
    train, val, train_labels, val_labels = train_test_split(image, label, random_state=None, shuffle=True)
    
    opt = SGD(learning_rate = 1e-3) #setup optimizer
    
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
              validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
               epochs = num_epochs, batch_size = 32, optimizer = opt, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = (np.expand_dims(val, axis = -1), to_categorical(val_labels)),
                        epochs = num_epochs, batch_size = 32, optimizer = opt, save = True) #train model


def test_model(model, data, labels, mc_steps = None):
    '''Test models and return probalities, predicted labels, entropy, and metrics.'''
    start_time = time()
    if mc_steps is None:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1))
    else:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1), mc_steps = mc_steps)
    elapsed_time = time() - start_time
    metrics = calculate_metrics(pred_labels, labels) + [elapsed_time]
    
    print('{} ~ Accuracy: {:.4f} ~ Time: {:.2f} seconds'.format(model.model.name, metrics[0], metrics[4]))
    return pred, pred_labels, entropy, metrics

def load_data(datapath):
    patient_name = os.listdir(datapath)
    image_total = []
    label = []
    for i in range(0, len(patient_name)):
        img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/PET.nii.gz')
        image = sitk.GetArrayFromImage(img_sitk)

        GT_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/GT.nii.gz')
        GT = sitk.GetArrayFromImage(GT_sitk)

        for j in range(0, GT.shape[0]):
            GT_ = GT[j,:,:]
            image_ = image[j,:,:]
            x_1, y_1 = np.where(GT_==1)
            #print(len(x_1))
            x_1[x_1<25] = 25
            x_1[x_1>GT_.shape[0]-25] = GT_.shape[0]-25
            y_1[y_1<25] = 25
            y_1[y_1>GT_.shape[1]-25] = GT_.shape[1]-25
            GT_[0:25,:] = 2
            GT_[:,0:25] = 2
            GT_[GT_.shape[0]-25:GT_.shape[0],:] = 2
            GT_[:,GT_.shape[1]-25:GT_.shape[1]] = 2
            x_0, y_0 = np.where(GT_==0)

            if len(x_1) < 32:
                ind = np.random.randint(0, len(x_0), len(x_1))
                for k in range(0, len(x_1)):
                    image_total.append(image_[x_1[k]-25:x_1[k]+25, y_1[k]-25:y_1[k]+25])
                    '''
                    plt.figure("Image")  # 图像窗口名称
                    plt.imshow(image_[x_1[k]-25:x_1[k]+25, y_1[k]-25:y_1[k]+25],plt.cm.seismic)
                    plt.axis('on')  # 关掉坐标轴为 off
                    plt.title('image')  # 图像题目
                    plt.show()
                    '''
                    label.append(1)
                    image_total.append(image_[x_0[ind[k]]-25:x_0[ind[k]]+25, y_0[ind[k]]-25:y_0[ind[k]]+25])
                    label.append(0)
                    #print(label)
            else:
                ind_x1 = np.random.randint(0, len(x_1), 32)
                ind_x0 = np.random.randint(0, len(x_0), 32)
                for k in range(0, 32):
                    image_total.append(image_[x_1[ind_x1[k]]-25:x_1[ind_x1[k]]+25, y_1[ind_x1[k]]-25:y_1[ind_x1[k]]+25])
                    label.append(1)
                    image_total.append(image_[x_0[ind_x0[k]]-25:x_0[ind_x0[k]]+25, y_0[ind_x0[k]]-25:y_0[ind_x0[k]]+25])
                    label.append(0)
    
    image_total = np.array(image_total, dtype='float32')
    label = np.array(label, dtype='int')
    return image_total, label

def main():
    #train model
    #datapath = 'E:/lvpao/fusion_result2/huaxi'
    #bayesian_resnet = get_models(load_saved_model = False) #load or create the models
    #train_models(bayesian_resnet, datapath, num_epochs = 100) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs
    
    #test models and print metrics
    resnet, bayesian_resnet = get_models(load_saved_model = True) #load or create the models
    center = ['gulou', 'huaxi', 'shengrenmin']
    for c in range(0, len(center)):
        datapath = 'E:/lvpao/fusion_result2/'+center[c]
        patient_name = os.listdir(datapath)
        for i in range(0, len(patient_name)):
            img_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/CT.nii.gz')
            image = sitk.GetArrayFromImage(img_sitk)

            GT_sitk = sitk.ReadImage(datapath+'/'+patient_name[i]+'/GT.nii.gz')
            GT = sitk.GetArrayFromImage(GT_sitk)
            z_index, _, _ = np.where(GT>0)
            z_index = np.unique(z_index)

            resnet_PET = np.zeros((len(z_index),50,50),dtype='float32')
            bayesian_resnet_PET_0 = np.zeros((len(z_index),50,50),dtype='float32')
            bayesian_resnet_PET_1 = np.zeros((len(z_index),50,50),dtype='float32')

            for j in range(0, len(z_index)):
                patch = []
                label = []
                GT_ = GT[z_index[j],:,:]
                image_ = image[z_index[j],:,:]
                x_1, y_1 = np.where(GT_==1)
                x_1_min = np.min(x_1)
                x_1_max = np.max(x_1)
                y_1_min = np.min(y_1)
                y_1_max = np.max(y_1)
                x_length = x_1_max-x_1_min
                y_length = y_1_max-y_1_min

                if x_length>50 or y_length>50:
                    GT_crop = GT_[x_1_min:x_1_max,y_1_min:y_1_max]
                    image_crop = image_[x_1_min:x_1_max,y_1_min:y_1_max]
                    image_crop = transform.resize(image_crop, (50, 50), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
                    GT_crop = transform.resize(GT_crop, (50, 50), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
                    GT_crop[GT_crop>0] = 1
                else:
                    if x_1_min<25:
                        x_1_min=25
                    else:
                        x_1_min_length = image_.shape[0]-x_1_min
                        if x_1_min_length<25+int(x_length/2):
                            x_1_min = image_.shape[0]-25-int(x_length/2)
                    if y_1_min<25:
                        y_1_min=25
                    else:
                        y_1_min_length = image_.shape[1]-y_1_min
                        if y_1_min_length<25+int(y_length/2):
                            y_1_min = image_.shape[1]-25-int(y_length/2)
                    GT_crop = GT_[x_1_min+int(x_length/2)-25:x_1_min+int(x_length/2)+25,y_1_min+int(y_length/2)-25:y_1_min+int(y_length/2)+25]
                    image_crop = image_[x_1_min+int(x_length/2)-25:x_1_min+int(x_length/2)+25,y_1_min+int(y_length/2)-25:y_1_min+int(y_length/2)+25]

                GT_expand = np.zeros((GT_crop.shape[0]+50,GT_crop.shape[1]+50),dtype='int')
                GT_expand[25:GT_crop.shape[0]+25,25:GT_crop.shape[1]+25] = GT_crop
                image_expand = np.zeros((image_crop.shape[0]+50,image_crop.shape[1]+50),dtype='float32')
                image_expand[25:image_crop.shape[0]+25,25:image_crop.shape[1]+25] = image_crop

                for k in range(25,image_crop.shape[0]+25):
                    for l in range(25,image_crop.shape[1]+25):
                        patch_ = image_expand[k-25:k+25,l-25:l+25]
                        patch.append(patch_)
                        label.append(GT_expand[k,l])
                
                patch = np.array(patch, dtype='float32')
                label = np.array(label, dtype='int')

                rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, patch, label)
                pred_prob_resnet = np.reshape(rpred[:,1], (image_crop.shape[0], image_crop.shape[1]))
                resnet_PET[j,:,:] = pred_prob_resnet
                
                bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, patch, label)
                pred_prob_bayesian_resnet = np.reshape(bpred[:,0], (image_crop.shape[0], image_crop.shape[1]))
                bayesian_resnet_PET_0[j,:,:] = pred_prob_bayesian_resnet
                pred_prob_bayesian_resnet = np.reshape(bpred[:,1], (image_crop.shape[0], image_crop.shape[1]))
                bayesian_resnet_PET_1[j,:,:] = pred_prob_bayesian_resnet
                

                '''
                fig = plt.figure()
                ax1 = fig.add_subplot(2,2,1)
                ax1.set_title("Original PET")
                ax1.imshow(image_crop, cmap=plt.cm.seismic)
                
                ax2 = fig.add_subplot(2,2,2)
                ax2.set_title("Resnet result")
                ax2.imshow(pred_prob_resnet, cmap=plt.cm.seismic)

                ax3 = fig.add_subplot(2,2,3)
                ax3.set_title("Bayesian_resnet result")
                ax3.imshow(pred_prob_bayesian_resnet, cmap=plt.cm.seismic)

                ax4 = fig.add_subplot(2,2,4)
                ax4.set_title("Ground Truth")
                ax4.imshow(GT_crop, cmap=plt.cm.seismic)
                
                fig.tight_layout()
                plt.show()
                
                print_metrics('resnet', rmetrics)
                print_metrics('bayesian', bmetrics)
                '''

            resnet_PET_sitk = sitk.GetImageFromArray(resnet_PET)
            sitk.WriteImage(resnet_PET_sitk, datapath+'/'+patient_name[i]+'/resnet_CT.nii.gz')

            
            bayesian_resnet_PET_sitk_0 = sitk.GetImageFromArray(bayesian_resnet_PET_0)
            sitk.WriteImage(bayesian_resnet_PET_sitk_0, datapath+'/'+patient_name[i]+'/bayesian_resnet_CT_0.nii.gz')

            bayesian_resnet_PET_sitk_1 = sitk.GetImageFromArray(bayesian_resnet_PET_1)
            sitk.WriteImage(bayesian_resnet_PET_sitk_1, datapath+'/'+patient_name[i]+'/bayesian_resnet_CT_1.nii.gz')
            

            print(datapath+'/'+patient_name[i])

    '''
    #eliminate predictions with high entropy and print new metrics
    idx1 = np.argwhere(rentropy < 0.50)
    idx2 = np.argwhere(bentropy < 0.50)
    print_metrics('resnet', calculate_metrics(rpred_labels[idx1], test_labels[idx1]))
    print_metrics('bayesian', calculate_metrics(bpred_labels[idx2], test_labels[idx2]))

    #generate fake data and create a mixed datasets
    fake, fake_labels = gen_fake_data(300)
    mixed = np.concatenate((test, fake), axis = 0)
    mixed_labels = np.concatenate((test_labels, fake_labels), axis = 0)
    
    #run prediction on mixed dataset and get metrics
    rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, mixed, mixed_labels)
    bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, mixed, mixed_labels)
    
    print_metrics('resnet', rmetrics)
    print_metrics('bayesian', bmetrics)
    
    #eliminate predictions with high entropy on mixed dataset and print new metrics
    idx1 = np.argwhere(rentropy < 0.50)
    idx2 = np.argwhere(bentropy < 0.50)
    print_metrics('resnet', calculate_metrics(rpred_labels[idx1], mixed_labels[idx1]))
    print_metrics('bayesian', calculate_metrics(bpred_labels[idx2], mixed_labels[idx2]))
    '''


if __name__ == '__main__':
    main()