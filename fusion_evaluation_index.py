import SimpleITK as sitk
import numpy as np
import os
import csv
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
 
# 归一化
def MaxMinNormalizer(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_normalize = (data - data_min) / (data_max - data_min)
    return data_normalize

def PSNR(image1, image2):
    mse = np.mean((image1 - image2)**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

def AG(image):
    grad_x = np.abs(np.gradient(image)[0])
    grad_y = np.abs(np.gradient(image)[1])
    average_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    return average_gradient
 
def main():
    modality = ['FUSION_non_lamda']
    for m in range(0, len(modality)):
        test_patient_name = []
        test_label = []
        test_patient_path = []
        f = open('../../lvpao/fusion_result2/test_'+modality[m]+'_subject30_random2023.csv','r')
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
        
        ssim_all = []
        psnr_all = []
        AG_all = []
        entropy_all = []
        patient_name = []
        label = []
        path = []
        for i in range(0, len(test_label)):
            pet_sitk = sitk.ReadImage('../../lvpao/fusion_result2/'+test_patient_path[i]+'/'+test_patient_name[i]+'/PET.nii.gz')
            pet_image = sitk.GetArrayFromImage(pet_sitk)
            if np.max(pet_image)<=1:
                pet_image = np.uint8(pet_image*255)

            ct_sitk = sitk.ReadImage('../../lvpao/fusion_result2/'+test_patient_path[i]+'/'+test_patient_name[i]+'/CT.nii.gz')
            ct_image = sitk.GetArrayFromImage(ct_sitk)
            if np.max(ct_image)<=1:
                ct_image = np.uint8(ct_image*255)

            img_sitk = sitk.ReadImage('../../lvpao/fusion_result2/'+test_patient_path[i]+'/'+test_patient_name[i]+'/'+modality[m]+'.nii.gz')
            image = sitk.GetArrayFromImage(img_sitk)
            if np.max(image)<=1:
                image = np.uint8(image*255)

            for n in range(0, pet_image.shape[0]):
                ssim1 = ssim(pet_image[n,:,:], image[n,:,:], multichannel=False)
                ssim2 = ssim(ct_image[n,:,:], image[n,:,:], multichannel=False)
                ssim_final = (ssim1+ssim2)/2
                ssim_all.append(ssim_final)

                psnr1 = PSNR(pet_image[n,:,:], image[n,:,:])
                psnr2 = PSNR(ct_image[n,:,:], image[n,:,:])
                psnr = (psnr1+psnr2)/2
                psnr_all.append(psnr)

                average_gradient = AG(image[n,:,:])
                AG_all.append(average_gradient)

                entropy_image = entropy(image[n,:,:].flatten())
                entropy_all.append(entropy_image)

                patient_name.append(test_patient_name[i])
                label.append(test_label[i])
                path.append(test_patient_path[i])
        
        ssim_all = np.array(ssim_all, dtype='float')
        psnr_all = np.array(psnr_all, dtype='float')
        AG_all = np.array(AG_all, dtype='float')
        entropy_all = np.array(entropy_all, dtype='float')
        patient_name = np.array(patient_name, dtype='<U32')
        label = np.array(label, dtype='int')
        path = np.array(path, dtype='<U32')

        fname = ['patient_name', 'label', 'center', 'ssim', 'psnr', 'ag', 'entropy']
        t0 = np.column_stack((patient_name, label))
        t1 = np.column_stack((t0, path))
        t2 = np.column_stack((t1, ssim_all))
        t3 = np.column_stack((t2, psnr_all))
        t4 = np.column_stack((t3, AG_all))
        t5 = np.column_stack((t4, entropy_all))
        with open(modality[m]+'_Evaluation_Index.csv','w',newline='') as fi:
            writer = csv.writer(fi)
            writer.writerow(fname)
            for l in range(0, len(label)):
                writer.writerow(t5[l,:])
        fi.close()
 

if __name__ == '__main__':
    main()