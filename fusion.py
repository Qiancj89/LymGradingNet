import nibabel as nib
import cv2
import SimpleITK as sitk
import numpy as np
import os
 
# 归一化
def MaxMinNormalizer(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_normalize = (data - data_min) / (data_max - data_min)
    return data_normalize
 
# 融合CT数据与PET数据
def Fusion(root, path_CT, path_PET):
    image_CT = nib.load(path_CT).get_data()
    image_CT = image_CT.astype(float)
    image_CT_size = np.shape(image_CT)
    # 归一化 CT
    image_CT = MaxMinNormalizer(image_CT)
 
    # 重采样后的PET全身图像
    image_PET = nib.load(path_PET).get_data()
    image_PET = image_PET.astype(float)
    image_PET[image_PET<0] = 0
    # 归一化 PET
    image_PET = MaxMinNormalizer(image_PET)
 
 
    #  融合比例
    percent_list = []
    for i in range(1, 10):
        percent_list.append(i / 10)
    # fusion CT , PET,
    # percent_list=[0.1] # best performance
    for percent in percent_list:
        ImageFusion = np.zeros(image_CT_size)
        for num in range(image_CT_size[2]):
            image_CT_slice = image_CT[:, :, num]
            image_PET_slice = image_PET[:, :, num]
            # 按照比例融合数据
            img_mix = cv2.addWeighted(image_CT_slice, percent, image_PET_slice, 1 - percent, 0)
            ImageFusion[:, :, num] = img_mix
        ImageFusion = np.transpose(ImageFusion, (2, 1, 0))
        ImageFusionISO = sitk.GetImageFromArray(ImageFusion, isVector=False)
 
        # 获取图像扫描及空间信息
        input_PET_property = sitk.ReadImage(path_PET)
        spacing = np.array(input_PET_property.GetSpacing())
        direction = np.array(input_PET_property.GetDirection())
        Origin = np.array(input_PET_property.GetOrigin())
        
        # 重新把融合后的数据设置图像信息
        ImageFusionISO.SetSpacing(spacing)
        ImageFusionISO.SetOrigin(Origin)
        ImageFusionISO.SetDirection(direction)
 
        # 保存重采样后的数据
        soffix_name = path_PET[-7:]
        # savepath_name = file_name + '_modified_fusioin_0.6' + soffix_name
        # savepath_name = file_name + 'CTPETFusioin_LINEAR_'+ str(percent) + soffix_name
        savepath_name = root + 'CT_resampledPET_Fusioin_' + str(percent) + soffix_name
 
        sitk.WriteImage(ImageFusionISO, savepath_name)
    
    pass
 
def main():
    patient_name_path = "../data/"
    patient_name = os.listdir(patient_name_path)
    name_list = ['CT', 'resampled_PET']
    for i in range(0,len(patient_name)):
        root = patient_name_path+patient_name[i]+'/images/'
    
        # CT全身图像数据路径
        input_CT = root + name_list[0] + '.nii.gz'
        # PET全身图像数据路径
        input_PET = root + name_list[1] + '.nii.gz'
        
        # 融合CT与PET
        Fusion(root, input_CT, input_PET)
 
 
 
if __name__ == '__main__':
    main()