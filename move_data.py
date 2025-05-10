import numpy as np
import shutil, os, xlrd

center = ['gulou', 'huaxi', 'shengrenmin']
for c in range(2, len(center)):
    readbook = xlrd.open_workbook(r'E:/lvpao/fusion_result2/'+center[c]+'.xls')
    sheet = readbook.sheet_by_name('Sheet1')
    nrows = sheet.nrows#行
    ncols = sheet.ncols#列
    patient_name_all = np.zeros((nrows-1,ncols),dtype='<U32')
    for i in range(1,nrows):
        for j in range(0,ncols):
            patient_name_all[i-1,j] = sheet.cell(i,j).value
        
    datapath = 'E:/lvpao/fusion_result2/'+center[c]
    patient_name = os.listdir(datapath)

    for i in range(0, len(patient_name)):
        for j in range(0, patient_name_all.shape[0]):
            if patient_name[i] == patient_name_all[j,0] and int(float(patient_name_all[j,2])) != 4:
                print(datapath+'/'+patient_name[i])
                print(i)
                if not os.path.exists('E:/lvpao1/fusion_result2/'+center[c]):
                    os.makedirs('E:/lvpao1/fusion_result2/'+center[c])
                shutil.copytree(datapath+'/'+patient_name[i], 'E:/lvpao1/fusion_result2/'+center[c]+'/'+patient_name[i])