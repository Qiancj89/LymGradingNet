import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from skimage import transform
from skimage.transform import radon
from skimage.transform import iradon

PET_sitk = sitk.ReadImage('E:/lvpao/data/gulou/ai guang hua/PET.nii.gz')
image_PET = sitk.GetArrayFromImage(PET_sitk)
print(image_PET.shape)

CT_sitk = sitk.ReadImage('E:/lvpao/data/gulou/ai guang hua/CT.nii.gz')
image_CT = sitk.GetArrayFromImage(CT_sitk)
image_CT = transform.resize(image_CT, (image_PET.shape[0], image_PET.shape[1], image_PET.shape[2]), order=0, preserve_range=True, mode='constant', anti_aliasing=True)

GT_sitk = sitk.ReadImage('E:/lvpao/data/gulou/ai guang hua/Segmentation.nii.gz')
image_GT = sitk.GetArrayFromImage(GT_sitk)

FUSION_sitk = sitk.ReadImage('E:/lvpao/fusion_result2/gulou/ai guang hua/FUSION_2_8.nii.gz')
FUSION = sitk.GetArrayFromImage(FUSION_sitk)

z_index, _, _ = np.where(image_GT>0)
'''
img_x = (image_CT-np.min(image_CT))/(np.max(image_CT)-np.min(image_CT))*255
img_x1 = np.max(img_x, axis = 2)
fig = plt.figure()
plt.imshow(img_x[z_index[0],:,:], cmap=plt.cm.Greys_r)
plt.axis('off')
'''
z_index = np.unique(z_index)
print(z_index)

for z in range(11, len(z_index)):
    print(z)
    FUSION_z = FUSION[z,:,:]
    FUSION_z = FUSION_z*255
    fig = plt.figure()
    plt.imshow(FUSION_z, cmap=plt.cm.seismic)
    plt.axis('off')



    image_z_PET = image_PET[z_index[z],:,:]
    image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_PET = image_z_PET[60:160,60:160]
    image_z_PET = transform.resize(image_z_PET, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_PET = (image_z_PET-np.min(image_z_PET))/(np.max(image_z_PET)-np.min(image_z_PET))*255
    

    image_z_CT = image_CT[z_index[z],:,:]
    image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_CT = image_z_CT[60:160,60:160]
    image_z_CT = transform.resize(image_z_CT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_CT = (image_z_CT-np.min(image_z_CT))/(np.max(image_z_CT)-np.min(image_z_CT))*255

    image_z_GT = image_GT[z_index[z],:,:]
    image_z_GT = transform.resize(image_z_GT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_GT = image_z_GT[60:160,60:160]
    image_z_GT = transform.resize(image_z_GT, (180, 180), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    image_z_GT[image_z_GT>0] = 1
    

    
    theta = np.linspace(0., 180., max(image_z_CT.shape), endpoint=False)
    sinogram_CT = radon(image_z_CT, theta=theta)
    sinogram_PET = radon(image_z_PET, theta=theta)
    sinogram_GT = radon(image_z_GT, theta=theta)
    sinogram = sinogram_CT+sinogram_PET
    #sinogram[sinogram<10000] = 0
    #sinogram[400:sinogram.shape[1],:] = 255           
    dx, dy = 0.5 * 180.0 / max(image_z_CT.shape), 0.5 / sinogram.shape[0]

    reconstruction_fbp = iradon(sinogram,theta=theta, filter_name='cosine')
    error_CT = reconstruction_fbp - image_z_CT
    error_PET = reconstruction_fbp - image_z_PET
    print(f'FBP rms reconstruction error_CT: {np.sqrt(np.mean(error_CT**2)):.3g}\n')
    print(f'FBP rms reconstruction error_PET: {np.sqrt(np.mean(error_PET**2)):.3g}\n')

    imkwargs = dict(vmin=-0.2, vmax=0.2)
  
    # 图像坐标转换为(theta,p)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,5,1)
    ax1.set_title("Original CT")
    ax1.imshow(image_z_CT, cmap=plt.cm.Greys_r)

    ax2 = fig.add_subplot(2,5,2)
    ax2.set_title("Original PET")
    ax2.imshow(image_z_PET, cmap=plt.cm.Greys_r)
    
    ax3 = fig.add_subplot(2,5,3)
    ax3.set_title("Reconstruction\nFiltered back projection")
    ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

    ax4 = fig.add_subplot(2,5,4)
    ax4.set_title("Adding")
    ax4.imshow(image_z_CT+image_z_PET, cmap=plt.cm.Greys_r)

    ax5 = fig.add_subplot(2,5,5)
    ax5.set_title("Ground Truth")
    ax5.imshow(image_z_GT, cmap=plt.cm.Greys_r)

    ax6 = fig.add_subplot(2,5,6)
    ax6.set_title("Original CT")
    ax6.imshow(image_z_CT, cmap=plt.cm.seismic)

    ax7 = fig.add_subplot(2,5,7)
    ax7.set_title("Original PET")
    ax7.imshow(image_z_PET, cmap=plt.cm.seismic)
    
    ax8 = fig.add_subplot(2,5,8)
    ax8.set_title("Reconstruction\nFiltered back projection")
    ax8.imshow(reconstruction_fbp, cmap=plt.cm.seismic)

    ax9 = fig.add_subplot(2,5,9)
    ax9.set_title("Adding")
    ax9.imshow(image_z_CT+image_z_PET, cmap=plt.cm.seismic)
    
    '''
    ax4 = fig.add_subplot(2,4,4)
    ax4.set_title("Reconstruction error_CT\nFiltered back projection")
    ax4.imshow(error_CT, cmap=plt.cm.Greys_r, **imkwargs)

    ax5 = fig.add_subplot(2,4,5)
    ax5.set_title("Reconstruction error_PET\nFiltered back projection")
    ax5.imshow(error_PET, cmap=plt.cm.Greys_r, **imkwargs)

    ax6 = fig.add_subplot(2,4,6)
    ax6.set_title("Ground Truth")
    ax6.imshow(image_z_GT, cmap=plt.cm.Greys_r)

    ax7 = fig.add_subplot(2,4,7)
    ax7.set_title("Radon transform\n(Sinogram)")
    ax7.set_xlabel("Projection angle (deg)")
    ax7.set_ylabel("Projection position (pixels)")
    ax7.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto')
    
    ax8 = fig.add_subplot(2,4,8)
    ax8.set_title("Adding")
    ax8.imshow(image_z_CT+image_z_PET, cmap=plt.cm.Greys_r)
    '''
    fig.tight_layout()
    plt.show()
    

    '''
    f_CT = np.fft.fft2(image_z_CT)
    f_PET = np.fft.fft2(image_z_PET)
    f = f_CT+f_PET
    fshift = np.fft.fftshift(f)
    #fshift[216:296,216:296] = 1
    res_img = np.log(np.abs(fshift))
    
    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    # 图像坐标转换为(theta,p)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,4,1)
    ax1.set_title("Original CT")
    ax1.imshow(image_z_CT, cmap=plt.cm.Greys_r)
    
    ax2 = fig.add_subplot(2,4,2)
    ax2.set_title("Original PET")
    ax2.imshow(image_z_PET, cmap=plt.cm.Greys_r)

    ax3 = fig.add_subplot(2,4,3)
    ax3.set_title("Fourier transform")
    ax3.imshow(res_img, cmap=plt.cm.Greys_r)
    
    ax4 = fig.add_subplot(2,4,4)
    ax4.set_title("Reconstruction")
    ax4.imshow(iimg, cmap=plt.cm.Greys_r)

    ax5 = fig.add_subplot(2,4,5)
    ax5.set_title("Reconstruction error_CT")
    ax5.imshow(iimg - image_z_CT, cmap=plt.cm.Greys_r)

    ax6 = fig.add_subplot(2,4,6)
    ax6.set_title("Reconstruction error_PET")
    ax6.imshow(iimg - image_z_PET, cmap=plt.cm.Greys_r)

    ax7 = fig.add_subplot(2,4,7)
    ax7.set_title("Ground Truth")
    ax7.imshow(image_z_GT, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    plt.show()
    '''