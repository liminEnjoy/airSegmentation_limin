import os
import shutil
import numpy as np
from config_training import config
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
import warnings

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip


#############################################################################

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing = [1,1,1],order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id,annos,filelist,data_path,prep_folder):  

    name = filelist[id]
    label = annos[annos[:,0]==int(name)]
    label = label[:,[3,2,1,4,5]].astype('float')
    
    patient = os.path.join(data_path,name+'.mhd')
    sliceim,origin,spacing,isflip  = load_itk_image(patient)
    

    sliceim[np.isnan(sliceim)]=-2000
    sliceim = lumTrans(sliceim)
#     shape = sliceim.shape
    
#     if shape[0]<100:
#         pad = ((0, 96-shape[0]), (0, 0), (0, 0))
#         sliceim = np.pad(sliceim,pad,'constant',constant_values =170)

    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    
    if len(label)==0:
        label = np.array([[0,0,0,0]])
        print(name)
    np.save(os.path.join(prep_folder,name+'_label.npy'), label)

    ctanno = []

    ctanno.append(spacing)
    ctanno.append(origin)
    ctanno.append(np.array([0,0,0]))

    np.save(os.path.join(prep_folder,name+'_ctanno.npy'), ctanno)

    #print(name)
    
def full_prep(prep_folder, data_path, alllabelfiles):
    warnings.filterwarnings("ignore")

    finished_flag = '.flag_preptianchi1'
    
    if not os.path.exists(finished_flag):

        alllabel = np.array(pandas.read_csv(alllabelfiles))
        
        filelist=[]
        for name in os.listdir(data_path):
            if name.endswith('.mhd'):
                filelist.append(name.split('.')[0])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        
        print('starting preprocessing')
        pool = Pool(100)
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder)

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
        # f= open(finished_flag,"w+")   
    
# prep_folder = '/home/xgz/main/DSB_3/data2/stage1/preprocess'
# data_path = '/home/xgz/main/DSB_3/data2/trainmhd/'
# alllabelfiles = '/home/xgz/main/DSB_3/data2/csv/train/annotations.csv'

# full_prep(prep_folder, data_path, alllabelfiles)

# prep_folder = '/home/xgz/main/DSB_3/data2/stage1/preprocess'
# data_path = '/home/xgz/main/DSB_3/data2/valmhd/'
# alllabelfiles = '/home/xgz/main/DSB_3/data2/csv/val/annotations.csv'
prep_folder = '/data/public_data/tianchi2/preprocess'
data_path = '/data/public_data/tianchi2/data/'
alllabelfiles = '/data/public_data/tianchi2/chestCT_round1_annotation_new.csv'

full_prep(prep_folder, data_path, alllabelfiles)