
from pathlib import Path

from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage import morphology, io
from PIL import Image, ImageOps
from tqdm import tqdm

from joblib import Parallel, delayed

import multiprocessing
cpus = multiprocessing.cpu_count()
#print("Nr. of CPU cores: ", cpus)

class MyDataset(Dataset):
    def __init__(self, data_path,image_path="images/",mask_path_1="cells/",mask_path_2="contas/", pytorch=True):
        super().__init__()
        self.image_files  = sorted(glob(data_path+image_path +"*.png"))
        self.mask_files_1 = sorted(glob(data_path+mask_path_1+"*.png"))
        self.mask_files_2 = sorted(glob(data_path+mask_path_2+"*.png"))
        self.pytorch      = pytorch


    def get_Grey(self, idx):
        return io.imread(self.image_files[idx])[:,:,0]

    def load_image(self, idx, invert=True):
        """load image from path """
        raw_grey = self.get_Grey(idx)
        norm = (raw_grey / np.iinfo(raw_grey.dtype).max)
        return torch.unsqueeze(torch.tensor(norm, dtype=torch.float32),0)

    def load_mask(self, idx, invert=False):
        """load mask from path """
        MASK_1 = io.imread(self.mask_files_1[idx])[:,:,1]
        MASK_2 = io.imread(self.mask_files_2[idx])[:,:,2]
        MASK   = np.zeros((2,MASK_1.shape[0],MASK_1.shape[1]))
        MASK[0][MASK_1>0] = 1
        MASK[1][MASK_2>0] = 2
        #mask = np.asarray(MASK)
        return torch.tensor(np.sum(MASK,0), dtype=torch.int64)

    def __getitem__(self, idx):
        image      = self.load_image(idx,invert=self.pytorch)
        mask       = self.load_mask (idx)
        return image, mask

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s


class MyDataset_NPY(Dataset):
    def __init__(self, file_path="training_data/NPYS/object_wise/512/", pytorch=True):
        super().__init__()
        self.image_files  = sorted(glob(file_path +"*.npy"))
        self.pytorch      = pytorch

    def load_file(self,idx):
        FILE  = np.load(self.image_files[idx])
        cells = FILE[:,:,0]
        conts = FILE[:,:,1]
        image = FILE[:,:,2]

        mask           = cells+(2*conts)

        image = torch.unsqueeze(torch.tensor(image, dtype=torch.float32),0)
        mask  = torch.tensor(mask, dtype=torch.int64)

        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_file(idx)
        return image, mask

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s


class MyDataset_GRID(Dataset):
    def __init__(self, file_path="training_data/NPYS/object_wise/512/", pytorch=True):
        super().__init__()
        self.image_files  = sorted(glob(file_path +"*.npy"))
        self.pytorch      = pytorch

    def load_file(self,idx):
        FILE  = np.load(self.image_files[idx])
        image = FILE[:,:,0]
        mask  = FILE[:,:,1]

        image = torch.unsqueeze(torch.tensor(image, dtype=torch.float32),0)
        mask  = torch.tensor(mask, dtype=torch.int64)

        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_file(idx)
        return image, mask

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s



class CreateMulti(Dataset):
    def __init__(self,in_path,out_path,image_path="images/", mask_path_1="cells/", mask_path_2="contas/", nr_of_ims=200):
        self.image_files = sorted(glob(in_path+image_path  +"*.png"))
        self.mask1_files = sorted(glob(in_path+mask_path_1 +"*.png"))
        self.mask2_files = sorted(glob(in_path+mask_path_2 +"*.png"))

        op               = self.dic_name(out_path)
        self.out_ims     = op+image_path
        Path(op+image_path).mkdir(parents=True, exist_ok=True)
        self.out_mas_1   = op+mask_path_1
        Path(op+mask_path_1).mkdir(parents=True, exist_ok=True)
        self.out_mas_2   = op+mask_path_2
        Path(op+mask_path_2).mkdir(parents=True, exist_ok=True)
        self.nr_of_ims   = nr_of_ims

    def dic_name(self,outpath, LEN=3):
        files = sorted(glob(outpath+"*/"))
        if len(files) == 0:
            name=str(10**(LEN+1)+1)[1:]
        else:
            nr  = len(files)+1
            name= str(10**(LEN+1)+nr)[1:]

        return outpath+name+"/"

    def transform_ims(self, image, mask_1, mask_2):


        # Random Rotation
        median = np.median(np.asarray(image))
        angle  = np.random.rand()*2.
        image  = TF.rotate(image ,angle,resample=Image.BICUBIC,fill=int(median))
        mask_1 = TF.rotate(mask_1,angle,resample=Image.BICUBIC,fill=int(0))
        mask_2 = TF.rotate(mask_2,angle,resample=Image.BICUBIC,fill=int(0))

        # Random change histogram
        if np.random.random() > 0.8:
            image = ImageOps.autocontrast(image,cutoff=np.random.rand()*0.01)

        # Random crop or not
        if np.random.random() > 0.1:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,scale=(0.1,2.), ratio=(1,1))
            image = TF.crop(image, i, j, h, w)
            mask_1= TF.crop(mask_1, i, j, h, w)
            mask_2= TF.crop(mask_2, i, j, h, w)

        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask_1= TF.hflip(mask_1)
            mask_2= TF.hflip(mask_2)

        # Random vertical flipping
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            mask_1= TF.vflip(mask_1)
            mask_2= TF.vflip(mask_2)

        # Resize
        resize = transforms.Resize(size=(512, 512))
        image  = resize(image)
        mask_1 = resize(mask_1)
        mask_2 = resize(mask_2)
        return image, mask_1, mask_2

    def synthesis_images(self):
        n = 1
        im_nr = len(self.image_files)
        for ims in range(im_nr):
            for inds in tqdm(range(self.nr_of_ims)):
                image = Image.open(self.image_files[ims])
                mask_1= Image.open(self.mask1_files[ims])
                mask_2= Image.open(self.mask2_files[ims])
                x,y,z = self.transform_ims(image, mask_1,mask_2)
                x.save(  self.out_ims+str(int(10**7 +n))[1:]+".png")
                y.save(self.out_mas_1+str(int(10**7 +n))[1:]+".png")
                z.save(self.out_mas_2+str(int(10**7 +n))[1:]+".png")
                n += 1

def load_train_val(DATASET, validation_ratio = 0.25,batch_size=12):
    DLEN  = int(DATASET.__len__())
    VALEN = int(DATASET.__len__()*validation_ratio)
    ratio = (DLEN-VALEN,VALEN)

    train_data, valid_data = torch.utils.data.random_split(DATASET, ratio)

    DATA_TRAIN = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)
    DATA_VALID = DataLoader(valid_data, batch_size=batch_size, shuffle=True,num_workers=0)

    return DATA_TRAIN, DATA_VALID

##################################################################################

def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def get_image_stack_P(FLIST,verbose=10):

    def load_image(path):
        return io.imread(path)

    STACK = mCPU(load_image,FLIST,verbose=verbose)

    return np.asarray(STACK)


#def transfrom_data()
