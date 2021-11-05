from tqdm import tqdm

import numpy as np
from glob import glob
from pathlib import Path
from skimage import io, transform
from PIL import Image, ImageOps
from psd_tools import PSDImage
from skimage import io, exposure, morphology

def thfilter(image,th=0.):
    mask = np.zeros_like(image)
    mask[image>th] = 1
    return mask

def get_outline(data,radius=3):
    disk = morphology.disk(radius)
    outlines = np.zeros_like(data[0])
    for i in range(4):
        outlines += data[1+i].astype(int)-morphology.erosion(data[1+i].astype(int),disk)
    outlines[outlines > 1] = 1
    return outlines

def psd_to_npy(path):
    psd      = PSDImage.open(path)
    image = np.asarray(psd[0].composite(psd.viewbox))[:,:,0].astype(float)
    image = image-image.min()
    image = image/image.max()

    conta = np.asarray(psd[3].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
    conta = thfilter(conta)
    cells1= np.asarray(psd[4].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
    cells1= thfilter(cells1)
    if len(psd) > 5:
        cells2= np.asarray(psd[5].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
        cells2= thfilter(cells2)
    else:
        cells2= np.zeros_like(cells1).astype("uint8")

    if len(psd) > 6:
        cells3= np.asarray(psd[6].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
        cells3= thfilter(cells3)
    else:
        cells3= np.zeros_like(cells1).astype("uint8")
    all_layers = np.asarray([image,conta,cells1,cells2,cells3])
    outlines   = get_outline(all_layers)
    return np.asarray([image,conta,cells1,cells2,cells3,outlines])


def psd_to_cell(path):
    psd      = PSDImage.open(path)
    image = np.asarray(psd[0].composite(psd.viewbox))[:,:,0].astype(float)
    image = image-image.min()
    image = image/image.max()

    conta = np.asarray(psd[3].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
    conta = thfilter(conta)
    cells1= np.asarray(psd[4].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
    cells1= thfilter(cells1)
    if len(psd) > 5:
        cells2 = np.asarray(psd[5].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
        cells2 = thfilter(cells2)
    else:
        cells2 = np.zeros_like(cells1).astype("uint8")

    if len(psd) > 6:
        cells3 = np.asarray(psd[6].composite(psd.viewbox))[:,:,-1].astype("uint8")/255
        cells3 = thfilter(cells3)
    else:
        cells3 = np.zeros_like(cells1).astype("uint8")
    return np.asarray([image,conta,cells1,cells2,cells3])



def npy_to_mask_levels(data):
    mask  = data[1]*2
    mask += data[2]*3
    mask[mask>4] = 4

    diff = data[3]+data[4]
    diff[diff>1] = -1
    diff[diff>0] = 0
    mask += data[3]*2
    mask += (data[4]+diff)*4
    #mask[data[-1]==1] = 1
    mask[mask>8] = 8
    return mask.astype(int)

def npy_to_mask(data):
    mask  = data[1]*2
    mask += data[2]*3
    mask[mask>4] = 4
    mask[data[-1]==1] = 1
    return mask.astype(int)

def get_crop(FRAME,INDEX,thresh=30):
    y,x   = np.argwhere(FRAME == INDEX).T
    dy    = int(y.max())-int(y.min())
    dx    = int(x.max())-int(x.min())
    yc    = (dy)/2+int(y.min())
    xc    = (dx)/2+int(x.min())
    d     = np.max([dx,dy])/2
    y0    = int(yc-d-thresh)
    x0    = int(xc-d-thresh)

    if yc+d+thresh > FRAME.shape[0]:
        y0 = int(y0+FRAME.shape[0]-(yc+d+thresh))

    if xc+d+thresh > FRAME.shape[1]:
        x0 = int(x0+FRAME.shape[1]-(xc+d+thresh))

    if y0 < 0:
        y0 = int(0)
    if x0 < 0:
        x0 = int(0)
    y1    = int(y0+2*(d+thresh))
    x1    = int(x0+2*(d+thresh))

    return y0,y1,x0,x1


def resi2D(data, sh=128):
    d0 = transform.resize(data[:,:,0],(sh,sh))
    d1 = np.round(transform.resize(data[:,:,1],(sh,sh)))

    return np.dstack([d0,d1])
