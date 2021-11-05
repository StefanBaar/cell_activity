import numpy as np

from scipy import spatial
from scipy import ndimage
from scipy.interpolate import interp1d

from skimage.morphology import disk,binary_dilation
from skimage.segmentation import watershed

from tqdm import tqdm

from activity.segmentation import get_segments,new_size_filter

from matplotlib import pylab as plt

### direct areqa overlap tracking:
import multiprocessing
cpus = multiprocessing.cpu_count()

import warnings
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed

def mCPU(func, var, n_jobs=20,verbose=1):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def to_bin(DATA):
    D = DATA.copy()
    D[D>0] = 1
    return D


def get_multip(a):
    """get indeces with more than one incidence"""
    u , ind, counts =  np.unique(a, return_index=True, return_counts=True)
    return u[counts>1]

def get_merger_pairs(CLOG):
    """ get all kernels with multiple indices"""
    A,B     = np.array(CLOG).T
    AU      = get_multip(A)
    KERNELS = [B[np.argwhere(A==i)] for i in AU]
    return KERNELS



def track_next(C1,C2):
    """compare two masks and return merging islands"""
    CELLS   = np.unique(C1)[1:]
    C2B     = to_bin(C2)
    C3      = np.zeros_like(C1)
    CM_LOG  = []
    for i in CELLS[::-1]:
        MASK    = (C1==i) & (C2B==1)
        CM      = np.unique(C2[MASK])
        for j in CM:
            CM_LOG.append([j,i])
            C3[C2==j] = i
    try:
        MERGERS=get_merger_pairs(CM_LOG)
    except:
        MERGERS=[]
        print("NO MERGERS")
    C3B   = C2B - to_bin(C3)
    C4B,_ = get_segments(C3B)
    C4B   = C4B + np.max([C1,C2])
    C4B[C4B==C4B.min()] = 0
    return MERGERS, C3+C4B



def replace_double_ind(SEGM, IND=4):
    MOLD            = SEGM.copy()
    MOLD[MOLD!=IND] = 0
    MOLD[MOLD==IND] = 1
    MOLD_SEGS , _   = get_segments(MOLD)
    NAMES = []
    if MOLD_SEGS.max() > 1:
        for i in np.unique(MOLD_SEGS)[1:]:
            Mask = np.where(MOLD_SEGS==i)

            if str(IND)[-1] == "0":
                NR=float(str(IND)[:-1]+str(i))
            else:
                NR=float(str(IND)+str(i))
            SEGM[Mask] = NR
    return SEGM

def check_double_ind(SEGS_IM0):
    SEGS_IM  = SEGS_IM0.astype(float)
    SUB_SEGS = np.unique(SEGS_IM)[1:]
    for i in SUB_SEGS:
        NEW_SEGS_IM = replace_double_ind(SEGS_IM,i)
    return NEW_SEGS_IM


def track_merger_splits(DATA):
    TRACKS   = [DATA[0]]
    MERGERS  = [[i for i in np.unique(DATA[0])[1:]]]
    for i in range(1,len(DATA)):
        M, T = track_next(TRACKS[-1],DATA[i])
        MERGERS.append(M)
        TRACKS.append(check_double_ind(T))
    #print(np.unique(TRACKS[-1]))
    MERGERS[-1] = [i for i in np.unique(TRACKS[-1])[1:]]
    return MERGERS, np.asarray(TRACKS)


def relable_cells(LABELS):
    UNIS = np.unique(LABELS)
    NEW_LABELS = np.zeros_like(LABELS,dtype="int")
    for i in range(len(UNIS)):
        NEW_LABELS[LABELS==UNIS[i]] = i
    return NEW_LABELS


def track_mergers(im1,im2):
    NM, NT = track_next(im1,im2)
    for i in NM:
        if len(i) > 0:
            NT = get_watershed(i,im1,NT)
    return NT

def get_watershed(nm,im1,im2,mrad=30):
    image1 = np.zeros_like(im1)
    image2 = image1.copy()

    coords = []
    ### copy images and extract merging cells as well as their kernels
    for i in nm:
        ### get cells
        image2[np.where(im2==i[0])] = 1
        image1[np.where(im1==i[0])] = 1
        ### get kernels
        zs = np.zeros_like(image1)
        zs[np.where(im1==i[0])] = 1
        maxc = np.argwhere(zs==zs.max()).mean(0).astype(int)
        coords.append(maxc)

    coords = np.asarray(coords)

    ### produce distance maps
    distance1  = ndimage.distance_transform_edt(image1)
    distance2  = ndimage.distance_transform_edt(image2)

    mask       = np.zeros(distance1.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    #### watershed based on distance computation takes too long
    cdist      = np.sqrt(np.sum(np.diff(coords,axis=0)**2))
    mrad       = cdist/10.
    #mask       = binary_dilation(mask,disk(mrad))

    markers, _ = ndimage.label(mask)
    label      = watershed(-distance1, markers, mask=image2,compactness=mrad)

    fimage = im2.copy()
    for n,i in enumerate(np.unique(label)[1:]):
        ### we need to check that the correct island is
        ### transfered from frame 1 to frame 2
        ### we check that each island in label is closed to its counter part in frame one
        zla           = np.zeros_like(label)
        zla[label==i] = 1
        zcent         = np.argwhere(zla==1).mean(0)
        dist = np.abs(np.sqrt(np.sum(zcent**2))-np.sqrt(np.sum(coords**2,axis=1)))
        indd = np.argmin(dist)
        fimage[label==i] = nm[indd][0]
    return fimage.astype(int)

def track_mergers(im1,im2):
    NM, NT = track_next(im1,im2)
    for i in NM:
        if len(i) > 0:
            NT = get_watershed(i,im1,NT)
    return NT

def check_double(MASK):
    """assign new index to splitting cells"""
    UM       = np.unique(MASK).astype("int")
    NEW_MASK = MASK.copy()
    for i in UM:
        N          = np.zeros_like(MASK)
        N[MASK==i] = 1
        MULTS      = get_segments(N)[0]
        if MULTS.max() > 1:
            for j in np.unique(MULTS)[2:]:
                NEW_MASK[MULTS==j] = j+MASK.max()-1
    return NEW_MASK

def track_clustering(DATA,minsize=600):
    TRACKS   = [DATA[0]]
    for i in tqdm(range(1,len(DATA))):
        T    = track_mergers(TRACKS[-1],DATA[i])
        TC   = check_double(T)
        CI,AR= np.unique(TC,return_counts=True)
        TC   = new_size_filter(TC,fsize=minsize)
        TRACKS.append(TC)
    return np.asarray(TRACKS, dtype=int)

def num_2_one(DATA0):
    DATA = DATA0.copy()
    DATA[DATA>0] = 1
    return DATA
