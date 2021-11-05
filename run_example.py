#### This script performes the entire activity analysis fo
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

import skimage
from skimage import io

from activity import models
from activity import activity_processing as ap
from activity import inference, tracking

from joblib import Parallel, delayed

import multiprocessing
cpus = multiprocessing.cpu_count()

print("-----------------------------------")
print("numpy version: ",np.__version__)
print("torch version: ",torch.__version__)
print("skimage version: ",skimage.__version__)
print("-----------------------------------")
print("Nr. of CPU cores: ",cpus)
print("-----------------------------------")

def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def get_cpus(data):
    if len(data) < cpus:
        return len(data)
    else:
        return cpus

def get_all_func(FUNC,ARRAY,n_jobs):
    def do_func(A):
        return [FUNC(i) for i in A]
    return mCPU(do_func,ARRAY,n_jobs)


def get_all_func_2(FUNC,A1,A2):
    def do_func(n):
        return [FUNC(A1[n][i],A2[n][i]) for i in range(len(A1[n]))]
    return [do_func(j) for j in tqdm(range(len(A1)))]


if __name__ == '__main__':
    from glob import glob

    image_path = "data/images/"
    image_paths= sorted(glob(image_path+"*"))

    print("-------------------------------------")
    print("Input images:")
    for i in image_paths:
        print(i)
    print("-------------------------------------")
    model_path = "data/models/UNET_weight_state.pt"
    npy_path   = "data/npy/"
    out_path   = "data/out/"


    Path(npy_path).mkdir(parents=True, exist_ok=True)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    size_filter= 600
    hole_size  = 4

    print("Load models:")
    model      = models.UNET(1,4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print("inference:")
    def process_tracks(path):
        image      =  io.imread(path)[:,:,0] ### for grey-scale images
        cells      =  inference.cell_inference(image, model, size_filter, hole_size)
        return [image, cells]

    out_arr           = np.asarray(mCPU(process_tracks,image_paths, get_cpus(image_paths)))
    images, cell_mask = np.swapaxes(out_arr,0,1)

    print("Tracking cells:")
    #cell_tracks       = tracking.track_clustering(cell_mask[::-1])
    #cell_tracks[-1]   = tracking.relable_cells(cell_tracks[-1])
    #cell_tracks       = tracking.track_clustering(cell_tracks[::-1])
    M, cell_tracks_float = tracking.track_merger_splits(cell_mask)
    uniques              = ap.find_unique(cell_tracks_float )
    cell_tracks          = ap.get_int_masks(cell_tracks_float,uniques)



    print("Cell activity estimation")
    print("Computing cell vertices")
    cell_vertices = ap.get_all_dist_centers(cell_tracks,get_cpus(cell_tracks))
    np.save(npy_path+"AI_ucents"       ,cell_vertices)

    print("Computing individual cell FOV parameters")
    radius        = ap.get_all_radii(cell_tracks, cell_vertices).max()

    print("Computing persistence")
    RTS           =  np.asarray([ap.get_persistence(cell_vertices,i) for i in tqdm(range(cell_vertices.shape[1]))]) #### needs error estimation

    print("Computing polar transformations")
    cells      = np.asarray([ap.extract_rad_frames(cell_tracks, cell_vertices,int(radius),i) for i in tqdm(uniques)],dtype="bool")
    polts      = np.asarray(get_all_func(ap.pol_trans,cells,get_cpus(cells))).astype(int)
    polins     = np.asarray(get_all_func(ap.pol_outline,polts,get_cpus(polts)))
    ris,ais    = np.asarray(get_all_func(ap.inner_circ,polins,get_cpus(polins))).T
    ros,aos    = np.asarray(get_all_func(ap.outer_max,polins,get_cpus(cells))).T
    LEN_RATIOS = ris/ros
    poltokis   = np.asarray(get_all_func(ap.get_toki,polins,get_cpus(cells)))
    cartokis   = np.asarray(get_all_func_2(ap.get_cart_toki,poltokis.T,cell_vertices)).T

    np.save(npy_path+"images"          ,images)
    np.save(npy_path+"cell_masks_float",cell_tracks_float)
    np.save(npy_path+"cell_masks"      ,cell_tracks)
    np.save(npy_path+"cell_crops"      ,cells)
    np.save(npy_path+"AI_tokis"        ,cartokis)
    np.save(npy_path+"AI_cell_lw_rats,",LEN_RATIOS)
    np.save(npy_path+"AI_persistence," ,RTS)
