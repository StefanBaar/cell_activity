### inference
import numpy as np
import torch
from activity.segmentation import get_segmentation, size_filter_SEGS,get_segments, rem_holes, mCPU

def cpu_inference_grey(IMAGE, model):
    IMAGE = (IMAGE / np.iinfo(IMAGE.dtype).max)
    IMAGE = torch.tensor(IMAGE, dtype=torch.float32)

    mask  = model(IMAGE.unsqueeze(0).unsqueeze(0))
    p     = torch.functional.F.softmax(mask[0], 0)
    return np.asarray(p.argmax(0).cpu())

def probability_cpu(IMAGE, model):
    IMAGE = (IMAGE / np.iinfo(IMAGE.dtype).max)
    IMAGE = torch.tensor(IMAGE, dtype=torch.float32)
    mask  = model(IMAGE.unsqueeze(0).unsqueeze(0))
    return torch.functional.F.softmax(mask[0], 0).detach().numpy()

def get_cells(mask):
    cells = np.zeros_like(mask)
    cells[mask == 1 ] = 1
    cells[mask == 3 ] = 1
    return cells

def cell_inference(image,model,cell_min_size=800, hole_size=4):
    #### AI detection
    AImask       = np.asarray(cpu_inference_grey(image,model))
    AIcells      = rem_holes(get_cells(AImask),cell_min_size,hole_size)
    AIcells,area = get_segments(AIcells)
    AIcells      = size_filter_SEGS([AIcells],[area],0, FILTER=cell_min_size)
    return np.asarray(AIcells, dtype="int")
