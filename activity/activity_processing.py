import numpy as np
from scipy import ndimage, signal, stats
from skimage import transform, measure
from tqdm import tqdm

from matplotlib import pylab as plt

from joblib import Parallel, delayed

def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)


def refine_image(IM,th=0.07):
    IMAGE = IM.astype(float)
    IMAGE = IMAGE-IMAGE.mean()
    IMAGE = IMAGE-ndimage.gaussian_filter(IMAGE,3)
    #IMAGE = exposure.adjust_gamma(IMAGE,1,2)

    minn,maxx = IMAGE.min()*th, IMAGE.max()*th
    IMAGE[IMAGE>maxx] = maxx
    IMAGE[IMAGE<0] = 0
    IMAGE = IMAGE/IMAGE.max()
    return IMAGE

def find_unique(MASK):
    mask = MASK.astype(int)
    unis = np.unique(mask[0])[1:]
    NEW_UNIS = []
    for i in tqdm(unis):
        n = 0
        for j in mask:
            if i in j:
                n+=1
            else:
                break
        if n == len(mask):
            NEW_UNIS.append(i)
    return np.asarray(NEW_UNIS)


def get_int_mask(mask,ints):
    """retrieves int mask from float mask"""
    unis  = np.unique(mask)[1:]
    unint = unis.astype(int)

    new_mask = np.zeros_like(mask)

    for i in ints:
        n  = np.argwhere(unint==i)[0][0]
        us = unis[n]
        ut = unint[n]
        new_mask[mask==us] = ut
    return new_mask

def get_int_masks(masks,ints):
    return np.asarray([get_int_mask(i,ints) for i in tqdm(masks)]).astype("uint8")

### Distance Maps

def get_dmap(mask):
    DMAP = np.zeros_like(mask,dtype="float")
    DMAP[mask>0] = 1
    return ndimage.distance_transform_edt(DMAP)

def get_dist_centers(mask):
    cells = np.unique(mask)[1:]
    mlist = []
    for i in tqdm(cells):
        cell = mask.copy()
        cell[cell!=i] = 0
        dist  = ndimage.distance_transform_edt(cell)
        ym,xm = np.argwhere(dist==dist.max()).mean(0)
        mlist.append([ym,xm,i])
    return np.asarray(mlist)

def get_all_dist_centers(mask,n_jobs=4):
    return np.asarray(mCPU(get_dist_centers,mask,n_jobs))

### Cell extraction

def max_frame_radius(mask,centers,ind):
    """Determin maximum cell radius for a single cell from a set of images

    Input: - mask, 3d numpy array (frame, y, x) with index numbers representing each cell
           - centers,

    Output: number (probably int maybe flaot ... not sure yet)
    """

    framesize = 0
    for n,i in enumerate(mask):
        cell    = np.argwhere(i==ind)
        cind    = np.argwhere(centers[n][:,-1].astype(int)==ind)[0][0]
        cy,cx,_ = centers[n][cind]
        y0,y1   = cell[:,0].min(),cell[:,0].max()
        x0,x1   = cell[:,1].min(),cell[:,1].max()
        dcxy    = [cy-y0,y1-cy,cx-x0,x1-cx]

        dd   = np.max(np.abs(dcxy))
        if dd > framesize:
            framesize = dd
    return framesize

def get_all_radii(mask, centers):
    """get maximum cell radius for all cells in the image
    """
    return np.asarray([max_frame_radius(mask,centers,i) for i in np.unique(mask)[1:]])

def roundup(num):
    """round up to 10, just for estetics

    takes a number -> returns a number"""

    unum = np.round(num,-1)
    if unum <= num:
        return int(unum+10)
    else:
        return int(unum)

def extract_rad_frames(mask,centers, radius, ind):
    """Extract frame from image/mask based on the center position,and radius
    Inputs: - mask    - 3d numpy array (frame, y, x) with index numbers representing each cell
            - centers -
            - radius  -
            - ind     - frame index
           """

    FRAMES = []
    CENTS  = []
    rad    = roundup(radius)
    d = int(2*rad+1)
    for n,i in enumerate(mask):
        #print(centers)
        #print(radius)
        #print(ind)

        cind    = np.argwhere(centers[n][:,-1].astype(int)==ind)[0][0]
        cy,cx,_ = centers[n][cind]
        cy,cx   = np.round(cy,0),np.round(cx,0)

        y0,x0 = int(cy-rad-1),int(cx-rad-1)
        if y0 < 0:
            y0 = 0
        if x0 < 0:
            x0 = 0
        f       = i.copy()
        frame   = f[y0:int(cy+rad),x0:int(cx+rad)]
        frame[frame != ind] = 0
        if cy-rad < 0:
            frame = np.vstack([np.zeros((int(rad-cy+1),frame.shape[1])),frame])
        if cx-rad < 0:
            frame = np.hstack([np.zeros((frame.shape[0],int(rad-cx+1))),frame])
        if cy+rad > i.shape[0]:
            frame = np.vstack([frame,np.zeros((int(cy-i.shape[0]+rad),frame.shape[1]))])
        if cx+rad > i.shape[1]:
            frame = np.hstack([frame,np.zeros((frame.shape[0],int(cx-i.shape[1]+rad)))])
        if frame.shape[1] == d-1:
            frame = np.hstack([frame,np.zeros_like(frame)[:,:2]])[:,:-1]
        if frame.shape[0] == d-1:
            frame = np.vstack([frame,np.zeros_like(frame)[:2]])[:-1]

        FRAMES.append(frame/ind)
    return np.asarray(FRAMES,dtype="bool")


### Polar transformations

def pol_trans(data2d):
    PT = transform.warp_polar(data2d,radius=data2d.shape[0]).astype("bool")[:,:].T
    return np.hstack([PT,PT[:,:20]])
    #return np.round(transform.rescale(rotim, (100,360),preserve_range=True),0)

def pol_outline(polim):
    #line = morphology.binary_erosion(polim,selem=morphology.disk(1))
    # line = polim.astype(int)-line.astype(int)
    #return np.argwhere(line==1)
    return measure.find_contours(polim, 0.9)[0]


def inner_circ(polin):
    Amin = polin[:,0].min()
    Tmin = polin[:,1][np.argwhere(polin[:,0]==Amin)[0,0]]
    if Tmin < 0:
        Tmin = Tmin+360.
    if Tmin > 360:
        Tmin = Tmin-360.
    return Amin, Tmin

def outer_max(polin):
    Amax = polin[:,0].max()
    Tmax = polin[:,1][np.argwhere(polin[:,0]==Amax)[0,0]]
    if Tmax < 0:
        Tmax = Tmax+360.
    if Tmax > 360:
        Tmax = Tmax-360.
    return Amax, Tmax

def get_toki(plin,flr=10.,dist=6,width=6):
    GA       = ndimage.gaussian_filter
    peaks, _ = signal.find_peaks(GA(plin[:,0],flr), distance=dist,width=width)
    return plin[peaks]

def gdy(phi,amp):
    return np.sin(phi/180.*np.pi)*amp

def gdx(phi,amp):
    return np.cos(phi/180.*np.pi)*amp

def gda(y,x):
    return np.sqrt(x**2+y**2)

def get_cart_toki(toki,center):
    cy,cx,n = center[0], center[1], center[2]
    Y     = gdy(toki[:,1],toki[:,0])+cy
    X     = gdx(toki[:,1],toki[:,0])+cx
    N     = [n]*len(X)
    return np.asarray([Y,X,N]).T


def derot_cells(masks,angles,pad=2):
    """Derotates Images based on angle of the initial image and width of the derotated image"""
    derots = []
    for i in range(len(masks)):
        nd    = int(np.sqrt(((masks[i].shape[0]//2)**2)*2)*2+1)
        derot = transform.rotate(masks[i],angles[i],
                                 preserve_range=True,
                                 center=[masks[i].shape[0]//2+1,masks[i].shape[1]//2+1],
                                 resize=True,
                                 clip  =False)
        derot = np.rot90(derot)
        derot = np.round(derot,0).astype(int)

        if derot.shape[0] < nd-1:
            ypadl = np.zeros((int(nd-derot.shape[0])//2, derot.shape[1]))
            ypadr = np.zeros((int(nd-derot.shape[0])//2, derot.shape[1]))
            derot = np.vstack([ypadl,derot,ypadr])

        if derot.shape[0] == nd:
            derot = derot[:-1]

        derots.append(derot)

    widths = []
    for i in derots:
        ones = np.argwhere(i==1)
        width= np.max(ones[:,1])-np.min(ones[:,1])
        widths.append(width)
    w = int(np.max(widths)+pad*2)
    new_derots = []
    for i in derots:
        cw = (i.shape[1]-w)//2
        ni = i[:,cw:-cw][:,:(w-1)]
        new_derots.append(ni)

    return np.asarray(new_derots)

def derotate_all_cells(CELLS,ANGLES0,pad=2):
    ANGLES = []
    for i in ANGLES:
        if i >= 180.:
            ANG = i-180.
        else:
            ANG = i
        ANGLES.append(ANG)
    ANGLES = np.asarray(ANGLES)

    CS     = CELLS.swapaxes(0,1)
    NC     = [derot_cells(CS[i],ANGLES[i], pad) for i in tqdm(range(len(CS)))]
    width  = int(np.max([i.shape[-1] for i in NC]))
    FRAMES = []
    for i in tqdm(NC):
        frame = i
        if i.shape[-1] < width-1:
            xpad  = np.zeros((frame.shape[0],frame.shape[1],int(width-frame.shape[-1])//2 ))
            frame = np.dstack([xpad,i,xpad])

        if frame.shape[-1] == width:
            frame = frame[:,:,:-1]
        FRAMES.append(frame)

    return np.asarray(FRAMES)

def derotate_all_cells_multi(CELLS,ANGLES,pad=2):
    CS = CELLS.swapaxes(0,1)
    def do_shit(i):
        return derot_cells(CS[i],ANGLES[i], pad)
    NC = mCPU(do_shit,range(len(CS)),40)
    return NC


def realign_rotcells(CELLS):
    """3D -> 2D array
       Reshapes array with (D,Y,X) into (Y,X*D)"""
    return CELLS.swapaxes(1,0).reshape((CELLS.shape[1],CELLS.shape[2]*CELLS.shape[0]))

def realign_rotcells4D(CELLS):
    """3D -> 2D array
       Reshapes array with (D,Y,X) into (Y,X*D)"""
    return CELLS.swapaxes(2,1).reshape((CELLS.shape[0],CELLS.shape[2],CELLS.shape[3]*CELLS.shape[1]))

def get_hight_width(MASK):
    """get hight and width of cell in single cell mask"""
    Y,X = np.argwhere(MASK==1).T
    H   = Y.max()-Y.min()
    W   = X.max()-X.min()
    return H,W


###### propagation properties

def recenter_points(p0, pad=0.1):
    points = p0.copy()
    maxx = points.max(0)
    minn = points.min(0)
    cm   = points.mean(0)
    dcm  = cm-np.max(np.hstack([cm-minn,maxx-cm]))
    cp   = points-dcm
    cmax = cp.max()
    dm   = cmax*pad
    return dm+cp, cmax+2*dm

def get_gauss_kernel(data,d = 20, size=100):
    y,x       = data.T
    dm = np.linspace(0,d,size)
    #xx, yy    = np.mgrid[min(x):max(x):size, min(y):max(y):size]
    yy,xx     = np.meshgrid(dm,dm)
    positions = np.vstack([yy.ravel(), xx.ravel()])
    values    = np.vstack([y,x])
    kernel    = stats.gaussian_kde(values)
    f =  np.reshape(kernel(positions).T, xx.shape)
    return f/f.max()

def pol_trans_d(data2d):
    PI = transform.warp_polar(data2d,radius=data2d.shape[0]).T
    return np.hstack([PI,PI[:,:(data2d.shape[0]-360)]])

def pol_outline_d(polim,th=0.9):
    #line = morphology.binary_erosion(polim,selem=morphology.disk(1))
    # line = polim.astype(int)-line.astype(int)
    #return np.argwhere(line==1)
    return measure.find_contours(polim, th)[0]

def get_rot_cont(ROT,thresh=0.05):
    ROTC  = np.zeros_like(ROT)
    ROTC[ROT>thresh] = 1
    return ROTC

def get_arrows(ry,rx):
    abig   = [np.max(ry),rx[np.argmax(ry)]]
    asmall = [np.min(ry),rx[np.argmin(ry)]]
    return [asmall,abig]

def get_persistence(centers,cellnr,imsize = 420):
    """Compute directional persistence.
       takes list of numpy arrays containing cell vertixes with shape: (Y,X,cell index)

       input: List of numpy 2D arrays
       out : number
    """
    try:
        GA1      = ndimage.gaussian_filter1d
        uc0m, sh = recenter_points(centers[:,cellnr,:-1])
        de0m     = get_gauss_kernel(uc0m, sh, imsize)

        de0R    = pol_trans_d(de0m)
        de0ROT  = get_rot_cont(de0R)
        RY,RX   = pol_outline_d(de0ROT).T
        RS, RB  = get_arrows(RY,RX) ### zeta1, zeta2 ### lenght, angle,
        RT      = RS[0]/RB[0]
    except:
        RT = 0.0 #### 0 means persistence could not be detected
    return RT
