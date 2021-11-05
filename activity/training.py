from glob import glob

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from kornia import morphology as kmorph
from skimage import morphology
import time
from IPython.display import clear_output
import tqdm
from tqdm import tqdm
import numpy as np
from pathlib import Path

from activity.data_prep import MyDataset_GRID, load_train_val
from activity.models import UNET, large_UNET, super_large_UNET

### check: https://www.kaggle.com/mlagunas/naive-unet-with-pytorch-tensorboard-logging

def to_np(x):
    return x.data.cpu().numpy()

def dic_name(outpath, LEN=9):
    files = sorted(glob(outpath+"*/"))
    if len(files) == 0:
        name=str(10**(LEN+1)+1)[1:]
    else:
        nr  = len(files)+1
        name= str(10**(LEN+1)+nr)[1:]

    return outpath+name+"/"

# define a class to log values during training
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)
    def forward(self, logits, targets):
        p = F.sigmoid(logits)
        pf = p.view(-1)
        tf = targets.view(-1)
        return self.bce_loss(pf, tf)

def dic_name(outpath, LEN=9):
    files = sorted(glob(outpath+"*/"))
    if len(files) == 0:
        name=str(10**(LEN+1)+1)[1:]
    else:
        nr  = len(files)+1
        name= str(10**(LEN+1)+nr)[1:]

    return outpath+name+"/"


### for masks containing no outline
def get_torch_outline(maskset):
    outline = torch.zeros_like(maskset)
    outline[maskset>2 ] = 1
    outline[maskset==1] = 0
    return outline

### for masks without outline
def create_torch_weightmap(OUTLINE, D=3):
    DISK = torch.tensor(morphology.disk(D))
    FILL = kmorph.dilation(OUTLINE.unsqueeze(0),DISK)
    FILL = kmorph.erosion(FILL,DISK)
    OUT  = FILL[0]-OUTLINE
    OUT[OUT<0] = 0
    return OUT

def create_wmap(data):
    OUTLINE = get_torch_outline(data)
    WMAP    = create_torch_weightmap(OUTLINE)
    return WMAP

### for masks containing outline in postions
def apply_torch_weightmap(TORCH,TMASK,D=3):
    WMAP        = create_wmap(TMASK)
    WD          = TORCH.max(3)[0].max(2)[0].max(1)[0]-TORCH.min(3)[0].min(2)[0].min(1)[0]*10.
    OFFSET      = torch.zeros_like(TORCH).cpu()
    OFFSET[:,1] = -(WMAP.T*WD.cpu()).T
    OFFSET[:,3] =  (WMAP.T*WD.cpu()).T
    TORCH2      = TORCH+OFFSET.to(TORCH.device)
    return TORCH2


def train(model, train_dl, valid_dl, device, optimizer, acc_fn, epochs=1,writer=None, wlog="weight_log/"):
    """
    model training without nudging the cell edges
    """
    start  = time.time()
    #loss_fn= BCELoss2d()
    loss_fn= nn.CrossEntropyLoss()
    losses = AverageMeter()
    train_loss, valid_loss = [], []
    acc_val = []

    best_acc = 0.0

    wlog = dic_name(wlog)
    Path(wlog).mkdir(parents=True, exist_ok=True)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            # iterate over data
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (j,k) in pbar:
                x = j.to(device)
                y = k.to(device)


                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    #### softmax required by BCS loss function
                    #outputs = torch.functional.F.softmax(outputs, 1).argmax(1)
                    loss = loss_fn(outputs, y.long())
                    losses.update(loss.data, x.size(0))
                    #writer.add_scalar("Loss/train", loss, epoch)
                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.requres_grad = True
                    loss.backward()
                    optimizer.step()

                    if writer != None:
                        #writer.add_graph(model, outputs)
                        # log loss values every iteration
                        writer.add_scalar('data/(train)loss_val', losses.val, i + 1)
                        writer.add_scalar('data/(train)loss_avg', losses.avg, i + 1)
                        # log the layers and layers gradient histogram and distributions
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('model/(train)' + tag, to_np(value), i + 1)
                            writer.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)
                        # log the outputs given by the model (The segmentation)
                        #writer.add_image('model/(train)output', make_grid(outputs.data), i + 1)

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        #outputs = torch.functional.F.softmax(outputs, 1).argmax(1)
                        loss = loss_fn(outputs, y.long())
                        losses.update(loss.data, x.size(0))
                        if writer != None:
                            writer.add_scalar('data/(test)loss_val', losses.val, i + 1)
                            writer.add_scalar('data/(test)loss_avg', losses.avg, i + 1)
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                writer.add_histogram('model/(test)' + tag, to_np(value), i + 1)
                                writer.add_histogram('model/(test)' + tag + '/grad', to_np(value.grad), i + 1)
                        #writer.add_scalar("Loss/valid", loss, epoch)

                acc = acc_fn(outputs, y, device)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size

                if i % 10 == 0:
                    # clear_output(wait=True)
                    tqdm.write('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(i, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    #print(torch.cuda.memory_summary())
                    ### Multi GPU use: model.module.state_dict(),
                    torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+"_"+str(10+i)[1:]+".pt")
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(float(epoch_loss)) if phase=='train' else valid_loss.append(float(epoch_loss))
            acc_val.append(float(epoch_acc))

        torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+".pt")

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss, acc_val, writer

def edge_train(model, train_dl, valid_dl, device, optimizer, acc_fn, epochs=1,writer=None, wlog="weight_log/"):
    """
    model training with nudging of cell edges
    """
    start  = time.time()
    #loss_fn= BCELoss2d()
    loss_fn= nn.CrossEntropyLoss()
    losses = AverageMeter()
    train_loss, valid_loss = [], []
    acc_val = []

    best_acc = 0.0

    wlog = dic_name(wlog)
    Path(wlog).mkdir(parents=True, exist_ok=True)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            # iterate over data
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (j,k) in pbar:
                x = j.to(device)
                y = k.to(device)


                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    #print(outputs.shape)
                    ### Force finding Gaps
                    outputs = apply_torch_weightmap(outputs,k)

                    #### softmax required by BCS loss function
                    #outputs = torch.functional.F.softmax(outputs, 1).argmax(1)
                    loss = loss_fn(outputs, y.long())
                    losses.update(loss.data, x.size(0))
                    #writer.add_scalar("Loss/train", loss, epoch)
                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.requres_grad = True
                    loss.backward()
                    optimizer.step()

                    if writer != None:
                        #writer.add_graph(model, outputs)
                        # log loss values every iteration
                        writer.add_scalar('data/(train)loss_val', losses.val, i + 1)
                        writer.add_scalar('data/(train)loss_avg', losses.avg, i + 1)
                        # log the layers and layers gradient histogram and distributions
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('model/(train)' + tag, to_np(value), i + 1)
                            writer.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)
                        # log the outputs given by the model (The segmentation)
                        #writer.add_image('model/(train)output', make_grid(outputs.data), i + 1)

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        #outputs = torch.functional.F.softmax(outputs, 1).argmax(1)
                        loss = loss_fn(outputs, y.long())
                        losses.update(loss.data, x.size(0))
                        if writer != None:
                            writer.add_scalar('data/(test)loss_val', losses.val, i + 1)
                            writer.add_scalar('data/(test)loss_avg', losses.avg, i + 1)
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                writer.add_histogram('model/(test)' + tag, to_np(value), i + 1)
                                writer.add_histogram('model/(test)' + tag + '/grad', to_np(value.grad), i + 1)
                        #writer.add_scalar("Loss/valid", loss, epoch)

                acc = acc_fn(outputs, y, device)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size

                if i % 10 == 0:
                    # clear_output(wait=True)
                    tqdm.write('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(i, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    #print(torch.cuda.memory_summary())
                    ### Multi GPU use: model.module.state_dict(),
                    torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+"_"+str(10+i)[1:]+".pt")
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(float(epoch_loss)) if phase=='train' else valid_loss.append(float(epoch_loss))
            acc_val.append(float(epoch_acc))

        torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+".pt")

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss, acc_val, writer


if __name__ == '__main__':
    vname        = "V1/"
    path         = "data/training/"
    weight_path  = path+"versions/"+vname+"weights/"
    sweight_path = path+"versions/"+vname+"sweights/"
    log_path     = path+"versions/"+vname+"logs/"
    Path(weight_path).mkdir(parents=True, exist_ok=True)

    dset                   = MyDataset_GRID(path+"window320/")
    train_data, valid_data = load_train_val(dset, validation_ratio = 0.8,batch_size=39)

    unet_path  = "data/"
    GPU        = 1
    device     = torch.device("cuda:"+str(GPU))
    unet_model = UNET(1,4).cuda(GPU)
    unet_model.load_state_dict(torch.load(unet_path,map_location=device))

    lr = 0.001

    n = len(glob(weight_path+"*"))+1
    print(n)
    logdir = dic_name(log_path)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)

    def acc_metric(predb, yb, device):
        return (predb.argmax(dim=1) == yb.to(device)).float().mean()

    opt = torch.optim.Adam(unet_model.parameters(), lr=lr)
    #tloss,vloss,acc,writer = edge_train(unet_model, train_data, valid_data, device, opt, acc_fn=acc_metric, epochs=100,writer=writer,wlog=sweight_path)
    tloss,vloss,acc,writer = train(unet_model, train_data, valid_data, device, opt, acc_fn=acc_metric, epochs=100,writer=writer,wlog=sweight_path)
    torch.save(unet_model.state_dict(), weight_path+"Adam_weight_"+str(10000+int(n))[1:]+"_"+str(lr))
    np.save(weight_path+"Adam_tloss_"+str(10000+int(n))[1:]+"_"+str(lr),tloss)
    np.save(weight_path+"Adam_vloss_"+str(10000+int(n))[1:]+"_"+str(lr),vloss)
    np.save(weight_path+"Adam_acc_"+str(10000+int(n))[1:]+"_"+str(lr),acc)
