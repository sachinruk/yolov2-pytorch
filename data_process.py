import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pickle
import numpy as np
import cv2
from copy import deepcopy
# Data Augmentation
# Transforms classes
# Random scale
# Flip
# x, y reposition
class RandomCrop(object):
    
    def imcv2_affine_trans(self, im):
        # Scale and translate
        h, w, c = im.shape
        scale = np.random.uniform() / 10. + 1.
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        im = cv2.resize(im, (0,0), fx = scale, fy = scale)
        im = im[offy : (offy + h), offx : (offx + w)]

        return im, [w, h], [scale, [offx, offy]]

    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        result = self.imcv2_affine_trans(image)
        image, dims, trans_param = result
        scale, offs = trans_param
        
        offs = np.array(offs*2)
        dims = np.array(dims*2)
        bboxes = deepcopy(bboxes)
        bboxes[:, 1:] = np.array(bboxes[:, 1:]*scale - offs, np.int64)
        bboxes[:, 1:] = np.maximum(np.minimum(bboxes[:, 1:], dims), 0)
        
        check_errors = (((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1))
        if sum(check_errors) > 0:
            bool_mask = ~ check_errors
            bboxes = bboxes[bool_mask]
        return {"image": image, "bboxes": bboxes}

class RandomFlip(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

        bboxes = deepcopy(bboxes)
        flip = np.random.binomial(1, .5)
        if flip: 
            w = image.shape[1]
            image = cv2.flip(image, 1)
            backup_min = deepcopy(bboxes[:, 1])
            bboxes[:, 1] = w - bboxes[:, 3]
            bboxes[:, 3] = w - backup_min
        
        if sum(((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1)) > 0:
            print ("random flip")
        
        return {"image": image, "bboxes": bboxes}

class Rescale(object):
    
    def __init__(self, output):
        self.new_h, self.new_w = output
        
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        h, w, c = image.shape
        new_h = int(self.new_h)
        new_w = int(self.new_w)
        image = cv2.resize(image, (new_w, new_h))
        
        bboxes = deepcopy(bboxes)
        bboxes = np.array(bboxes, np.float64)
        bboxes[:, 1] *= new_w*1.0/w
        bboxes[:, 2] *= new_h*1.0/h
        bboxes[:, 3] *= new_w*1.0/w
        bboxes[:, 4] *= new_h*1.0/h
        if sum(((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1)) > 0:
            print ("random scale", bboxes, sample['bboxes'], new_w, new_h, w, h)

        return {"image": image, "bboxes": bboxes}

class TransformBoxCoords(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        height, width, _ = image.shape
        
        bboxes = deepcopy(bboxes)
        bboxes = np.array(bboxes, np.float64)
        x = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
        y = 0.5 * (bboxes[:, 2] + bboxes[:, 4])
        w = 1. * (bboxes[:, 3] - bboxes[:, 1])
        h = 1. * (bboxes[:, 4] - bboxes[:, 2])
        if sum(((w <= 0) | (h <= 0) | (x <= 0) | (y <= 0)) & (bboxes[:, 0]!=-1))>0:
            print ("up", bboxes, sample["bboxes"])
        bboxes[:, 1] = x/width
        bboxes[:, 2] = y/height
        bboxes[:, 3] = w/width
        bboxes[:, 4] = h/height
        if sum(((bboxes[:, 1] <0) | (bboxes[:, 2]<0) | (bboxes[:, 3]<=0) | (bboxes[:, 4]<=0)) & (bboxes[:, 0]!=-1)) > 0:
            print ("random transform box coords")

        
        return {"image": image, "bboxes": bboxes}

class Normalize(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        image = np.array(image, np.float64)
        image /= 255.0
        return {"image": image, "bboxes": bboxes}

class EliminateSmallBoxes(object):
    
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        bool_mask = ((bboxes[: , 3] > self.thresh) & (bboxes[: , 4] > self.thresh))
        bboxes = bboxes[bool_mask]
        return {"image": image, "bboxes": bboxes}


class ToTensor(object):

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if len(bboxes) == 0:
            return {'image': torch.from_numpy(image), 'bboxes': torch.FloatTensor()}
        return {'image': torch.from_numpy(image), 'bboxes': torch.from_numpy(bboxes)}
    
class VOCDataset(Dataset):
    
    def __init__(self, file, sample=-1, transform=None, files = [], 
                 max_truth=30):
        
        """
            Reading all image names without the extension.
        """
        with open(file, 'rb') as f:
            self.images, self.bboxes = pickle.load(f)
        idx = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in idx]
        self.bboxes = [np.array(self.bboxes[i]) for i in idx]
#         print(self.bboxes[0].dtype)
        if sample != -1:
            self.images = self.images[:sample]
            self.bboxes = self.bboxes[:sample]
        self.transform = transform
        self.max_truth = max_truth
    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        datum = {"image": self.images[idx], "bboxes": self.bboxes[idx]}
            
        if self.transform:
            datum = self.transform(datum)
            
        bboxes = datum['bboxes']
        n_true = len(bboxes)
        if len(bboxes) > self.max_truth:
            bboxes = bboxes[:self.max_truth]
            n_true = self.max_truth
        else:
            zero_fill = self.max_truth - len(bboxes)
            null_pad = -1 * (torch.ones((zero_fill, 5), dtype=torch.double))
            if n_true == 0:
                bboxes = null_pad
            else:
#                 from IPython.core.debugger import Tracer; Tracer()()
#                 print([bboxes.dtype, null_pad.dtype])
                bboxes = torch.cat([bboxes, null_pad])
#                 bboxes = torch.cat([torch.from_numpy(bboxes).float(), 
#                                     null_pad])
        
        datum['bboxes'] = bboxes
        datum['n_true'] = n_true
        return datum
    
def get_data(image_size=416, sample=-1, batch_size=64, dataset='train'):
    transform_fn = transforms.Compose([RandomCrop(), 
                                   RandomFlip(), 
                                   Rescale((image_size, image_size)), 
                                   TransformBoxCoords(), 
                                   Normalize(),
                                   EliminateSmallBoxes(0.025),
                                   ToTensor()])
    
    if dataset == 'train':
        voc_train = VOCDataset("./data/train1000.pickle", 
                               sample=sample, 
                               transform=transform_fn)

        train_loader = DataLoader(voc_train, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=0)
        return train_loader, len(voc_train)
    else:
        voc_test = VOCDataset("./data/test.pickle", 
                              sample=sample, 
                              transform=transform_fn)
        test_loader = DataLoader(voc_test_12, 
                                 batch_size=batch_size, 
                                 num_workers=4)
        return test_loader, len(voc_test)