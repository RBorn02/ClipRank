import torch
import torchvision
from torchvision import transforms as ts
from torch.utils.data import DataLoader
from clip import tokenize

import random
import os
import math
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,args, batch_first=False):
        self.batch_first = batch_first
        self.ranking = args.ranking
    
    def __call__(self,batch):
        imgs = []
        caps = []
        for item in batch:
            img, txts = item
            txt = random.choice(txts)
            imgs.append(img.unsqueeze(0))
            caps.append(txt)
            
        imgs = torch.cat(imgs,dim=0)
        targets = tokenize(caps)
        return imgs, targets


def get_contrastive_dataloader(args, preprocess_fn):
    data_path = args.data_train
    train_path = os.path.join(data_path, 'train2014/train2014/')
    train_ann_path = os.path.join(data_path, 'annotations_trainval2014/annotations/captions_train2014.json')
    val_path = os.path.join(data_path, 'val2014/val2014/')
    val_ann_path = os.path.join(data_path, 'annotations_trainval2014/annotations/captions_val2014.json')


    train_set = torchvision.datasets.CocoCaptions(root=train_path, annFile=train_ann_path,
                                                  transform=preprocess_fn)
    val_set = torchvision.datasets.CocoCaptions(root=val_path, annFile=val_ann_path,
                                                  transform=preprocess_fn)
    
    train_set, val_set = create_coco_subsets(train_set, val_set)
    print(len(train_set))

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=CapsCollate(args, batch_first=True))
    
    val_loader = DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=CapsCollate(args, batch_first=True))

    ###Implement Test Loader
    return train_loader, val_loader

class RankingLoader:
    def __init__(self):
        pass

    def get_sim(self, n_px, h, w, rb, rc, rs, rh, p):
        sim = math.log(1 + (h/n_px * w/n_px) * (1/(abs(rb) + abs(rc) 
                         + abs(rs) + abs(rh) + 0.00001)) * 1/p)
        return sim
    
    def get_transforms(self, n_transforms, n_px):
        transform_dict = {}
        for n in range(n_transforms):
            
            h = np.random.randint(int(0.8*n_px), n_px)
            w = np.random.randint(int(0.8*n_px), n_px)
            rb = np.random.uniform(-1, 1)
            rc = np.random.uniform(-1, 1)
            rs = np.random.uniform(-1, 1)
            rh = np.random.uniform(-0.5, 0.5)
            p = np.random.uniform(0.01, 1)
            
            transform = ts.Compose([
                ts.Resize((n_px, n_px), interpolation=BICUBIC),
                ts.RandomResizedCrop((h, w)),
                ts.RandomPerspective(p, 1.0),
                CustomColorJitter(rb, rc, rs, rh),
                ts.Resize((n_px, n_px), interpolation=BICUBIC),
                self._convert_image_to_rgb,
                ts.ToTensor(),
                ts.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
            sim = self.get_sim(n_px, h, w, rb, rc, rs, rh, p)
            
            transform_dict[transform] = sim
        sorted_transforms = {k: v for k, v in sorted(transform_dict.items(), key=lambda item: item[1])}
        
        view_list = list(sorted_transforms.keys())
        sorted_list = list(reversed(view_list))
        return sorted_list
    
    def get_ranking_loader(self, args, n_px):
        transform_list = self.get_transforms(args.num_trans, n_px)
        base_transform = ts.Compose([
            ts.Resize((n_px, n_px), interpolation=BICUBIC),
            ts.CenterCrop((n_px, n_px)),
            self._convert_image_to_rgb,
            ts.ToTensor(),
            ts.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        data_path = args.data_train
        train_path = os.path.join(data_path, 'train2014/train2014/')
        train_ann_path = os.path.join(data_path, 'annotations_trainval2014/annotations/captions_train2014.json')
        val_path = os.path.join(data_path, 'val2014/val2014/')
        val_ann_path = os.path.join(data_path, 'annotations_trainval2014/annotations/captions_val2014.json')

        train_set = torchvision.datasets.CocoCaptions(root=train_path, annFile=train_ann_path,
                                                  transform=N_Transform(transform_list, base_transform))
        val_set = torchvision.datasets.CocoCaptions(root=val_path, annFile=val_ann_path,
                                                  transform=N_Transform(transform_list, base_transform))
        
        train_set, _ = create_coco_subsets(train_set, val_set)

        ranking_train_loader = DataLoader(train_set, batch_size=args.ranking_batch, 
                                    num_workers=args.workers, collate_fn=CapsCollate(args, batch_first=True), shuffle=True, pin_memory=False)

        return ranking_train_loader
    
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")


 


class CustomColorJitter:
    def __init__(self, rb, rc, rs, rh):
        self.rb = rb
        self.rc = rc
        self.rs = rs
        self.rh = rh
        
    def __call__(self, x):
        x = ts.functional.adjust_brightness(x, self.rb+1.0)
        x = ts.functional.adjust_contrast(x, self.rc+1.0)
        x = ts.functional.adjust_saturation(x, self.rs+1.0)
        return ts.functional.adjust_hue(x, self.rh) 


class N_Transform:
    def __init__(self, transform_list, base_transform):
        self.transform_list = transform_list
        self.base_transform = base_transform
        
    def __call__(self, x):
        view_list = []
        view_list.append(self.base_transform(x))
        for transform in self.transform_list:
            view = transform(x)
            view_list.append(view)
        return torch.cat(view_list)
   
def create_coco_subsets(train_set, val_set):
    idx = np.arange(len(val_set))
    idx_pos = np.random.choice(idx, 30000)
    neg_idx = np.setdiff1d(idx, idx_pos, assume_unique=True)

    test_idx = np.random.choice(neg_idx, 5000)
    rest_idx = np.setdiff1d(neg_idx, test_idx, assume_unique=True)
    val_idx = np.random.choice(rest_idx, 5000)

    train_subset = Subset(val_set, idx_pos)
    val_subset = Subset(val_set, val_idx)

    total_train_set = ConcatDataset([train_set, train_subset])
    return total_train_set, val_subset
