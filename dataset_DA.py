import os

import cv2
import numpy as np
import torch
import torch.utils.data

# source_train_ids source_img_dir source_mask_dir img_ext num_classes transform
# target_train_ids target_img_dir


class Dataset_DA(torch.utils.data.Dataset):
    def __init__(self, source_train_ids, source_img_dir, source_mask_dir, target_train_ids, target_img_dir, img_ext, mask_ext, num_classes, transform=None):

        self.source_train_ids = source_train_ids
        self.source_img_dir = source_img_dir
        self.source_mask_dir = source_mask_dir
        self.target_train_ids = target_train_ids
        self.target_img_dir = target_img_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        # 扩展一下target train集的长度
        self.target_train_ids = self.target_train_ids * int(np.ceil(len(self.source_train_ids)/len(self.target_train_ids)))

    def __len__(self):
        return len(self.source_train_ids)

    def __getitem__(self, idx):
        source_train_id = self.source_train_ids[idx]
        target_train_id = self.target_train_ids[idx]

        source_img = cv2.imread(os.path.join(self.source_img_dir,
                                             source_train_id + self.img_ext))
        source_mask = []
        for i in range(self.num_classes):
            source_mask.append(cv2.imread(os.path.join(self.source_mask_dir,
                                                       source_train_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        source_mask = np.dstack(source_mask)

        target_train_img = cv2.imread(os.path.join(self.target_img_dir,
                                                   target_train_id + self.img_ext))

        if self.transform is not None:
            augmented = self.transform(image=source_img, mask=source_mask)
            source_img = augmented['image']
            source_mask = augmented['mask']
            target_train_img = self.transform(image=target_train_img)['image']

        source_img = source_img.astype('float32') / 255
        source_img = source_img.transpose(2, 0, 1)
        source_mask = source_mask.astype('float32') / 255
        source_mask = source_mask.transpose(2, 0, 1)
        target_train_img = target_train_img.astype('float32') / 255
        target_train_img = target_train_img.transpose(2, 0, 1)

        return source_img, source_mask, target_train_img, {'source_train_id': source_train_id, 'target_train_id': target_train_id}


class TestDataset_DA(torch.utils.data.Dataset):
    def __init__(self, target_val_ids, target_img_dir, target_mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.target_val_ids = target_val_ids
        self.target_img_dir = target_img_dir
        self.target_mask_dir = target_mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.target_val_ids)

    def __getitem__(self, idx):
        target_val_id = self.target_val_ids[idx]

        target_val_img = cv2.imread(os.path.join(self.target_img_dir,
                                                 target_val_id + self.img_ext))

        target_val_mask = []
        for i in range(self.num_classes):
            target_val_mask.append(cv2.imread(os.path.join(self.target_mask_dir,
                                                           target_val_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        target_val_mask = np.dstack(target_val_mask)

        if self.transform is not None:
            augmented = self.transform(
                image=target_val_img, mask=target_val_mask)
            target_val_img = augmented['image']
            target_val_mask = augmented['mask']

        target_val_img = target_val_img.astype('float32') / 255
        target_val_img = target_val_img.transpose(2, 0, 1)
        target_val_mask = target_val_mask.astype('float32') / 255
        target_val_mask = target_val_mask.transpose(2, 0, 1)

        return target_val_img, target_val_mask, {'target_val_id': target_val_id}


class Dataset_DA_pseuLabel(torch.utils.data.Dataset):

    def __init__(self, source_train_ids, source_img_dir, source_mask_dir, target_train_ids, target_img_dir, target_pseuLabel_dir, img_ext, mask_ext, num_classes, transform=None):

        self.source_train_ids = source_train_ids
        self.source_img_dir = source_img_dir
        self.source_mask_dir = source_mask_dir
        self.target_train_ids = target_train_ids
        self.target_img_dir = target_img_dir
        self.target_pseuLabel_dir = target_pseuLabel_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        # 扩展一下target train集的长度
        self.target_train_ids = self.target_train_ids * \
            int(np.ceil(len(self.source_train_ids)/len(self.target_train_ids)))

    def __len__(self):
        return len(self.source_train_ids)

    def __getitem__(self, idx):
        source_train_id = self.source_train_ids[idx]
        target_train_id = self.target_train_ids[idx]

        source_img = cv2.imread(os.path.join(self.source_img_dir,
                                             source_train_id + self.img_ext))
        source_mask = []
        for i in range(self.num_classes):
            source_mask.append(cv2.imread(os.path.join(self.source_mask_dir,
                                                       source_train_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        source_mask = np.dstack(source_mask)

        target_train_img = cv2.imread(os.path.join(self.target_img_dir,
                                                   target_train_id + self.img_ext))

        target_pseuLabel = []
        for i in range(self.num_classes):
            target_pseuLabel.append(cv2.imread(os.path.join(self.target_pseuLabel_dir,
                                                            target_train_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        target_pseuLabel = np.dstack(target_pseuLabel)

        if self.transform is not None:
            augmented = self.transform(image=source_img, mask=source_mask)
            source_img = augmented['image']
            source_mask = augmented['mask']

            augmented_target = self.transform(
                image=target_train_img, mask=target_pseuLabel)
            target_train_img = augmented_target['image']
            target_pseuLabel = augmented_target['mask']

        source_img = source_img.astype('float32') / 255
        source_img = source_img.transpose(2, 0, 1)
        source_mask = source_mask.astype('float32') / 255
        source_mask = source_mask.transpose(2, 0, 1)
        target_train_img = target_train_img.astype('float32') / 255
        target_train_img = target_train_img.transpose(2, 0, 1)
        target_pseuLabel = target_pseuLabel.astype('float32') / 255
        target_pseuLabel = target_pseuLabel.transpose(2, 0, 1)

        return source_img, source_mask, target_train_img, target_pseuLabel, {'source_train_id': source_train_id, 'target_train_id': target_train_id}

# add meaningless labels