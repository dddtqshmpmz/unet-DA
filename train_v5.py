import argparse
import os
from collections import OrderedDict
from glob import glob
import cv2

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import segmentation_models.segmentation_models_pytorch as smp
from discriminator import FCDiscriminator
import torch.nn.functional as F

import archs
import losses
from dataset import Dataset
from dataset_DA import Dataset_DA, TestDataset_DA, Dataset_DA_pseuLabel
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=320, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=320, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    # mmd_loss
    parser.add_argument('--mmd_loss', default='MMD_loss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: MMD_loss)')

    # dataset
    parser.add_argument('--dataset', default='patient_datasetRandom',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # optimizer_D
    parser.add_argument('--optimizer_D', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr_D', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum_D', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay_D', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov_D', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    # scheduler_D
    parser.add_argument('--scheduler_D', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr_D', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor_D', default=0.1, type=float)
    parser.add_argument('--patience_D', default=2, type=int)
    parser.add_argument('--milestones_D', default='1,2', type=str)
    parser.add_argument('--gamma_D', default=2/3, type=float)
    parser.add_argument('--early_stopping_D', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    # lambda_adv_target
    parser.add_argument('--lambda_adv_target', default=0.05, type=float)
    parser.add_argument('--lambda_target_train_pseu_loss',
                        default=0.005, type=float)
    parser.add_argument('--lambda_mmd_loss', default=0.01, type=float)

    parser.add_argument('--source_train_num', default=12000, type=int)
    parser.add_argument('--target_train_num', default=4000, type=int)

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, model_D, criterion, criterion_D, criterion_mmd, optimizer, optimizer_D):
    avg_meters = {'source_train_loss': AverageMeter(),
                  'source_train_dice': AverageMeter(),
                  'loss_D': AverageMeter(),
                  'loss_adv': AverageMeter(),
                  'loss_pseu': AverageMeter(),
                  'loss_mmd': AverageMeter()}

    model.train()
    model_D.train()
    optimizer.zero_grad()
    optimizer_D.zero_grad()

    interp_source = nn.Upsample(
        size=(config['input_h'], config['input_w']), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        config['input_h'], config['input_w']), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    pbar = tqdm(total=len(train_loader))
    for source_img, source_mask, target_train_img, target_pseuLabel, _ in train_loader:
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        loss_D_value = 0
        loss_adv_value = 0
        loss_pseu_value = 0
        loss_mmd_value = 0

        source_img = source_img.cuda()
        source_mask = source_mask.cuda()
        target_train_img = target_train_img.cuda()
        target_pseuLabel = target_pseuLabel.cuda()

        # train G
        for param in model_D.parameters():
            param.requires_grad = False
        # train with source
        source_pred = model(source_img)
        # source_pred = interp_source(source_pred)

        # source segment loss
        source_train_loss = criterion(source_pred, source_mask)
        source_train_dice = dice_coef(source_pred, source_mask)
        loss = source_train_loss
        loss.backward(retain_graph=True)

        # train with target
        # target train pseu loss
        target_train_pred = model(target_train_img)
        target_train_pseu_loss = criterion(target_train_pred, target_pseuLabel)
        loss_pseu = config['lambda_target_train_pseu_loss'] * \
            target_train_pseu_loss
        loss_pseu_value = target_train_pseu_loss.item()
        loss_pseu.backward(retain_graph=True)

        # adv loss
        # target_train_pred = interp_target(target_train_pred)
        D_out = model_D(torch.sigmoid(target_train_pred))
        loss_adv_target = criterion_D(D_out, torch.FloatTensor(
            D_out.data.size()).fill_(source_label).cuda())

        loss = config['lambda_adv_target'] * loss_adv_target
        loss_adv_value = loss_adv_target.item()
        loss.backward(retain_graph=True)

        # mmd_loss
        loss_mmd = criterion_mmd(source_pred, target_train_pred)
        loss_mmd_value = loss_mmd.item()
        loss_mmd = config['lambda_mmd_loss'] * loss_mmd
        loss_mmd.backward()

        # train D
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        source_pred = source_pred.detach()
        D_out = model_D(torch.sigmoid(source_pred))

        loss_D = criterion_D(D_out, torch.FloatTensor(
            D_out.data.size()).fill_(source_label).cuda())
        loss_D = loss_D / 2
        loss_D_value += loss_D.item()
        loss_D.backward()

        # train with target
        target_train_pred = target_train_pred.detach()
        D_out = model_D(torch.sigmoid(target_train_pred))

        loss_D = criterion_D(D_out, torch.FloatTensor(
            D_out.data.size()).fill_(target_label).cuda())
        loss_D = loss_D / 2
        loss_D_value += loss_D.item()
        loss_D.backward()

        # do optimizing step
        optimizer.step()
        optimizer_D.step()

        avg_meters['source_train_loss'].update(
            source_train_loss.item(), source_img.size(0))
        avg_meters['source_train_dice'].update(
            source_train_dice.item(), source_img.size(0))
        avg_meters['loss_D'].update(
            loss_D_value, source_img.size(0))
        avg_meters['loss_adv'].update(
            loss_adv_value, source_img.size(0))
        avg_meters['loss_pseu'].update(
            loss_pseu_value, source_img.size(0))
        avg_meters['loss_mmd'].update(
            loss_mmd_value, source_img.size(0))

        postfix = OrderedDict([
            ('source_train_loss', avg_meters['source_train_loss'].avg),
            ('source_train_dice', avg_meters['source_train_dice'].avg),
            ('loss_pseu', avg_meters['loss_pseu'].avg),
            ('loss_D', avg_meters['loss_D'].avg),
            ('loss_adv', avg_meters['loss_adv'].avg),
            ('loss_mmd', avg_meters['loss_mmd'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return OrderedDict([('source_train_loss', avg_meters['source_train_loss'].avg), ('source_train_dice', avg_meters['source_train_dice'].avg),
                        ('loss_D', avg_meters['loss_D'].avg), ('loss_adv', avg_meters['loss_adv'].avg), (
                            'loss_pseu', avg_meters['loss_pseu'].avg),
                        ('loss_mmd', avg_meters['loss_mmd'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'target_val_loss': AverageMeter(),
                  'target_val_dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for target_val_img, target_val_mask, meta in val_loader:

            target_val_img = target_val_img.cuda()
            target_val_mask = target_val_mask.cuda()

            # compute output
            target_val_pred = model(target_val_img)
            target_val_loss = criterion(target_val_pred, target_val_mask)
            target_val_dice = dice_coef(target_val_pred, target_val_mask)

            # output the pred mask pics
            pred_target = torch.sigmoid(target_val_pred)
            zero = torch.zeros_like(pred_target)
            one = torch.ones_like(pred_target)
            pred_target = torch.where(pred_target > 0.5, one, zero)
            pred_target = pred_target.data.cpu().numpy()
            for i in range(len(pred_target)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('inputs', config['dataset'], 'pred_target_1k_patches', meta['target_val_id'][i] + '.png'),
                                (pred_target[i, c] * 255).astype('uint8'))

            avg_meters['target_val_loss'].update(
                target_val_loss.item(), target_val_img.size(0))
            avg_meters['target_val_dice'].update(
                target_val_dice, target_val_img.size(0))

            postfix = OrderedDict([
                ('target_val_loss', avg_meters['target_val_loss'].avg),
                ('target_val_dice', avg_meters['target_val_dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

    return OrderedDict([('target_val_loss', avg_meters['target_val_loss'].avg),
                        ('target_val_dice', avg_meters['target_val_dice'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    # ??????D???loss????????????  nn.BCEWithLogitsLoss().cuda() or nn.MSELoss().cuda()
    criterion_D = nn.MSELoss().cuda()
    criterion_mmd = losses.__dict__[config['mmd_loss']]().cuda()

    cudnn.benchmark = True  # ??????????????????cuDNN???auto-tuner????????????????????????????????????????????????

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()
    model_D = FCDiscriminator(num_classes=config['num_classes']).cuda()

    model = model.cuda()
    model_D = model_D.cuda()
    # model= nn.DataParallel(model,device_ids=[0,1,2])
    # summary(model, (3, 320, 320))

    # params
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # params_D
    params_D = filter(lambda p: p.requires_grad, model_D.parameters())
    if config['optimizer_D'] == 'Adam':
        optimizer_D = optim.Adam(
            params_D, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer_D'] == 'SGD':
        optimizer_D = optim.SGD(params_D, lr=config['lr_D'], momentum=config['momentum_D'],
                                nesterov=config['nesterov_D'], weight_decay=config['weight_decay_D'])
    else:
        raise NotImplementedError

    # scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(
            e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # scheduler_D
    if config['scheduler_D'] == 'CosineAnnealingLR':
        scheduler_D = lr_scheduler.CosineAnnealingLR(
            optimizer_D, T_max=config['epochs'], eta_min=config['min_lr_D'])
    elif config['scheduler_D'] == 'ReduceLROnPlateau':
        scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=config['factor_D'], patience=config['patience_D'],
                                                     verbose=1, min_lr=config['min_lr_D'])
    elif config['scheduler_D'] == 'MultiStepLR':
        scheduler_D = lr_scheduler.MultiStepLR(optimizer_D, milestones=[int(
            e) for e in config['milestones_D'].split(',')], gamma=config['gamma_D'])
    elif config['scheduler_D'] == 'ConstantLR':
        scheduler_D = None
    else:
        raise NotImplementedError

    # Data loading code
    # source training img_ids
    source_img_ids = glob(os.path.join(
        'inputs', config['dataset'], 'images_10pics_30k_patches', '*' + config['img_ext']))
    source_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in source_img_ids]
    source_train_img_ids, source_val_img_ids = train_test_split(
        source_img_ids, test_size=1/3, random_state=88)
    # ?????????
    source_train_img_ids = source_train_img_ids[:config['source_train_num']]

    # target training & validating img_ids
    target_img_ids = glob(os.path.join(
        'inputs', config['dataset'], 'images_30pics_6k_patches', '*' + config['img_ext']))
    target_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in target_img_ids]
    test_size_ = 1 - config['target_train_num']/6000
    target_train_img_ids, target_val_img_ids = train_test_split(
        target_img_ids, test_size=test_size_, random_state=88)
    target_train_img_ids = target_train_img_ids[:]  # ?????????  4k 2k

    # source & target train???????????????transform
    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # target val ????????????transform
    target_val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # ??????dataset ??????????????????batch???source_train ??? target_train??????
    train_dataset = Dataset_DA_pseuLabel(
        source_train_ids=source_train_img_ids,
        source_img_dir=os.path.join(
            'inputs', config['dataset'], 'images_10pics_30k_patches'),
        source_mask_dir=os.path.join(
            'inputs', config['dataset'], 'masks_10pics_30k_patches'),
        target_train_ids=target_train_img_ids,
        target_img_dir=os.path.join(
            'inputs', config['dataset'], 'images_30pics_6k_patches'),
        target_pseuLabel_dir=os.path.join(
            'inputs', config['dataset'], 'masks_target_pseudo_labels_0.6053'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform
    )

    val_dataset = TestDataset_DA(
        target_val_ids=target_val_img_ids,
        target_img_dir=os.path.join(
            'inputs', config['dataset'], 'images_30pics_6k_patches'),
        target_mask_dir=os.path.join(
            'inputs', config['dataset'], 'masks_30pics_6k_patches'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=target_val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('source_train_loss', []),
        ('source_train_dice', []),
        ('loss_D', []),
        ('loss_adv', []),
        ('loss_pseu', []),
        ('loss_mmd', []),
        ('lr_D', []),
        ('target_val_loss', []),
        ('target_val_dice', []),
    ])

    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model,
                          model_D, criterion, criterion_D, criterion_mmd, optimizer, optimizer_D)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['target_val_loss'])

        if config['scheduler_D'] == 'CosineAnnealingLR':
            scheduler_D.step()
        elif config['scheduler_D'] == 'ReduceLROnPlateau':
            scheduler_D.step(val_log['target_val_loss'])

        print('source_train_loss %.4f - source_train_dice %.4f - loss_pseu %.4f - loss_D %.4f - loss_adv %.4f - loss_mmd %.4f - target_val_loss %.4f - target_val_dice %.4f'
              % (train_log['source_train_loss'], train_log['source_train_dice'], train_log['loss_pseu'], train_log['loss_D'], train_log['loss_adv'], train_log['loss_mmd'], val_log['target_val_loss'], val_log['target_val_dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['source_train_loss'].append(train_log['source_train_loss'])
        log['source_train_dice'].append(train_log['source_train_dice'])
        log['loss_pseu'].append(train_log['loss_pseu'])
        log['loss_D'].append(train_log['loss_D'])
        log['loss_adv'].append(train_log['loss_adv'])
        log['loss_mmd'].append(train_log['loss_mmd'])
        log['lr_D'].append(config['lr_D'])
        log['target_val_loss'].append(val_log['target_val_loss'])
        log['target_val_dice'].append(val_log['target_val_dice'])

        pd.DataFrame(log).to_csv('models/%s/log_DA.csv' %
                                 config['name'], index=False)

        trigger += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'models/%s/model_epoch_%d.pth' %
                       (config['name'], epoch))
            print("=> saved model every 10 epochs")

        if val_log['target_val_dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['target_val_dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


def pred_target_1k_patches():

    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    # ??????D???loss????????????  nn.BCEWithLogitsLoss().cuda() or nn.MSELoss().cuda()
    criterion_D = nn.MSELoss().cuda()

    cudnn.benchmark = True  # ??????????????????cuDNN???auto-tuner????????????????????????????????????????????????

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()

    model = model.cuda()
    # model= nn.DataParallel(model,device_ids=[0,1,2])
    # summary(model, (3, 320, 320))

    # target training & validating img_ids
    target_img_ids = glob(os.path.join(
        'inputs', config['dataset'], 'images_30pics_6k_patches', '*' + config['img_ext']))
    target_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in target_img_ids]
    target_train_img_ids, target_val_img_ids = train_test_split(
        target_img_ids, test_size=2/6, random_state=88)
    target_train_img_ids = target_train_img_ids[:]  # ?????????  4k 2k

    # target val ????????????transform
    target_val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = TestDataset_DA(
        target_val_ids=target_val_img_ids,
        target_img_dir=os.path.join(
            'inputs', config['dataset'], 'images_30pics_6k_patches'),
        target_mask_dir=os.path.join(
            'inputs', config['dataset'], 'masks_30pics_6k_patches'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=target_val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    model.load_state_dict(torch.load(
        'models/%s/model.pth' % config['name']))
    model.eval()

    val_log = validate(config, val_loader, model, criterion)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    # pred_target_1k_patches()
