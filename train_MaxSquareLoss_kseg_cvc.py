import argparse
import os
from collections import OrderedDict
from glob import glob
import cv2
import random

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data, model_zoo
from torchsummary import summary
from model_MaxSquareLoss.deeplab_multi import DeeplabMulti

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
from dataset_DA import Dataset_DA, TestDataset_DA
from metrics import iou_score, dice_coef, precision, recall
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

    # dataset
    parser.add_argument('--dataset', default='patient_datasetRandom',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.jpg',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
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

    # lambda
    parser.add_argument('--lambda_seg', default=0.1, type=float)
    parser.add_argument('--lambda_target', default=1, type=float)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--ignore_index', default=0, type=int)

    # MaxSquareloss
    parser.add_argument('--msl_loss', default='MaxSquareloss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: MaxSquareloss)')

    parser.add_argument('--source_train_num', default=1000, type=int)
    parser.add_argument('--target_train_num', default=400, type=int)

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model,  criterion, optimizer):
    avg_meters = {'source_train_loss': AverageMeter(),
                  'source_train_dice': AverageMeter(),
                  'loss_seg': AverageMeter(),
                  'loss_target': AverageMeter(),
                  'loss_seg2': AverageMeter(),
                  'loss_target2': AverageMeter()}

    model.train()
    optimizer.zero_grad()

    interp_source = nn.Upsample(
        size=(config['input_h'], config['input_w']), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        config['input_h'], config['input_w']), mode='bilinear', align_corners=True)

    pbar = tqdm(total=len(train_loader))
    for source_img, source_mask, target_train_img, _ in train_loader:
        optimizer.zero_grad()

        loss_seg_value = 0
        loss_target_value = 0
        loss_seg_value_2 = 0
        loss_target_value_2 = 0

        source_img = source_img.cuda()
        source_mask = source_mask.cuda()
        target_train_img = target_train_img.cuda()

        # source supervised loss
        # train with source
        pred, pred_2 = model(source_img)  # 两个输出
        source_train_loss = criterion(pred, source_mask)
        source_train_dice = dice_coef(pred, source_mask)

        loss_ = source_train_loss
        loss_2 = config['lambda_seg'] * criterion(pred_2, source_mask)
        loss_ += loss_2
        loss_seg_value_2 += loss_2.cpu().item()

        loss_.backward()
        loss_seg_value += source_train_loss.cpu().item()

        # target loss
        # train with target
        pred, pred_2 = model(target_train_img)
        pred_P = torch.softmax(pred, dim=1)
        pred_P_2 = torch.softmax(pred_2, dim=1)

        label = pred_P
        label_2 = pred_P_2

        maxpred, argpred = torch.max(pred_P.detach(), dim=1)
        maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

        criterion_msl = losses.__dict__[config['msl_loss']]().cuda()
        loss_target = config['lambda_target'] * \
            criterion_msl(pred, label)

        loss_target_ = loss_target

        # Multi-level Self-produced Guidance
        pred_c = (pred_P+pred_P_2)/2
        maxpred_c, argpred_c = torch.max(pred_c, dim=1)
        mask = (maxpred > config['threshold']) | (
            maxpred_2 > config['threshold'])

        label_2 = torch.where(mask, argpred_c, torch.ones(1).to(
            torch.device('cuda'), dtype=torch.long)*config['ignore_index'])
        label_2 = label_2.unsqueeze(1).float()

        loss_target_2 = config['lambda_seg'] * config['lambda_target'] * \
            criterion(pred_2, label_2)
        loss_target_ += loss_target_2
        loss_target_value_2 += loss_target_2.item()

        loss_target_.backward()
        loss_target_value += loss_target.item()

        # do optimizing step
        optimizer.step()

        avg_meters['source_train_loss'].update(
            source_train_loss.item(), source_img.size(0))
        avg_meters['source_train_dice'].update(
            source_train_dice.item(), source_img.size(0))
        avg_meters['loss_seg'].update(
            loss_seg_value, source_img.size(0))
        avg_meters['loss_target'].update(
            loss_target_value, source_img.size(0))
        avg_meters['loss_seg2'].update(
            loss_seg_value_2, source_img.size(0))
        avg_meters['loss_target2'].update(
            loss_target_value_2, source_img.size(0))

        postfix = OrderedDict([
            ('source_train_loss', avg_meters['source_train_loss'].avg),
            ('source_train_dice', avg_meters['source_train_dice'].avg),
            ('loss_seg', avg_meters['loss_seg'].avg),
            ('loss_target', avg_meters['loss_target'].avg),
            ('loss_seg2', avg_meters['loss_seg2'].avg),
            ('loss_target2', avg_meters['loss_target2'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return OrderedDict([('source_train_loss', avg_meters['source_train_loss'].avg), ('source_train_dice', avg_meters['source_train_dice'].avg),
                        ('loss_seg', avg_meters['loss_seg'].avg), (
                            'loss_target', avg_meters['loss_target'].avg),
                        ('loss_seg2', avg_meters['loss_seg2'].avg), ('loss_target2', avg_meters['loss_target2'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'target_val_loss': AverageMeter(),
                  'target_val_dice': AverageMeter(),
                  'target_val_iou': AverageMeter(),
                  'target_val_precision': AverageMeter(),
                  'target_val_recall': AverageMeter()}

    interp_source = nn.Upsample(
        size=(config['input_h'], config['input_w']), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(
        config['input_h'], config['input_w']), mode='bilinear', align_corners=True)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for target_val_img, target_val_mask, meta in val_loader:

            target_val_img = target_val_img.cuda()
            target_val_mask = target_val_mask.cuda()

            # compute output
            target_val_pred1, target_val_pred2 = model(target_val_img)
            target_val_pred1 = interp_target(target_val_pred1)
            target_val_pred2 = interp_target(target_val_pred2)
            target_val_loss = criterion(target_val_pred1, target_val_mask)
            target_val_dice = dice_coef(target_val_pred1, target_val_mask)

            target_val_iou = iou_score(target_val_pred1, target_val_mask)
            target_val_precision = precision(target_val_pred1, target_val_mask)
            target_val_recall = recall(target_val_pred1, target_val_mask)

            # output the pred mask pics
            pred_target = torch.sigmoid(target_val_pred1)
            zero = torch.zeros_like(pred_target)
            one = torch.ones_like(pred_target)
            pred_target = torch.where(pred_target > 0.5, one, zero)
            pred_target = pred_target.data.cpu().numpy()
            for i in range(len(pred_target)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('inputs', 'CVC-ClinicDB', 'pred_target_212_patches', meta['target_val_id'][i] + '.jpg'),
                                (pred_target[i, c] * 255).astype('uint8'))

            avg_meters['target_val_loss'].update(
                target_val_loss.item(), target_val_img.size(0))
            avg_meters['target_val_dice'].update(
                target_val_dice, target_val_img.size(0))
            avg_meters['target_val_iou'].update(
                target_val_iou, target_val_img.size(0))
            avg_meters['target_val_precision'].update(
                target_val_precision, target_val_img.size(0))
            avg_meters['target_val_recall'].update(
                target_val_recall, target_val_img.size(0))

            postfix = OrderedDict([
                ('target_val_loss', avg_meters['target_val_loss'].avg),
                ('target_val_dice', avg_meters['target_val_dice'].avg),
                ('target_val_iou', avg_meters['target_val_iou'].avg),
                ('target_val_precision',
                 avg_meters['target_val_precision'].avg),
                ('target_val_recall', avg_meters['target_val_recall'].avg)
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

    cudnn.benchmark = True  # 可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法

    # create model
    print("=> creating model %s" % config['arch'])
    '''
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()
    model_D = FCDiscriminator(num_classes=config['num_classes']).cuda()'''
    model = DeeplabMulti(num_classes=config['num_classes'])

    model = model.cuda()

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

    # Data loading code
    # source training img_ids
    source_img_ids = glob(os.path.join(
        'inputs', 'Kvasir-SEG', 'images', '*' + config['img_ext']))
    source_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in source_img_ids]
    random.shuffle(source_img_ids)
    source_train_img_ids = source_img_ids
    '''
    source_train_img_ids, source_val_img_ids = train_test_split(
        source_img_ids, test_size=1/3, random_state=88)
    source_train_img_ids = source_train_img_ids[:12000]  # 改数量'''

    # target training & validating img_ids
    target_img_ids = glob(os.path.join(
        'inputs', 'CVC-ClinicDB', 'images', '*' + config['img_ext']))
    target_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in target_img_ids]
    test_size_ = 1 - config['target_train_num']/612.0

    target_train_img_ids, target_val_img_ids = train_test_split(
        target_img_ids, test_size=test_size_, random_state=88)
    target_train_img_ids = target_train_img_ids[:]  # 改数量  400 212

    # source & target train数据集统一transform
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

    # target val 数据集的transform
    target_val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 重写dataset 一次同步读取batch个source_train 和 target_train数据
    train_dataset = Dataset_DA(
        source_train_ids=source_train_img_ids,
        source_img_dir=os.path.join(
            'inputs', 'Kvasir-SEG', 'images'),
        source_mask_dir=os.path.join(
            'inputs', 'Kvasir-SEG', 'masks'),
        target_train_ids=target_train_img_ids,
        target_img_dir=os.path.join(
            'inputs', 'CVC-ClinicDB', 'images'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform
    )
    val_dataset = TestDataset_DA(
        target_val_ids=target_val_img_ids,
        target_img_dir=os.path.join(
            'inputs', 'CVC-ClinicDB', 'images'),
        target_mask_dir=os.path.join(
            'inputs', 'CVC-ClinicDB', 'masks'),
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
        ('loss_seg', []),
        ('loss_target', []),
        ('loss_seg2', []),
        ('loss_target2', []),
        ('target_val_loss', []),
        ('target_val_dice', []),
    ])

    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model,
                          criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['target_val_loss'])

        print('source_train_loss %.4f - source_train_dice %.4f - loss_seg %.4f - loss_target %.4f - loss_seg2 %.4f - loss_target2 %.4f - target_val_loss %.4f - target_val_dice %.4f'
              % (train_log['source_train_loss'], train_log['source_train_dice'], train_log['loss_seg'], train_log['loss_target'], train_log['loss_seg2'], train_log['loss_target2'], val_log['target_val_loss'], val_log['target_val_dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['source_train_loss'].append(train_log['source_train_loss'])
        log['source_train_dice'].append(train_log['source_train_dice'])
        log['loss_seg'].append(train_log['loss_seg'])
        log['loss_target'].append(train_log['loss_target'])
        log['loss_seg2'].append(train_log['loss_seg2'])
        log['loss_target2'].append(train_log['loss_target2'])
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


def pred_target_212_patches(model_dir):
    # model_dir = 'model_20201221_kseg_source'
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

    with open('models/%s/%s/config.yml' % (config['name'], model_dir), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    # 设置D的loss计算方式  nn.BCEWithLogitsLoss().cuda() or nn.MSELoss().cuda()
    criterion_D = nn.MSELoss().cuda()

    cudnn.benchmark = True  # 可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法

    # create model
    print("=> creating model %s" % config['arch'])
    '''
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()'''
    model = DeeplabMulti(num_classes=config['num_classes'])

    model = model.cuda()
    # model= nn.DataParallel(model,device_ids=[0,1,2])
    # summary(model, (3, 320, 320))

    # target training & validating img_ids
    target_img_ids = glob(os.path.join(
        'inputs', 'CVC-ClinicDB', 'images', '*' + '.jpg'))
    target_img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in target_img_ids]
    test_size_ = 1 - config['target_train_num']/612.0

    target_train_img_ids, target_val_img_ids = train_test_split(
        target_img_ids, test_size=test_size_, random_state=88)
    target_train_img_ids = target_train_img_ids[:]  # 改数量  400 212

    # target val 数据集的transform
    target_val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = TestDataset_DA(
        target_val_ids=target_val_img_ids,
        target_img_dir=os.path.join(
            'inputs', 'CVC-ClinicDB', 'images'),
        target_mask_dir=os.path.join(
            'inputs', 'CVC-ClinicDB', 'masks'),
        img_ext='.jpg',
        mask_ext='.jpg',
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
        'models/%s/%s/model.pth' % (config['name'], model_dir)))
    model.eval()

    val_log = validate(config, val_loader, model, criterion)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # main()
    pred_target_212_patches(model_dir='model_20201222_cvc_msl')
