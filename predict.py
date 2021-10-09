import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models.segmentation_models_pytorch as smp
import torch.nn as nn
from dataset_DA import TestDataset_DA

import archs
from dataset import Dataset, TestDataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from PIL import Image
from PIL import ImageFilter

from math import floor
import random
import numpy as np
import shutil
import denseCRF
import matplotlib.pyplot as plt


def sliding_window(image, stepSize, windowSize, height, width, savepath):
    # slide cut off the patches from the whole ihc
    # stepsize 滑窗单步步长
    # windowsize 窗口大小

    width, height = image.shape[1], image.shape[0]
    assert(windowSize[0] == windowSize[1])
    winSize = windowSize[0]
    stepSize = int(winSize/2)
    w_num = floor(width/stepSize-1+1)
    h_num = floor(height/stepSize-1+1)

    count = 0
    for y in range(0, h_num):  # 行数循环
        for x in range(0, w_num):  # 列数循环
            # 没超出下边界，也超出下边界
            if (y*stepSize + winSize) <= height and (x*stepSize + winSize) <= width:
                slide = image[y*stepSize:y*stepSize + winSize,
                              x*stepSize:x*stepSize + winSize, :]
                cv2.imwrite(savepath+'/'+str(count)+'.png', slide)
                count = count+1  # count持续加1
            elif(x == (w_num-1)) and (y*stepSize + winSize) <= height:
                slide = image[y*stepSize:y*stepSize +
                              winSize, width-winSize:width, :]
                cv2.imwrite(savepath+'/'+str(count)+'.png', slide)
                count += 1  # count持续加1

            elif(y == (h_num-1)) and (x*stepSize + winSize) <= width:
                slide = image[height-winSize:height,
                              x*stepSize:x*stepSize + winSize, :]
                cv2.imwrite(savepath+'/'+str(count)+'.png', slide)
                count += 1  # count持续加1
            elif(x == (w_num-1)) and (y == (h_num-1)):
                slide = image[height-winSize:height,
                              width-winSize: width, :]
                cv2.imwrite(savepath+'/'+str(count)+'.png', slide)
                count += 1  # count持续加1
            else:
                continue

    assert(count == (w_num * h_num))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='patient_datasetRandom_NestedUNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def dice_simple(output, target):

    smooth = 1e-5
    output_ = output > 125
    target_ = target > 125

    intersection = (output_ & target_).sum()
    return (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)


def predict_pseudoLabels():
    # produce the pseudoLables

    model_path = 'models/patient_datasetRandom_NestedUNet_woDS/model_20201208_0.87/model.pth'

    args = parse_args()

    with open('models/%s/model_20201208_0.87/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()
    model = model.cuda()

    img_ids = glob(os.path.join(
        'inputs', config['dataset'], 'images_30pics_6k_patches', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    test_img_ids = img_ids
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize()])

    test_dataset = TestDataset_DA(
        target_val_ids=test_img_ids,
        target_img_dir=os.path.join(
            'inputs', config['dataset'], 'images_30pics_6k_patches'),
        target_mask_dir=os.path.join(
            'inputs', config['dataset'], 'masks_30pics_6k_patches'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],  # config['num_workers']
        drop_last=False)

    avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output)
            zero = torch.zeros_like(output)
            one = torch.ones_like(output)
            output = torch.where(output > 0.5, one, zero)
            output = output.data.cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('inputs', config['dataset'], 'masks_target_pseudo_labels', meta['target_val_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('Dice: %.4f' % avg_meter.avg)

    torch.cuda.empty_cache()


def predict_img(predict_img='78-MHC-2', cut_size=144):

    args = parse_args()
    # load the model
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print the config information
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()

    model = model.cuda()
    # model= nn.DataParallel(model,device_ids=[0,1,2])

    stepSize = int(0.5*cut_size)  # 步长就是0.5倍的滑窗大小
    windowSize = [cut_size, cut_size]  # 滑窗大小

    path = 'inputs/image_ori'  # 文件路径
    mask_path = 'inputs/mask'
    savepath = 'inputs/patients_TestDataset/image'
    save_maskpath = 'inputs/patients_TestDataset/mask/0'
    output_path = os.path.join('outputs', config['name'], str(0))
    result_path = 'inputs/patients_TestDataset/result'

    if os.path.exists(savepath):
        shutil.rmtree(savepath)  # 先清空一下
        os.makedirs(savepath)
    if os.path.exists(save_maskpath):
        shutil.rmtree(save_maskpath)  # 先清空一下
        os.makedirs(save_maskpath)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # 先清空一下
        os.makedirs(output_path)

    image_name = path+'/' + predict_img+'.png'
    mask_name = mask_path+'/' + predict_img+'.png'
    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name)

    height, width = image.shape[:2]
    mask = cv2.resize(mask, (width, height))
    size1 = (int(floor(width/stepSize)*stepSize),
             int(floor(height/stepSize)*stepSize))  # round-四舍五入 ;int-图像大小必须是整数 
    print(size1)
    sliding_window(image, stepSize, windowSize, height, width, savepath)
    sliding_window(mask, stepSize, windowSize, height, width, save_maskpath)

    # Data loading code
    #  config['dataset']
    img_ids = glob(os.path.join(
        'inputs', config['dataset'], 'images_30pics_6k_patches', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, test_img_ids = train_test_split(img_ids, test_size=1.0, random_state=20)
    test_img_ids = img_ids

    model.load_state_dict(torch.load(
        'models/%s/model_epoch_70.pth' % config['name']))
    model.eval()

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize()])

    test_dataset = Dataset(  # Dataset
        img_ids=test_img_ids,
        # config['dataset']
        img_dir=os.path.join(
            'inputs', config['dataset'], 'images_30pics_6k_patches'),
        mask_dir=os.path.join(
            'inputs', config['dataset'], 'masks_30pics_6k_patches'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],  # config['num_workers']
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join(
            'outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):  # target
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))  # iou

            # torch.sigmoid(output).cpu().numpy()
            output = torch.sigmoid(output)
            zero = torch.zeros_like(output)
            one = torch.ones_like(output)
            output = torch.where(output > 0.5, one, zero)
            output = output.data.cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    # print('IoU: %.4f' % avg_meter.avg)
    print('Dice: %.4f' % avg_meter.avg)

    torch.cuda.empty_cache()
    '''
    image= cv2.imread(image_name)
    (height ,width, _)= image.shape
    #  width_num= floor(width*2/cut_size-1)  floor(width /cut_size )
    width_num= floor(width*2/cut_size-1)
    height_num= floor(height*2/cut_size-1)

    img_ids = glob( os.path.join('outputs', config['name'], str(0), '*.png') )
    num_patch= len(img_ids) #获取patch总数
    assert(num_patch == (width_num*height_num))

    Canvas =np.zeros(( height , width ),dtype=np.uint8)
    step_size = int(cut_size/2)
    for i in range( 0 ,height_num ):
        for j in range(0,width_num):
            patchFirst = cv2.imread( os.path.join(
                'outputs', config['name'], str(0), str( i * width_num + j )+'.png'),0 )
            patchFirst = cv2.resize(patchFirst, (cut_size,cut_size))
            patchFirst = (patchFirst>0).astype(np.uint8)
            if j == 0 :
                Canvas[ i* step_size : int(i*step_size + cut_size) , j* \
                                           step_size : int(j*step_size + cut_size) ] = patchFirst

            if ( j+1 ) < width_num: #同一行的可以blend起来
                patchNext = cv2.imread( os.path.join( 'outputs', config['name'], str(
                    0), str( i * width_num + j + 1 )+'.png'),0 )
                patchNext = cv2.resize(patchNext, (cut_size,cut_size))

                patchNext = (patchNext>0).astype(np.uint8)
                Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] = \
                    np.bitwise_and( Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] ,\
                         patchNext[ : , 0: int(cut_size/2) ] )
                Canvas[ i* step_size : int(i*step_size + cut_size) , int(j*step_size + cut_size) : int(j*step_size + 3*cut_size/2 )  ] =\
                     patchNext[ : , int(cut_size/2) : int(cut_size) ]

    cv2.imwrite('inputs/patients_TestDataset/result/' + \
                str(cut_size) +'.png', (Canvas*255).astype(np.uint8))
    '''

    '''
    image_temp = Image.new('RGB', ( int((width_num+1)*cut_size/2) ,int(
        (height_num+1)*cut_size/2)  ))  # int((width_num+1)*cut_size/2)  (height_num)*cut_size
    for i in range(0,num_patch):
        from_image = Image.open( os.path.join('outputs', config['name'], str(
            0), str(i)+'.png') ).resize((cut_size,cut_size))

        x = int( i % width_num)
        y = int (i / width_num)
        image_temp.paste(from_image, ( int(x*cut_size/2 ),
                         int(y*cut_size/2 )   ) ) #/2
    # image_temp = image_temp.filter(ImageFilter.BLUR)

    image_temp.save( result_path+'/'+predict_img + '-' + str(cut_size) + '.png')'''


def mergeMultiSizeImg(img_name='78-MHC-2'):
    cut_size_list = [100, 144, 300, 500]

    for cut_size in cut_size_list:
        predict_img(predict_img=img_name, cut_size=cut_size)

    oriImg = cv2.imread('inputs/image/'+img_name+'.png')
    height, width, channel = oriImg.shape
    MultiSizeResult = np.zeros((height, width)).astype(np.float)
    result_dir = 'inputs/patients_TestDataset/result'
    for cut_size in cut_size_list:
        img = cv2.imread(os.path.join(result_dir, str(cut_size)+'.png'), 0)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float)
        MultiSizeResult = MultiSizeResult + img

    MultiSizeResult /= len(cut_size_list)
    MultiSizeResult = MultiSizeResult.astype(np.uint8)
    MultiSizeResult_label = (MultiSizeResult > 125)
    MultiSizeResult[MultiSizeResult_label] = 255
    MultiSizeResult[~MultiSizeResult_label] = 0

    cv2.imwrite(result_dir + '/'+img_name+'-merged.png', MultiSizeResult)
    ori_mask = cv2.imread('inputs/mask/' + img_name + '.png', 0)

    dice_merge = dice_simple(MultiSizeResult, ori_mask)
    print('dice_merge = ', dice_merge)


def predict_img_target(predict_img='78-MHC-2', cut_size=144, model_path='model_20201215_v6'):

    args = parse_args()
    # load the model
    with open('models/%s/%s/config.yml' % (args.name, model_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print the config information
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()

    model = model.cuda()
    # model= nn.DataParallel(model,device_ids=[0,1,2])

    stepSize = int(0.5*cut_size)  # 步长就是0.5倍的滑窗大小
    windowSize = [cut_size, cut_size]  # 滑窗大小

    path = 'inputs/image_target'  # 原始文件路径
    mask_path = 'inputs/mask_target'
    savepath = 'inputs/patients_TestDataset/image'  # 裁剪小块路径
    save_maskpath = 'inputs/patients_TestDataset/mask/0'  # 裁剪小块mask路径
    output_path = os.path.join('outputs', config['name'], str(0))  # 预测小块mask路径
    result_path = 'inputs/patients_TestDataset/result'  # 预测小块拼接结果

    if os.path.exists(savepath):
        shutil.rmtree(savepath)  # 先清空一下
        os.makedirs(savepath)
    if os.path.exists(save_maskpath):
        shutil.rmtree(save_maskpath)  # 先清空一下
        os.makedirs(save_maskpath)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # 先清空一下
        os.makedirs(output_path)

    image_name = path+'/' + predict_img+'.png'
    mask_name = mask_path+'/' + predict_img+'.png'
    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name)

    height, width = image.shape[:2]
    mask = cv2.resize(mask, (width, height))
    sliding_window(image, stepSize, windowSize,
                   height, width, savepath)  # 滑窗裁剪小块
    sliding_window(mask, stepSize, windowSize, height,
                   width, save_maskpath)  # 滑窗裁剪小块mask

    # Data loading code
    img_ids = glob(os.path.join(
        'inputs', 'patients_TestDataset', 'image', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    test_img_ids = img_ids

    model.load_state_dict(torch.load(
        'models/%s/%s/model.pth' % (config['name'], model_path)))
    model.eval()

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize()])

    test_dataset = Dataset(  # Dataset
        img_ids=test_img_ids,
        img_dir=os.path.join(
            'inputs', 'patients_TestDataset', 'image'),
        mask_dir=os.path.join(
            'inputs', 'patients_TestDataset', 'mask', '0'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output)
            zero = torch.zeros_like(output)
            one = torch.ones_like(output)
            output = torch.where(output > 0.5, one, zero)
            output = output.data.cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('Dice: %.4f' % avg_meter.avg)
    torch.cuda.empty_cache()

    image = cv2.imread(image_name)
    (height, width, _) = image.shape

    width_num = floor(width/stepSize-1+1)
    height_num = floor(height/stepSize-1+1)

    img_ids = glob(os.path.join('outputs', config['name'], str(0), '*.png'))
    num_patch = len(img_ids)  # 获取patch总数
    assert(num_patch == (width_num*height_num))

    Canvas = np.zeros((height, width), dtype=np.float32)
    step_size = stepSize
    for i in range(0, height_num):  # 行遍历
        for j in range(0, width_num):  # 列遍历
            patch = cv2.imread(os.path.join(
                'outputs', config['name'], str(0), str(i * width_num + j)+'.png'), 0)
            patch = cv2.resize(patch, (cut_size, cut_size))
            patch = (patch > 0).astype(np.float32)

            if(i != height_num-1) and (j != width_num-1):
                Canvas[i * step_size: int(i*step_size + cut_size), j * step_size: int(j*step_size + cut_size)] = \
                    Canvas[i * step_size: int(i*step_size + cut_size),
                           j * step_size: int(j*step_size + cut_size)] + patch

    print('-'*20)
    Canvas[1*step_size: int(height_num-1)*step_size, 1*step_size:int(width_num-1)*step_size] = \
        Canvas[1*step_size: int(height_num-1)*step_size,
               1*step_size:int(width_num-1)*step_size] / 4.0

    for i in range(0, height_num):  # 行遍历
        for j in range(0, width_num):  # 列遍历
            patch = cv2.imread(os.path.join(
                'outputs', config['name'], str(0), str(i * width_num + j)+'.png'), 0)
            patch = cv2.resize(patch, (cut_size, cut_size))
            patch = (patch > 0).astype(np.float32)

            if (i != height_num-1) and (j == width_num-1):
                Canvas[i * step_size: int(i*step_size + cut_size), width-cut_size:width] = \
                    Canvas[i * step_size: int(i*step_size + cut_size),
                           width-cut_size:width] + patch
            elif (i == height_num-1) and (j != width_num-1):
                Canvas[height-cut_size:height, j * step_size: int(j*step_size + cut_size)] = \
                    Canvas[height-cut_size:height, j *
                           step_size: int(j*step_size + cut_size)] + patch
            elif (i == height_num-1) and (j == width_num-1):
                Canvas[height-cut_size:height, width-cut_size:width] = \
                    Canvas[height-cut_size:height,
                           width-cut_size:width] + patch
            else:
                continue

    Canvas[Canvas >= 0.5] = 1
    Canvas[Canvas < 0.5] = 0

    cv2.imwrite('inputs/patients_TestDataset/result/' +
                str(cut_size) + '.png', (Canvas*255).astype(np.uint8))


def mergeMultiSizeImg_target(img_name='78-MHC-2', model_path='model_20201215_v6', cut_size_list=[100, 144, 300, 500]):
    # cut_size_list = [100, 144, 300, 500]

    for cut_size in cut_size_list:
        predict_img_target(img_name, cut_size, model_path)

    oriImg = cv2.imread('inputs/image_target/'+img_name+'.png')
    height, width, channel = oriImg.shape
    MultiSizeResult = np.zeros((height, width)).astype(np.float)
    result_dir = 'inputs/patients_TestDataset/result'
    for cut_size in cut_size_list:
        img = cv2.imread(os.path.join(result_dir, str(cut_size)+'.png'), 0)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float)
        MultiSizeResult = MultiSizeResult + img

    MultiSizeResult /= len(cut_size_list)
    MultiSizeResult = MultiSizeResult.astype(np.uint8)
    MultiSizeResult_label = (MultiSizeResult > 128)
    MultiSizeResult[MultiSizeResult_label] = 255
    MultiSizeResult[~MultiSizeResult_label] = 0

    os.makedirs(result_dir + '/mask_target/'+model_path, exist_ok=True)

    ori_mask = cv2.imread('inputs/mask_target/' + img_name + '.png', 0)
    dice_merge = dice_simple(MultiSizeResult, ori_mask)
    cv2.imwrite(result_dir + '/mask_target/' + model_path+'/' +
                img_name+'-merged_'+str(round(dice_merge, 4))+'.png', MultiSizeResult)
    print('dice_merge = ', dice_merge)
    return dice_merge


def predict_allMasks():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'],
                     classes=config['num_classes'], encoder_weights='imagenet').cuda()

    model = model.cuda()
    # model = nn.DataParallel(model)
    model = nn.DataParallel(model, device_ids=[0, 1, 2])

    path = 'inputs/image'  # 文件路径
    savepath = 'inputs/patients_TestDataset/image'
    output_path = os.path.join('outputs', config['name'], str(0))

    image_dir = 'inputs/image'
    result_dir = 'inputs/patients_TestDataset/result/mask'
    img_name_list = os.listdir(image_dir)

    for img in img_name_list:
        predict_img = img.split('.')[0]
        cut_size_list = [100, 144, 300, 500]
        if os.path.exists(os.path.join(result_dir, predict_img + '.png')):
            continue

        for cut_size in cut_size_list:
            stepSize = int(0.5*cut_size)  # 步长就是0.5倍的滑窗大小
            windowSize = [cut_size, cut_size]  # 滑窗大小

            if os.path.exists(savepath):
                shutil.rmtree(savepath)  # 先清空一下
                os.makedirs(savepath)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)  # 先清空一下
                os.makedirs(output_path)

            image_name = path+'/' + predict_img+'.jpg'
            image = cv2.imread(image_name)
            # print('predict_img',predict_img)

            height, width = image.shape[:2]
            sliding_window(image, stepSize, windowSize,
                           height, width, savepath)

            img_ids = glob(os.path.join(
                'inputs', 'patients_TestDataset', 'image', '*' + config['img_ext']))
            img_ids = [os.path.splitext(os.path.basename(p))[
                0] for p in img_ids]

            test_img_ids = img_ids

            model.load_state_dict(torch.load(
                'models/%s/model.pth' % config['name']))
            model.eval()

            test_transform = Compose([
                transforms.Resize(config['input_h'], config['input_w']),
                transforms.Normalize()])

            test_dataset = TestDataset(  # Dataset
                img_ids=test_img_ids,
                # config['dataset']
                img_dir=os.path.join(
                    'inputs', 'patients_TestDataset', 'image'),
                img_ext=config['img_ext'],
                num_classes=config['num_classes'],
                transform=test_transform)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],  # config['num_workers']
                drop_last=False)

            for c in range(config['num_classes']):
                os.makedirs(os.path.join(
                    'outputs', config['name'], str(c)), exist_ok=True)
            with torch.no_grad():
                for input, meta in tqdm(test_loader, total=len(test_loader)):  # target
                    input = input.cuda()

                    # compute output
                    if config['deep_supervision']:
                        output = model(input)[-1]
                    else:
                        output = model(input)

                    # torch.sigmoid(output).cpu().numpy()
                    output = torch.sigmoid(output)
                    zero = torch.zeros_like(output)
                    one = torch.ones_like(output)
                    output = torch.where(output > 0.5, one, zero)
                    output = output.data.cpu().numpy()

                    for i in range(len(output)):
                        for c in range(config['num_classes']):
                            cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                        (output[i, c] * 255).astype('uint8'))

            torch.cuda.empty_cache()

            image = cv2.imread(image_name)
            (height, width, _) = image.shape
            # width_num= floor(width*2/cut_size-1)  floor(width /cut_size )
            width_num = floor(width*2/cut_size-1)
            height_num = floor(height*2/cut_size-1)

            img_ids = glob(os.path.join(
                'outputs', config['name'], str(0), '*.png'))
            num_patch = len(img_ids)  # 获取patch总数
            assert(num_patch == (width_num*height_num))

            Canvas = np.zeros((height, width), dtype=np.uint8)
            step_size = int(cut_size/2)
            for i in range(0, height_num):
                for j in range(0, width_num):
                    patchFirst = cv2.imread(os.path.join(
                        'outputs', config['name'], str(0), str(i * width_num + j)+'.png'), 0)
                    patchFirst = cv2.resize(patchFirst, (cut_size, cut_size))
                    patchFirst = (patchFirst > 0).astype(np.uint8)
                    if j == 0:
                        Canvas[i * step_size: int(i*step_size + cut_size), j * step_size: int(
                            j*step_size + cut_size)] = patchFirst

                    if (j+1) < width_num:  # 同一行的可以blend起来
                        patchNext = cv2.imread(os.path.join(
                            'outputs', config['name'], str(0), str(i * width_num + j + 1)+'.png'), 0)
                        patchNext = cv2.resize(patchNext, (cut_size, cut_size))

                        patchNext = (patchNext > 0).astype(np.uint8)
                        Canvas[i * step_size: int(i*step_size + cut_size), int(j * step_size + cut_size/2):  int(j*step_size + cut_size)] = np.bitwise_and(Canvas[i * step_size: int(i*step_size + cut_size), int(j * step_size + cut_size/2):  int(j*step_size + cut_size)],
                                                                                                                                                           patchNext[:, 0: int(cut_size/2)])
                        Canvas[i * step_size: int(i*step_size + cut_size), int(j*step_size + cut_size): int(
                            j*step_size + 3*cut_size/2)] = patchNext[:, int(cut_size/2): int(cut_size)]

            cv2.imwrite('inputs/patients_TestDataset/result/' +
                        str(cut_size) + '.png', (Canvas*255).astype(np.uint8))

        oriImg = cv2.imread('inputs/image/'+predict_img+'.jpg')
        height, width, channel = oriImg.shape
        MultiSizeResult = np.zeros((height, width)).astype(np.float)
        result_dir = 'inputs/patients_TestDataset/result'
        for cut_size in cut_size_list:
            img = cv2.imread(os.path.join(result_dir, str(cut_size)+'.png'), 0)
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float)
            MultiSizeResult = MultiSizeResult + img

        MultiSizeResult /= len(cut_size_list)
        MultiSizeResult = MultiSizeResult.astype(np.uint8)
        MultiSizeResult_label = (MultiSizeResult > 125)
        MultiSizeResult[MultiSizeResult_label] = 255
        MultiSizeResult[~MultiSizeResult_label] = 0

        cv2.imwrite(result_dir + '/mask/' +
                    predict_img+'.png', MultiSizeResult)


def stich_img(img_name='0-3D-1'):

    windowSize = 144

    ori_img = cv2.imread('inputs/patient_pic/'+img_name+'.jpg')
    (height, width, _) = ori_img.shape
    width_num = floor(width*2/windowSize-1)
    height_num = floor(height*2/windowSize-1)
    # dir_path= 'inputs/patients_dataset/images'
    dir_path = 'outputs/patient_datasetRandom_NestedUNet_woDS_0-3D-1/0'

    img_ids = glob(dir_path+'/'+img_name + '-*' + '.jpg')
    num_patch = len(img_ids)  # 获取patch总数
    assert(num_patch == (width_num*height_num))

    image_temp = Image.new(
        'RGB', (int((width_num+1)*windowSize/2), int((height_num+1)*windowSize/2)))

    for i in range(0, num_patch):
        from_image = Image.open(dir_path+'/'+img_name + '-' + str(i)+'.jpg')
        x = int(i % width_num)
        y = int(i / width_num)
        image_temp.paste(
            from_image, (int(x*windowSize/2), int(y*windowSize/2)))

    image_temp.save(img_name+'-stiched'+'.png')


def produce_randomCropDataset(output_size=[100, 144, 300, 500], cropPicName=['0-2D-1', '0-2D-2'], cropNumEach=3000):
    cropPicDir = 'inputs/image_ori'
    cropMastDir = 'inputs/mask'
    savePicDir = 'inputs/patient_datasetRandom/images_10pics_30k_patches'
    saveMaskDir = 'inputs/patient_datasetRandom/masks_10pics_30k_patches'
    if not os.path.exists(savePicDir):
        os.makedirs(savePicDir)
    if not os.path.exists(saveMaskDir):
        os.makedirs(saveMaskDir)

    imglist = [cv2.imread(cropPicDir+'/'+p+'.png') for p in cropPicName]
    masklist = [cv2.imread(cropMastDir+'/'+p+'.png') for p in cropPicName]

    for ind in range(len(imglist)):
        (h, w, c) = imglist[ind].shape
        print('ind:', ind)
        for i in range(cropNumEach):
            pos1, pos2 = random.uniform(0, 1), random.uniform(0, 1)
            outputInd = random.randint(0, 3)  # 随机裁剪大小不定
            w1 = int(pos1 * (w - output_size[outputInd]))
            sw = output_size[outputInd]
            h1 = int(pos2 * (h - output_size[outputInd]))
            sh = output_size[outputInd]

            mask_gray = masklist[ind][h1:h1+sh, w1:w1+sw]
            # mask_label = (mask_bgr[:,:,2]>0)
            # mask= np.zeros( (mask_bgr.shape[0],mask_bgr.shape[1]) )
            # mask[mask_label]=1
            cv2.imwrite(
                saveMaskDir+'/'+cropPicName[ind]+'-'+str(i)+'.png', (mask_gray).astype('uint8'))

            image = imglist[ind][h1:h1+sh, w1:w1+sw, :]
            cv2.imwrite(savePicDir+'/' +
                        cropPicName[ind]+'-'+str(i)+'.png', image)


def colorMask():
    sourceDir = 'inputs/patient_datasetRandom/masks_10pics_30k_patches'
    colorDir = 'inputs/patient_datasetRandom/masks_source_Color'
    id_to_trainid = {0: 0, 255: 1}

    palette = [0, 0, 0, 128, 64, 128]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    for _, _, files in os.walk(sourceDir):
        # print(files)
        for patch_mask in files:
            mask = Image.open(sourceDir+'/'+patch_mask).convert('L')
            mask = np.asarray(mask, np.float32)
            mask_copy = 255 * np.ones(mask.shape, dtype=np.float32)
            for k, v in id_to_trainid.items():
                mask_copy[mask == k] = v

            new_mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
            new_mask.putpalette(palette)
            new_mask.save(colorDir+'/'+patch_mask)


def noColorMask():
    sourceDir = 'inputs/patient_datasetRandom/masks_30pics_6k_patches'
    colorDir = 'inputs/patient_datasetRandom/masks_target_noColor'
    id_to_trainid = {0: 0, 255: 1}

    for _, _, files in os.walk(sourceDir):
        # print(files)
        for patch_mask in files:
            mask = Image.open(sourceDir+'/'+patch_mask).convert('L')
            mask = np.asarray(mask, np.float32)
            mask_copy = 255 * np.ones(mask.shape, dtype=np.float32)
            for k, v in id_to_trainid.items():
                mask_copy[mask == k] = v

            mask_copy = Image.fromarray(mask_copy.astype(np.uint8))
            mask_copy.save(colorDir+'/'+patch_mask)


def densecrf(I, P, param):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices. 
    """
    out = denseCRF.densecrf(I, P, param)
    return out


def densecrf1(img_id, mask_id, model_path):
    image_dir = './inputs/image_target/'
    mask_pred_dir = './inputs/patients_TestDataset/result/mask_target/'+model_path+'/'
    save_crf_dir = './inputs/patients_TestDataset/result/mask_target_crf/'+model_path
    os.makedirs(save_crf_dir, exist_ok=True)

    I = Image.open(image_dir+img_id+'.png')
    Iq = np.asarray(I)

    # load initial labels, and convert it into an array 'prob' with shape [H, W, C]
    # where C is the number of labels
    # prob[h, w, c] means the probability of pixel at (h, w) belonging to class c.
    L = Image.open(mask_pred_dir+mask_id+'.png').convert("RGB")
    Lq = np.asarray(L, np.float32) / 255
    prob = Lq[:, :, :2]
    prob[:, :, 0] = 1.0 - prob[:, :, 0]

    w1 = 10.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta = 13    # rgb  std
    w2 = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it = 5.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    lab = densecrf(Iq, prob, param)
    lab = Image.fromarray(lab*255).convert('L')
    lab.save(os.path.join(save_crf_dir, mask_id+'.png'))
    '''
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(I)
    plt.title('input image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(L)
    plt.title('initial label')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(lab)
    plt.title('after dense CRF')
    plt.savefig("demo_densecrf1.png")
    plt.show()'''


def produce_crfmasks(model_path='model_20201212_v4_adam_0.78_0.7179'):
    mask_ids = glob(os.path.join(
        'inputs', 'patients_TestDataset', 'result', 'mask_target', model_path, '*.png'))
    mask_ids = [os.path.splitext(os.path.basename(p))[0] for p in mask_ids]
    for mask_id in mask_ids:
        img_id = mask_id.split('-merged')[0]
        densecrf1(img_id, mask_id, model_path)


def calculate_dice_crfmasks(model_path):
    mask_ids = glob(os.path.join(
        'inputs', 'patients_TestDataset', 'result', 'mask_target_crf', model_path, '*.png'))
    mask_ids = [os.path.splitext(os.path.basename(p))[0] for p in mask_ids]
    dice_list = []
    for mask_id in mask_ids:
        img_id = mask_id.split('-merged')[0]
        mask_target = cv2.imread(os.path.join(
            'inputs', 'mask_target', img_id+'.png'), 0)
        mask_target_crf = cv2.imread(os.path.join(
            'inputs', 'patients_TestDataset', 'result', 'mask_target_crf', model_path, mask_id+'.png'), 0)
        dice = dice_simple(mask_target_crf, mask_target)
        dice_list.append(dice)
    dice_mean = np.mean(np.array(dice_list))
    print('dice_mean = ', dice_mean)


if __name__ == '__main__':
    '''
    image_dir = './inputs/image'
    cropPicName=[ ]
    filelist = os.listdir(image_dir)
    for file_ in filelist:
        fileID = file_.split('.')[0]
        cropPicName.append(fileID)
    '''

    '''
    # 随机裁剪生成训练集
    cropPicName = ['0-2D-1','0-2D-2','0-3D-1',
        '2-GAL9-2','42-PD1-2','44-PDL1-1','67-TIM3-2']
    produce_randomCropDataset(output_size=[100,144,300,500], cropPicName= cropPicName,cropNumEach=3000)'''

    # predict_img(predict_img='138-OX40-2', cut_size=144)
    # mergeMultiSizeImg(img_name='89-LAG3-1')

    # predict_allMasks()

    # stich_img(img_name='0-3D-1')

    # 生成source 数据集
    '''
    cropPicName = ['0-2D-1','0-2D-2','0-3D-1','2-GAL9-2','42-PD1-2',
            '44-PDL1-1','67-TIM3-2','78-MHC-2','89-LAG3-1','138-OX40-2']
    produce_randomCropDataset(output_size=[100,144,300,500], cropPicName= cropPicName,cropNumEach=3000)'''

    # colorMask()
    # noColorMask()

    # 生成 target 数据集
    '''
    image_target_dir = 'inputs/image_target'
    for _, _, files in os.walk(image_target_dir):
        # print(files)
        cropPicName = [x.split('.')[0]  for x in files]
    produce_randomCropDataset(
        output_size=[100,144,300,500], cropPicName= cropPicName,cropNumEach=200)
    '''
    # colorMask()
    # noColorMask()

    # predict_pseudoLabels()

    # produce the predicted results
    '''
    img_ids = glob(os.path.join(
        'inputs', 'image_target', '*' + '.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    dice_list = []
    for img_id in img_ids[:]:
        dice_merge = mergeMultiSizeImg_target(img_name=img_id, model_path='model_20201208_0.87', cut_size_list=[
            200, 300, 500, 700, 800])
        dice_list.append(dice_merge)
    dice_mean = np.mean(np.array(dice_list))
    print('dice_mean = ', dice_mean)'''

    # use denseCRF modify the masks
    # demo_densecrf1()
    # crf 之后dice都下降了
    # 0.5570->0.5035 0.7179->0.6819 0.6903->0.6413
    '''
    produce_crfmasks(model_path='model_20201212_v4_adam_0.78_0.7179')
    calculate_dice_crfmasks(model_path='model_20201212_v4_adam_0.78_0.7179')'''
