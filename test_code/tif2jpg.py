import numpy as np
import cv2
import os
from glob import glob

if __name__ == '__main__':
    masks_dir = './inputs/pic_for_method/target_pseulabel'
    # masks_dir_color = masks_dir
    masks_dir_color = './inputs/pic_for_method/target_pseulabel_color'

    ext = 'png'
    img_ids = glob(os.path.join(
        'inputs', 'pic_for_method', 'target_pseulabel', '*.'+ext))

    img_ids = [os.path.splitext(os.path.basename(p))[
        0] for p in img_ids]

    for img_id in img_ids:
        img = cv2.imread(os.path.join(masks_dir, img_id+'.'+ext), 1)
        lower = (0, 0, 0)
        upper = (10, 10, 10)
        mask = cv2.inRange(img, lower, upper)
        white_mask = ~mask

        # 彩色化
        # 浅蓝色 [255, 255, 187]  红色 [48, 48, 255] 绿色 [0, 139, 0]
        img[mask != 0] = [255, 255, 187]
        img[white_mask != 0] = [48, 48, 255]

        cv2.imwrite(os.path.join(masks_dir_color, img_id+'.'+ext), img)
        print('hello')
