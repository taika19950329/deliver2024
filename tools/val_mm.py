import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
import matplotlib.pyplot as plt


PALETTE = torch.tensor([
    [70, 70, 70],
    [100, 40, 40],
    [55, 90, 80],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [244, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0],
    [70, 130, 180],
    [81, 0, 81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180],
    [250, 170, 30],
    [110, 190, 160],
    [170, 120, 50],
    [45, 60, 150],
    [145, 170, 100],
    [0, 0, 230],
    [0, 60, 100],
    [0, 0, 70],
])

def unnormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图像数据。

    参数:
      image: tensor，形状 [C, H, W]
      mean: list or tuple，每个通道的均值
      std: list or tuple，每个通道的标准差

    返回:
      反归一化后的 tensor
    """
    # 克隆一份以避免修改原始数据
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image


def decode_segmap(mask, palette):
    """
    将 segmentation mask 映射成彩色图像。

    参数：
      mask: numpy 数组，形状 (H, W)，每个像素值为 0-24 的类别索引。
      palette: numpy 数组，形状 (25, 3)，每个类别对应的 RGB 颜色（0-255）。

    返回：
      color_mask: numpy 数组，形状 (H, W, 3)，彩色图像。
    """
    # 确保 mask 为整数类型
    mask = mask.astype(np.int64)
    # 利用向量化索引直接映射：mask 的每个值作为索引选取 palette 中对应的颜色
    color_mask = palette[mask]
    return color_mask

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img
@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2] * 1)), int(ceil(image_size[3] * 1)))
    overlap = 1 / 3
    stride = ceil(tile_size[0] * (1 - overlap))
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])
            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction, _, _, _, _ = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions, _, _, _, _ = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)
    return total_predictions.unsqueeze(0)
@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            preds = model(images).softmax(dim=1) #######\

            # 针对当前 batch 中的每个样本进行可视化
            batch_size = 1
            # for i in range(batch_size):
            #     # 转换 tensor 为 numpy 数组，便于保存
            #     images[i] = torch.squeeze(images[i])
            #     image_unnorm = unnormalize(images[i].cpu())
            #     image_unnorm = image_unnorm.clamp(0, 1)
            #     image_np = image_unnorm.numpy().transpose(1, 2, 0)  # 转换为 [H, W, C]
            #     # print(image_np)
            #     palette_np = PALETTE.numpy()
            #     # label_np = labels[i].cpu().numpy()  # [H, W]
            #     # label_np = decode_segmap(label_np, palette_np).astype(np.uint8)
            #     # print("label_np", label_np.shape)
            #     pred_mask = np.ascontiguousarray(preds[i].cpu().numpy())  # [H, W]
            #     pred_mask = decode_segmap(pred_mask, palette_np).astype(np.uint8)
            #     print("pred_mask", pred_mask.shape)
            #     # raise Exception
            #
            #     # 保存输入图像
            #     # input_save_path = os.path.join('/home/yi/Documents/DELIVER', f'input_batch{i}_img{i}.png')
            #     # plt.imsave(input_save_path, image_np)
            #     #
            #     # # 保存标签图像，使用 'jet' colormap 显示分割结果
            #     # label_save_path = os.path.join('/home/yi/Documents/DELIVER', f'label_batch{i}_img{i}.png')
            #     # plt.imsave(label_save_path, label_np, cmap='jet')
            #
            #     # 保存预测图像，使用 'jet' colormap 显示分割结果
            #     pred_save_path = os.path.join('/home/yi/Documents/DELIVER', f'prediction_batch{i}_img{i}.png')
            #     plt.imsave(pred_save_path, pred_mask, cmap='jet')
            #
            #     print(f"Saved input image to {input_save_path}")
            #     print(f"Saved label image to {label_save_path}")
            #     print(f"Saved prediction image to {pred_save_path}")
            #     raise Exception

        metrics.update(preds, labels)
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in
                             images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


# @torch.no_grad()
# def evaluate_msf(model, dataloader, device, scales, flip):
#     model.eval()
#
#     n_classes = dataloader.dataset.n_classes
#     metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
#
#     for images, labels in tqdm(dataloader):
#         labels = labels.to(device)
#         B, H, W = labels.shape
#         scaled_logits = torch.zeros(B, n_classes, H, W).to(device)
#
#         for scale in scales:
#             new_H, new_W = int(scale * H), int(scale * W)
#             new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
#             scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in
#                              images]
#             scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
#             logits = model(scaled_images)
#             logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
#             scaled_logits += logits.softmax(dim=1)
#
#             if flip:
#                 scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
#                 logits = model(scaled_images)
#                 logits = torch.flip(logits, dims=(3,))
#                 logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
#                 scaled_logits += logits.softmax(dim=1)
#
#         metrics.update(scaled_logits, labels)
#
#     acc, macc = metrics.compute_pixel_acc()
#     f1, mf1 = metrics.compute_f1()
#     ious, miou = metrics.compute_iou()
#     return acc, macc, f1, mf1, ious, miou

def main(cfg):
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None]  # all
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")
    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))
    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)
        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        # msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu')['model_state_dict'])  ######
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))  ######
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'],
                                pin_memory=False, sampler=sampler_val)
        if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'],
                                                              eval_cfg['MSF']['FLIP'])
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)
            table = {
                'Class': list(dataset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(eval_cfg['MODEL_PATH']))
        with open(eval_path, 'a+') as f:
            f.writelines(eval_cfg['MODEL_PATH'])
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)