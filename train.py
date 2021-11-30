import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'



def train_net(net,  # 模型
              device,  # 设备
              dataname,
              epochs,  # 训练轮数
              batch_size,  # batch大小
              learning_rate,  # 学习率
              val_percent,  # 验证集所占数据集比例 #TODO 0.1
              save_checkpoint: bool = True,  # 是否保存断点
              img_scale: float = 0.5,  # 图像缩放
              amp: bool = False):  # 自动混合精度 https://zhuanlan.zhihu.com/p/348554267

    dir_img = Path(os.path.join('./', dataname, 'imgs'))
    dir_mask = Path(os.path.join('./', dataname, 'masks'))
    dir_checkpoint = Path(os.path.join('./', dataname, 'checkpoints/'))
    # 1. 创建数据集
    # TODO
    # try:
    #     #汽车数据集
    #     print("using dataset :CarvanaDataset")
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    print("using dataset :{}".format(dataname))
    # 普通数据集
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix=args.mask_suffix)

    # 2. 将数据集按val_percent参数比例划分为train_set和val_set
    # n_val验证集数量
    # n_train训练集数量
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建数据加载器
    # num_worker是batch加载到RAM(内存开销大，速度快)
    # load_args为train和val的共同参数
    # 内存充足时设置锁页内存更好支持GPU
    # droplast动态图不能整除batchsize不会保错，主动舍弃
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # TODO
    # 配置wandb(weight and bias)
    # project添加项目名称
    experiment = wandb.init(project=dataname, resume='allow', anonymous='must')
    # 添加wandb参数
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    # 日志信息列表
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. 设置优化器，损失函数，学习率变化器以及损失自动混合精度
    # 运用RMSprop算法 学习率 权重衰减 动量
    if(device.type == 'cuda'):
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-12)
        #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-12, momentum=0.6)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-12)
        #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-12, momentum=0.6)
    # 学习率变化器 ReduceLROnPlateau当loss不变时改进时降低学习率
    # mode为max 因为目标是最大化Dice score
    # patience 没有改进的epoch数量 (例:在两次epoch都没有loss改进时,降低学习率)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    # GradScaler 防止下溢(损失小幅值)，"梯度缩放"将网络的损失乘以一个比例因子
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 全局步数统计
    global_step = 0

    # 5. 开始训练
    for epoch in range(epochs):

        # 训练
        net.train()
        # 此次epoch的loss
        epoch_loss = 0
        # tqdm包装输出信息
        # total进度条总长度
        # desc添加描述信息
        # as重命名为pbar(process_bar)
        # epochs为epoch总数 epoch+1为当前epoch数
        with tqdm(total=n_train, ascii=True, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # 从dataloader中取数据
            # 这里的batch是个dict字典 image：图片 mask：遮罩
            for batch in train_loader:
                # TODO
                # images的形状为(1,C,H,W)
                images = batch['image']
                # true_masks的形状为(1,H,W) 没有channel
                true_masks = batch['mask']
                # 检查channel数是否为应输入的channel数(n_channels)
                # TODO mudule
                if(device.type == 'cuda'):
                    assert images.shape[1] == net.module.n_channels, \
                        f'Network has been defined with {net.module.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                else:
                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                # 将images和true_masks放到device上(GPU)
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # TODO
                with torch.cuda.amp.autocast(enabled=amp):
                    # TODO
                    # mask_pred形状为(B, 2, H, W)
                    masks_pred = net(images)
                    # loss为交叉熵+dice
                    if(device.type == 'cuda'):
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.module.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                    else:
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                # 将所有优化的torch.Tensors的梯度设置为零
                # TODO
                optimizer.zero_grad(set_to_none=True)
                # 反向传播
                grad_scaler.scale(loss).backward()
                # step优化一步
                grad_scaler.step(optimizer)
                # 更新
                grad_scaler.update()
                # 进度条更新
                pbar.update(images.shape[0])
                # 总步数+1
                global_step += 1
                # 累计epoch loss
                epoch_loss += loss.item()
                # 每一步的日志信息
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                # 进度条打印每一步loss
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round

                # DEBUG division_step = 1
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # TODO
                        # wandb绘制直方图
                        histograms = {}
                        if (device.type == 'cuda'):
                            for tag, value in net.module.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        else:
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())



                        # 将验证集全部验证 计算val_score
                        val_score = evaluate(net, val_loader, device)
                        # scheduler的step相当于计数 一般是用在epoch循环的(可以点开看源码) optimizer的step通常是batch为单位的
                        scheduler.step(val_score)
                        # 输出dice score日志信息
                        logging.info('Validation Dice score: {}'.format(val_score))
                        # 将日志信息输出到wandb
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            # parents 如果父目录不存在，是否创建父目录
            # exist_ok 只有在目录不存在时创建目录，目录已存在时不会抛出异常
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # 每个epoch保存模型
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    # 获取自定义参数信息
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataname', '-src', metavar='S', type=str, default='dataset_CVC-ClinicDB', help='Dataset root path')
    # parser.add_argument('--dataname', '-src', metavar='S', type=str, default='dataset_CarvanaDataset', help='Dataset root path')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=20, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.9, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,  # TODO 0.1十分之一的验证集
                        help='Percent of the data that is used as validation 0.1 = 10%')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--mask_suffix', '-suf', type=str, default='', help='Mask image suffix')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # 日志输出为INFO级别
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # 是否使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # 自定义输入通道数 类别数 是否双线性插值
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    # 上采样方法选择 1.双线性插值 2.转置卷积
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # 导入已有模型
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # 将模型放到GPU(CPU)上
    if (device.type == 'cuda') & (torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net, device_ids=[0]).to(device)
    net.to(device)

    # 不被中断的情况下训练，否则保存中断模型
    try:
        train_net(net=net,
                  dataname=args.dataname,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
