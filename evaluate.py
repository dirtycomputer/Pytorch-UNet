import torch
import torch.nn.functional as F
from tqdm import tqdm

# 引入多类别dice系数和单类别dice系数
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    # 将模型设置为eval模式 相当于self.train(False)
    net.eval()
    # 验证集的batch数
    num_val_batches = len(dataloader)
    # dice评分初始化0
    dice_score = 0

    # 在验证集上迭代
    for batch in tqdm(dataloader, ascii=True, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        # 提取image和mask_true
        image, mask_true = batch['image'], batch['mask']
        # 将image和mask_true以特定数据类型移动到正确的设备上
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # n_classes是类别 比如只想区分 单个实例和背景 那么n_classes=2
        # mask_true重新排列后是(1,n_classes,H,W)
        if(device.type == 'cuda'):
            mask_true = F.one_hot(mask_true, net.module.n_classes).permute(0, 3, 1, 2).float()
        else:
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        # evaluate不需要更新梯度 如果确定不用backward那么用no_grad会减少内存的计算消耗
        # no_grad这种模式下，即使输入具有requires_grad=True，每次计算的结果也会有 requires_grad=False
        with torch.no_grad():
            # 将验证集输入模型中做mask预测
            mask_pred = net(image)
            if(device.type == 'cuda'):
                # 如果n_classes=1
                if net.module.n_classes == 1:
                    #以sigmoid后0.5为阈值二分类
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # 计算dice系数(越大越相似)
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # TODO 多分类
                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            else:
                # 如果n_classes=1
                if net.n_classes == 1:
                    # 以sigmoid后0.5为阈值二分类
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # 计算dice系数(越大越相似)
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # TODO 多分类
                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                        reduce_batch_first=False)

    # 将模型置为train模式
    net.train()

    # 无验证集 dice = 0
    if num_val_batches == 0:
        return dice_score

    # 求各个batch的平均dice score
    return dice_score / num_val_batches
