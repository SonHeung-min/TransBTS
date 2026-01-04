import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from predict import tailor_and_concat

cudnn.benchmark = True


# ================= Dice utilities =================
def dice_score(pred, gt, eps=1e-8):
    intersection = np.sum(pred * gt)
    return (2. * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


def compute_brats_dice(pred, gt):
    """
    pred, gt: (H, W, D) with labels {0,1,2,4}
    """
    # Whole Tumor (WT): 1,2,4
    wt_pred = (pred > 0)
    wt_gt = (gt > 0)
    dice_wt = dice_score(wt_pred, wt_gt)

    # Tumor Core (TC): 1,4
    tc_pred = (pred == 1) | (pred == 4)
    tc_gt = (gt == 1) | (gt == 4)
    dice_tc = dice_score(tc_pred, tc_gt)

    # Enhancing Tumor (ET): 4
    et_pred = (pred == 4)
    et_gt = (gt == 4)

    if np.sum(et_gt) == 0:
        dice_et = None   # ET không tồn tại → bỏ qua
    else:
        dice_et = dice_score(et_pred, et_gt)

    return dice_wt, dice_tc, dice_et


# ================= Main evaluation =================
def main():
    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # -------- Load model --------
    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    checkpoint_path = "/kaggle/input/epoch-79/pytorch/default/1/model_epoch_last.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path}")

    # -------- Validation set --------
    valid_list = "/kaggle/working/TransBTS/data/train.txt"
    valid_root = "/kaggle/input/brats2020-pkl/BraTS2020_pkl"

    valid_set = BraTS(valid_list, valid_root, mode='valid')
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validation samples: {len(valid_set)}")

    # -------- Metric containers --------
    dice_wt_list = []
    dice_tc_list = []
    dice_et_list = []

    # -------- Evaluation loop --------
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            x, gt = data
            x = x.cuda(non_blocking=True)
            gt = gt.numpy()[0][..., :155]   # (H,W,D)

            logit = tailor_and_concat(x, model)
            pred = torch.argmax(logit, dim=1).cpu().numpy()[0]

            # Convert label 3 → 4 (BraTS convention)
            pred[pred == 3] = 4

            dice_wt, dice_tc, dice_et = compute_brats_dice(pred, gt)

            dice_wt_list.append(dice_wt)
            dice_tc_list.append(dice_tc)
            if dice_et is not None:
                dice_et_list.append(dice_et)

            print(f"[{i+1:03d}] WT={dice_wt:.4f} | TC={dice_tc:.4f} | ET={'NA' if dice_et is None else f'{dice_et:.4f}'}")

    # -------- Final scores --------
    WT = np.mean(dice_wt_list)
    TC = np.mean(dice_tc_list)
    ET = np.mean(dice_et_list)
    MEAN_DICE = (WT + TC + ET) / 3.0

    print("\n================ FINAL VALIDATION RESULT ================")
    print(f"Whole Tumor (WT) Dice : {WT:.4f}")
    print(f"Tumor Core (TC) Dice  : {TC:.4f}")
    print(f"Enhancing Tumor (ET)  : {ET:.4f}")
    print(f"Mean Dice             : {MEAN_DICE:.4f}")
    print("=========================================================")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
