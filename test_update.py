import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_softmax, softmax_output_dice, tailor_and_concat
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--root', default='path to testing set', type=str)

parser.add_argument('--valid_dir', default='Valid', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='', type=str)

parser.add_argument('--test_date', default='', type=str)

parser.add_argument('--test_file', default='', type=str)

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    model = torch.nn.DataParallel(model).cuda()

    load_file = '/kaggle/input/epoch-79/pytorch/default/1/model_epoch_last.pth'

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        # args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = "/kaggle/working/TransBTS/data/train.txt"
    valid_root = "/kaggle/input/brats2020-pkl/BraTS2020_pkl"
    valid_set = BraTS(valid_list, valid_root, mode='valid')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    dice_wt_list = []
    dice_tc_list = []
    dice_et_list = []


    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True,
                         valid_in_train=True
                         )
    
    print("\n========== CALCULATING MEAN DICE ==========")

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]

            # inference (GIỐNG validate_softmax)
            logit = tailor_and_concat(x, model)
            output = torch.softmax(logit, dim=1)
            output = output.argmax(dim=1).cpu().numpy()[0]
            target = target.cpu().numpy()[0]

            # Dice
            dice_wt, dice_tc, dice_et = softmax_output_dice(output, target)

            dice_wt_list.append(dice_wt)
            dice_tc_list.append(dice_tc)
            dice_et_list.append(dice_et)

            print(f"[{i+1}/{len(valid_loader)}] "
                f"WT: {dice_wt:.4f}, TC: {dice_tc:.4f}, ET: {dice_et:.4f}")
    
    mean_wt = np.mean(dice_wt_list)
    mean_tc = np.mean(dice_tc_list)
    mean_et = np.mean(dice_et_list)
    mean_dice = (mean_wt + mean_tc + mean_et) / 3

    print("\n================ FINAL DICE RESULTS ================")
    print(f"Mean WT Dice: {mean_wt:.4f}")
    print(f"Mean TC Dice: {mean_tc:.4f}")
    print(f"Mean ET Dice: {mean_et:.4f}")
    print(f"Mean Dice   : {mean_dice:.4f}")
    print("===================================================")


    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))



if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


