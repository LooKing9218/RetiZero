import torch
import shutil
import random
import imgaug as ia
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn


def save_checkpoint_epoch(state,
                          epoch,is_best,checkpoint_path,stage="val",
                          filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}.pth.tar'.format(
            stage,(epoch + 1))))


def setup_seed(seed=1234):
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed_all()为多个GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    ia.seed(seed)

    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True