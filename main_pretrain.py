"""
Main function to pretrain RetiZero model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
import argparse

from clip_modules.pretraining.data.dataloader import get_loader
from clip_modules.pretraining.data.transforms import augmentations_pretraining
from clip_modules.modeling.model import CLIPRModel
from config import *


def process(args):

    # Set data for training
    datalaoders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # Init RetiZero model
    model = CLIPRModel(vision_type=args.architecture,
                       out_path=args.out_path, from_checkpoint=False, vision_pretrained=True,
                       R=8
                  )

    # Training
    model.fit(datalaoders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining)


def main():
    parser = argparse.ArgumentParser()
    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)
    parser.add_argument('--out_path', default="./ModelSaved/", help='output path')

    # Training options
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--store_num', default=1, type=int)

    # Architecture and pretrained weights options
    parser.add_argument('--architecture', default='lora', help='lora -- based RETFound')

    # Resources
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()
    process(args=args)




if __name__ == "__main__":
    main()