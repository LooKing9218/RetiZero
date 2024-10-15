import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import argparse
import torch
import tqdm
from sklearn import metrics
import torch.nn as nn
import utils.utils as u
from clip_modules import CLIPRModel
from utils.dataset_finetuning import CusImageDataset
from torch.utils.data import DataLoader
import glob

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def val(val_dataloader, model, epoch, args, mode):
    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()
    labels = []
    outputs = []

    predictions = []
    gts = []
    num_total = 0
    tbar = tqdm.tqdm(val_dataloader, desc='\r')

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].cuda()
            cls_label = img_data_list[1].long().cuda()
            pred = model.forward(Fundus_img)

            pred = torch.softmax(pred, dim=1)

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.num_classes).cuda().scatter_(1, cls_label.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(cls_label.cpu().detach().float().numpy()[idx])
    Acc = metrics.accuracy_score(gts, predictions)
    if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
        os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))

    with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
        Txt.write("Epoch {}: {} == Acc: {}.\n".format(
            epoch,mode, round(Acc,6)
        ))
    print("Epoch {}: {} == Acc: {}.\n".format(
            epoch,mode,round(Acc,6)
        ))
    torch.cuda.empty_cache()
    return Acc

import numpy as np
def train(train_loader, val_loader, test_loader, model, optimizer, criterion,args):
    step = 0
    model = model.cuda()
    best_Acc = 0.0
    for epoch in range(1, args.num_epochs+1):
        labels = []
        outputs = []
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description('Epoch %d, lr %f' % (epoch, args.lr))
        loss_record = []
        train_loss = 0.0
        for i, img_data_list in enumerate(train_loader):
            Fundus_img = img_data_list[0].cuda()
            cls_label = img_data_list[1].long().cuda()
            optimizer.zero_grad()
            pretict = model(Fundus_img)
            loss_CE = criterion(pretict, cls_label)
            loss = loss_CE
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            one_hot = torch.zeros(pretict.size(0), args.num_classes).cuda().scatter_(1, cls_label.unsqueeze(1), 1)
            pretict = torch.softmax(pretict, dim=1)
            for idx_data in range(pretict.size(0)):
                outputs.append(pretict.cpu().detach().float().numpy()[idx_data])
                labels.append(one_hot.cpu().detach().float().numpy()[idx_data])
            loss_record.append(loss.item())
        tq.close()
        torch.cuda.empty_cache()
        loss_train_mean = np.mean(loss_record)

        del labels,outputs

        print('loss for train : {}'.format(loss_train_mean))
        if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
            os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
        with open(os.path.join(args.save_model_path, "{}/{}_Metric.txt".format(args.net_work,args.net_work)), 'a+') as f:
            f.write('EPOCH:' + str(epoch) + ',')

        mean_ACC = val(val_loader, model, epoch,args,mode="val")
        is_best = mean_ACC > best_Acc
        if is_best:
            best_Acc = max(best_Acc, mean_ACC)
            mean_ACC_test = val(test_loader, model, epoch,
                                               args, mode="test")

            checkpoint_dir = os.path.join(args.save_model_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            all_previous_models = glob.glob(checkpoint_dir+"/*.pth.tar")
            if len(all_previous_models):
                for pre_model in all_previous_models:
                    os.remove(pre_model)
            print('===> Saving models...')
            u.save_checkpoint_epoch({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mean_ACC': mean_ACC,
            }, epoch, True, checkpoint_dir, stage="Test",
                filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))

class Model_Finetuing(torch.nn.Module):
    def __init__(self,model_name,class_num,weight_path):
        super().__init__()

        Model_Pretrained = CLIPRModel(vision_type=model_name, from_checkpoint=True,
                           weights_path=weight_path, R=8)
        self.img_encoder = Model_Pretrained.vision_model.model
        for para in self.img_encoder.parameters():
            para.requires_grad = False

        if model_name == "lora":
            feature_dim = 1024
        else:
            feature_dim = 2048
        self.classifier = torch.nn.Linear(feature_dim, class_num,
                                  bias=True)

    def forward(self,x):
        x_features = self.img_encoder(x)
        out = self.classifier(x_features)
        return out


def main(args):
    # bulid model
    weight_path = "./Pretrained/RetiZero.pth"
    model = Model_Finetuing(model_name="lora",class_num=args.num_classes,weight_path=weight_path)
    # get datamodule
    data_path = "./Dataset/Dataset_DonwStream"
    csv_path = "./CSV_Dir"
    train_csv = os.path.join(csv_path, "train.csv")
    valid_csv = os.path.join(csv_path, "valid.csv")
    test_csv = os.path.join(csv_path, "test.csv")
    train_dataset = CusImageDataset(
        csv_file=train_csv,
        data_path=data_path)
    train_loader = DataLoader(
            train_dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    val_dataset = CusImageDataset(csv_file=valid_csv,data_path=data_path)
    val_loader =  DataLoader(
            val_dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    test_dataset = CusImageDataset(csv_file=test_csv,data_path=data_path)
    test_loader = DataLoader(
            test_dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()
    train(train_loader, val_loader,test_loader, model, optimizer, criterion,args)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classes",default=15
    )

    parser.add_argument(
        "--num_workers",default=8
    )

    parser.add_argument(
        "--save_model_path",default="./Model_saved"
    )

    parser.add_argument(
        "--net_work",default="lora"
    )

    parser.add_argument(
        "--num_epochs",default=100
    )

    parser.add_argument(
        "--batch_size",default=64
    )

    parser.add_argument(
        "--lr",
        default=5e-4
    )
    return parser

if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = get_parser()
    args = parser.parse_args()
    args.seed = 1234
    u.setup_seed(args.seed)

    main(args)