"""
Dataset and Dataloader preparation for vision-language pre-training, Fine-tuning
"""
import torch
import pandas as pd
import os
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, dataframe_csv, data_path,transform=None):
        self.transform = transform
        self.data_path = data_path

        self.df = pd.read_csv(dataframe_csv)

        # load studies and study to text mapping
        self.filenames, self.path2sent = [], {}
        for index, row in self.df.iterrows():
            content = row.to_list()
            self.filenames.append(content[0])
            self.path2sent[content[0]]=str(content[1])



    def get_caption(self, path):
        text = str(self.path2sent[path])

        Sent = "A fundus photograph of {}".format(text)

        return Sent

    def get_imgs(self, img_path, transform=None):
        img = Image.open(str(img_path)).convert('RGB')
        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]
        image_file = os.path.join(self.data_path, key)

        imgs = self.get_imgs(image_file, self.transform)

        # randomly select a sentence
        sentens = self.get_caption(key)

        return {"image":imgs,"report":sentens}

    def __len__(self):
        return len(self.filenames)




def get_loader(dataframes_path, data_root_path,  batch_size=8, num_workers=0):

    """
    Dataloaders generation for vision-language pretraining. Read all dataframes from assembly model and combines
    them into a unified dataframe. Also, a dataloader is conditioned for training.
    """

    # Prepare data sample pre-processing transforms
    transforms_proce = Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Assembly dataframes into a combined data structure
    print("Setting assebly data...")
    dataframe_csv = dataframes_path


    # Set data

    train_dataset = MultimodalPretrainingDataset(dataframe_csv,data_path=data_root_path,transform=transforms_proce)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Set dataloaders in dict
    datalaoders = {"train": train_loader}

    return datalaoders

class DatasetFinetuing(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, IsTrain):
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        if IsTrain:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        super().__init__()

    def get_imgs(self, img_path, transform=None):
        img = Image.open(str(img_path)).convert('RGB')
        img = transform(img)

        return img

    def __getitem__(self, index):
        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]
        return imgs, labels

    def __len__(self):
        return len(self.filenames)
