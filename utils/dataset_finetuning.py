import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
import glob




class CusImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path):
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        self.transform = Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))

        if transform is not None:
            img = transform(img)


        return img


    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        return imgs, labels, key

    def __len__(self):
        return len(self.filenames)

class RetrivalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            self.image_paths += [(os.path.join(category_path, f), category) for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, category = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path, category
