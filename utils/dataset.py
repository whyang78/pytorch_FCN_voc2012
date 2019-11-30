import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class voc_dataset(Dataset):
    def __init__(self,root,transfrom=None,crop_size=(256,256)):
        self.images_path=os.path.join(root,'images')
        self.labels_path=os.path.join(root,'labels')
        labels=[] # 去除图片大小小于crop size的图片  注意：PIL image：W * H
        for i in os.listdir(self.labels_path):
            im=os.path.join(self.labels_path,i)
            if Image.open(im).size[1] >=crop_size[0] and Image.open(im).size[0] >= crop_size[1]:
                labels.append(i)

        self.filenames = [i.split('.')[0] for i in labels]
        self.filenames.sort()

        self.transform=transfrom
        self.crop_size=crop_size

    def __getitem__(self, index):
        filename=self.filenames[index]
        image=Image.open(os.path.join(self.images_path,filename+'.jpg')).convert('RGB')
        label=Image.open(os.path.join(self.labels_path,filename+'.png')).convert('RGB')

        if self.transform is not None:
            image,label=self.transform(image,label,self.crop_size)

        return image,label

    def __len__(self):
        return len(self.filenames)



