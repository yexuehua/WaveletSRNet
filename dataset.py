import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import pandas as pd


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".tif"])


def readlinesFromFile(path, datasize):
    print("Load from file %s" % path)
    df = pd.read_csv(path)
    return df["name"].values


def loadFromFile(path, datasize):
    if path is None:
      return None, None
      
    print("Load from file %s" % path)
    df = pd.read_csv(path)
    data = df["overlay256"].values
    label = df["overlay512"].values
    return data, label     


def load_image(lr_file_path,hr_file_path):
    hr_img = Image.open(hr_file_path)
    lr_img = Image.open(lr_file_path)
    return lr_img, hr_img


def my_transfroms(lr,hr,crop_size=(128,128),scale=2):
    """
    Args lr: low resolution Image
         hr: high resolution Image
         crop_size: output size
    return: cropped image alina to the output size
    """
    w, h = lr.size #get the original size
    tw, th = crop_size

    if w == tw and h == th:
        lr_crop = F.crop(lr, 0, 0, w, h)
        hr_crop = F.crop(hr, 0, 0, scale*w, scale*h)
        return F.to_tensor(lr_crop), F.to_tensor(hr_crop)

    i = random.randint(0, h-th)
    j = random.randint(0, w-tw)
    lr_crop = F.crop(lr, i, j, th, tw)
    hr_crop = F.crop(hr, 2*i, 2*j, scale*th, scale*tw)

    return F.to_tensor(lr_crop), F.to_tensor(hr_crop)
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, label_list, root_path):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list
        self.label_filenames = label_list
        self.root_path = root_path
                       
        # self.input_transform = transforms.Compose([
        #
        #     transforms.RandomCrop(128,128),
        #     transforms.ToTensor()
        #                        ])
        # self.target_transform = transforms.Compose([
        #     transforms.RandomCrop(256,256),
        #     transforms.ToTensor()
        #                        ])
        

    def __getitem__(self, index):
        
        lr, hr = load_image(join(self.root_path, self.image_filenames[index]),join(self.root_path, self.label_filenames[index]))

        return my_transfroms(lr,hr)
        return my_transfroms(lr,hr)


    def __len__(self):
        return len(self.image_filenames)
        
        
        
