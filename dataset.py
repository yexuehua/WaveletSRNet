import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
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
      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, label_list, root_path):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list
        self.label_filenames = label_list
        self.root_path = root_path
                       
        self.input_transform = transforms.Compose([
                                   
            transforms.RandomCrop(128,128),
            transforms.ToTensor()                                                                      
                               ])
        self.target_transform = transforms.Compose([
            transforms.RandomCrop(256,256),
            transforms.ToTensor()
                               ])
        

    def __getitem__(self, index):
        
        lr, hr = load_image(join(self.root_path, self.image_filenames[index]),join(self.root_path, self.label_filenames[index]))
        
        
        input = self.input_transform(lr)
        target = self.target_transform(hr)
        
        return input, target


    def __len__(self):
        return len(self.image_filenames)
        
        
        
