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
    label = df["overlay512"]
    return data, label     
    
def load_image(hr_file_path,lr_file_path):
    hr_img = Image.open(hr_file_path)
    lr_img = Image.open(lr_file_path)
      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list        
        self.root_path = root_path
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])
        

    def __getitem__(self, index):
          
        lr, hr = load_image(join(self.root_path, self.image_filenames[index]),
                            join(self.root_path, self.image_filenames[index]))
        
        
        input = self.input_transform(lr)
        target = self.input_transform(hr)
        
        return input, target


    def __len__(self):
        return len(self.image_filenames)
        
        
        
