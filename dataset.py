import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".tif"])
    
def readlinesFromFile(path, datasize):
    print("Load from file %s" % path)
    f=open(path)
    data=[]    
    for idx in xrange(0, datasize):
      line = f.readline()
      data.append(line)      
    
    f.close()  
    return data  
    
def loadFromFile(path, datasize):
    if path is None:
      return None, None
      
    print("Load from file %s" % path)
    f=open(path)
    data=[]
    label=[]
    for idx in xrange(0, datasize):
      line = f.readline().split()
      data.append(line[0])         
      label.append(line[1])
       
    f.close()  
    return data, label     
    
def load_image(hr_file_path,lr_file_path):
    hr_img = Image.open(hr_file_path)
    lr_img = Image.open(lr_file_path)
      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path, input_height=128, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True,
              is_gray=False, upscale=1.0, is_scale_back=False):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list        
        self.upscale = upscale
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_scale_back = is_scale_back
        self.is_gray = is_gray
                       
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
        
        
        