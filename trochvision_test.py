from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as F

def my_transfroms(lr,hr,crop_size=(128,128),scale=2):
    w,h = lr.size
    tw,th = crop_size
    if w==tw and h==th:
        return F.crop(lr, 0, 0, w, h), F.crop(hr, 0, 0, scale*w, scale*h)
    i = random.randint(0, h-th)
    j = random.randint(0, w-tw)
    return F.crop(lr, i, j, th, tw), F.crop(hr, 2*i, 2*j, scale*th, scale*tw)


top_path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data\overlay256\0_overlay.tif"
h_top_path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data\overlay512\0_overlay.tif"
lr_img = Image.open(top_path)
hr_img = Image.open(h_top_path)
lr,hr = my_transfroms(lr_img,hr_img)
# trans = transforms.Compose([transforms.RandomCrop(1500,padding=True,pad_if_needed=True)])
# out = trans(img)
print(lr.size,hr.size)
lr.show()
hr.show()