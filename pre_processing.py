import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

scale = 512
def dye(img, color="red"):
    # get index of channel
    idx = color_dict[color]
    rgb_list = [0,1,2]
    rgb_list.remove(idx)

    # normalize the image
    img[:, :, idx] = img[:, :, idx]/np.max(img[:, :, idx])*255
    # img[:, :, idx] = cv2.equalizeHist(img[:, :, idx])

    img[:, :, rgb_list[0]] = 0
    img[:, :, rgb_list[1]] = 0

    return img

top_path = "./data/"+str(scale)+"/TimePoint_1"
# path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\example"
color_dict = {"red": 2, "green": 1, "blue": 0}


# get the path of each images
files_name = os.listdir(top_path)
files_path = [os.path.join(top_path, i) for i in files_name]
# get rid of the description file
files_path = files_path[1:]

# read and dye the grayscale image
for i in tqdm(range(0, len(files_name)//4)):
    # read a image to a grayscale
    img_1 = cv2.imread(files_path[4*i+1], cv2.IMREAD_COLOR)
    img_2 = cv2.imread(files_path[4*i+2], cv2.IMREAD_COLOR)
    img_3 = cv2.imread(files_path[4*i+3], cv2.IMREAD_COLOR)

    img_1 = dye(img_1, "blue")
    img_2 = dye(img_2, "green")
    img_3 = dye(img_3, "red")

    img_merge = img_1 + img_2 + img_3
    if not os.path.exists("data/overlay"+str(scale)):
        os.mkdir("data/overlay"+str(scale))
    cv2.imwrite("data/overlay"+str(scale)+"/"+str(i)+"_overlay.tif", img_merge)

# img = cv2.imread(files_path[0], cv2.IMREAD_COLOR)
# # img = dye(img, "green")
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img[:,:,2] = clahe.apply(img[:,:,2])
# img[:,:,0] = 0
# img[:,:,1] = img[:,:,2]
# show the result
# cv2.imshow("merge", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
