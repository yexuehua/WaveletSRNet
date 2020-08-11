from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
from math import log10
import torchvision
from torchsummary import summary
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enables test during training')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--num_layers_res', type=int, help='number of the layers in residual block', default=2)
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=10)
parser.add_argument('--trainfiles', default="train.csv", type=str, help='the list of training files')
parser.add_argument('--dataroot', default="data/overlay", type=str, help='path to dataset')
parser.add_argument('--testfiles', default="test.csv", type=str, help='the list of training files')
parser.add_argument('--testroot', default="data/overlay", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data',default=80)
parser.add_argument('--testsize', type=int, help='number of testing data',default=20)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=64, help='test batch size')
parser.add_argument('--save_iter', type=int, default=10, help='the interval iterations for saving models')
parser.add_argument('--test_iter', type=int, default=500, help='the interval iterations for testing')
parser.add_argument('--cdim', type=int, default=3, help='the channel-size  of the input image to network')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_height', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--upscale', type=int, default=2, help='the depth of wavelet tranform')
parser.add_argument('--scale_back', action='store_true', help='enables scale_back')
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='results/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)


    device = torch.device('cuda')
    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    ngpu = int(opt.ngpu)   
    nc = opt.cdim
    mag = int(math.pow(2, opt.upscale))
    groups = mag ** 2
    if opt.scale_back:      
      is_scale_back = True
    else:      
      is_scale_back = False
    
    #--------------build models--------------------------
    srnet = NetSR(1, num_layers_res=opt.num_layers_res)
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            pretrained_dict = weights['model'].state_dict()
            model_dict = srnet.state_dict()
            # print(model_dict)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            srnet.load_state_dict(model_dict)
            # srnet.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    print(srnet)
    #summary(srnet,(3,256,256))
    
    wavelet_dec = WaveletTransform(scale=1, dec=True)
    wavelet_rec = WaveletTransform(scale=1, dec=False)          
    #summary(wavelet_dec,(3,256,256))
    #summary(wavelet_rec,(48,64,64))

    criterion_m = nn.MSELoss(size_average=True)

    if opt.cuda:
        srnet = srnet.to("cuda:0")      
        wavelet_dec = wavelet_dec.to(device)
        wavelet_rec = wavelet_rec.to(device)
        criterion_m = criterion_m.to(device)
     
    
    optimizer_sr = optim.Adam(srnet.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0005)
    
    #-----------------load dataset--------------------------
    train_list, train_label_list = loadFromFile(opt.trainfiles, opt.trainsize)
    train_set = ImageDatasetFromFile(train_list, train_label_list, opt.dataroot)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    
    test_list, test_label_lists = loadFromFile(opt.testfiles, opt.testsize)
    test_set = ImageDatasetFromFile(test_list, test_label_lists, opt.testroot)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

    start_time = time.time()
    srnet.train()
    print("-------start training.......--------")
    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        if epoch%opt.save_iter == 0:
            save_checkpoint(srnet, epoch, 0, 'sr_')
        
        for iteration, batch in enumerate(train_data_loader, 0):
            #--------------test-------------
            if iteration % opt.test_iter is 0 and opt.test:
                srnet.eval()
                avg_psnr = 0
                for titer, batch in enumerate(test_data_loader,0):
                    input, target = Variable(batch[0]), Variable(batch[1])
                    if opt.cuda:
                        input = input.to(device)
                        target = target.to(device)

                    wavelets = forward_parallel(srnet, input, opt.ngpu)                    
                    prediction = wavelet_rec(wavelets)
                    mse = criterion_m(prediction, target)
                    psnr = 10 * log10(1 / (mse.item()) )
                    avg_psnr += psnr

                    save_images(input, "input_Epoch_{:03d}_{:02d}_o.jpg".format(epoch, titer),
                                path=opt.outf, nrow=opt.nrow)
                    save_images(target, "target_Epoch_{:03d}_{:02d}_o.jpg".format(epoch, titer),
                                path=opt.outf, nrow=opt.nrow)
                    save_images(prediction, "pred_Epoch_{:03d}_{:02d}_o.jpg".format(epoch, titer),
                                path=opt.outf, nrow=opt.nrow)
                    
                    
                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))
                srnet.train()
              
            #--------------train------------
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)          
            if opt.cuda:
              input = input.to(device)
              target = target.to(device)
              
            target_wavelets = wavelet_dec(target)
          
            batch_size = target.size(0)
            wavelets_lr = target_wavelets[:,0:3,:,:]
            wavelets_sr = target_wavelets[:,3:,:,:]
            
            wavelets_predict = forward_parallel(srnet, input, opt.ngpu)            
            img_predict = wavelet_rec(wavelets_predict)
            
            
            loss_lr = loss_MSE(wavelets_predict[:,0:3,:,:], wavelets_lr, opt.mse_avg)
            loss_sr = loss_MSE(wavelets_predict[:,3:,:,:], wavelets_sr, opt.mse_avg)
            loss_textures = loss_Textures(wavelets_predict[:,3:,:,:], wavelets_sr)
            loss_img = loss_MSE(img_predict, target, opt.mse_avg)
            
            loss = loss_sr.mul(0.99) + loss_lr.mul(0.01) + loss_img.mul(0.1) + loss_textures.mul(1)           
            
            optimizer_sr.zero_grad()    
            loss.backward()                       
            optimizer_sr.step()
            
            info = "===> Epoch[{}]({}/{}): time: {:4.4f}:".format(epoch, iteration, len(train_data_loader), time.time()-start_time)
            info += "Rec: {:.4f}, {:.4f}, {:.4f},Texture:{:.4f}".format(loss_lr.item(), loss_sr.item(), loss_img.item(), loss_textures.item())            
                          
            print(info)
             

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def save_images(images, name, path, nrow=10):   
  #print(images.size())
  img = images.to("cpu")
  im = img.data.numpy().astype(np.float32)
  #print(im.shape)       
  im = im.transpose(0,2,3,1)
  imsave(im, [nrow, int(math.ceil(im.shape[0]/float(nrow)))], os.path.join(path, name) )
  
def merge(images, size):
  #print(images.shape())
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  #print(img)
  for idx, image in enumerate(images):
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  img = merge(images, size)
  # print(img) 
  return cv2.imwrite(path, img)

if __name__ == "__main__":
    main()    
