import argparse

import torch
import numpy as np
import sys
import os
import dlib
import cv2
sys.path.insert(0, "./hairstyle_transfer")
from pathlib import Path
import argparse
import torchvision
from utils.drive import open_url
from utils.shape_predictor import align_face
import PIL


from PIL import Image


from models.Embedding import Embedding
from models.Alignment import Alignment
from models.Blending import Blending

def hairstyle_transfer(face_path, hair_path):
  
    face_name = face_path.split('/')
    face_name = face_name[-1]
    face_name = face_name[:-4]
    
    hair_name = hair_path.split('/')
    hair_name = hair_name[-1]
    hair_name = hair_name[:-4]
    

    parser = argparse.ArgumentParser(description='Barbershop')



    # # parser.add_argument('-unprocessed_dir', type=str, default='../images/unprocessed', help='directory with unprocessed images')
    # parser.add_argument('-output_dir_face', type=str, default='../images/face', help='output directory')

    # parser.add_argument('-output_size', type=int, default=1024, help='size to downscale the input images to, must be power of 2')
    # # parser.add_argument('-seed', type=int, help='manual seed to use')
    # parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

    # ###############
    # parser.add_argument('-inter_method', type=str, default='bicubic')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='./images/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='./results/output',
                        help='The directory to save the latent codes and inversion images')
    parser.add_argument('--im_path1', type=str, default= face_name + '.png', help='Identity image')
    parser.add_argument('--im_path2', type=str, default= hair_name + '.png', help='Structure image')
    parser.add_argument('--im_path3', type=str, default= hair_name + '.png', help='Appearance image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="./checkpoint/Barbershop/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')
    parser.add_argument('--seg_ckpt', type=str, default='./checkpoint/Barbershop/seg.pth')

    # Embedding loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
    parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
    parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
    parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')

    # Alignment loss options
    parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
    parser.add_argument('--align_steps1', type=int, default=140, help='')
    parser.add_argument('--align_steps2', type=int, default=100, help='')

    # Blend loss options
    parser.add_argument('--face_lambda', type=float, default=1.0, help='')
    parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
    parser.add_argument('--blend_steps', type=int, default=400, help='')


    args = parser.parse_args()


    ii2s = Embedding(args)
    #
    # ##### Option 1: input folder
    # # ii2s.invert_images_in_W()
    # # ii2s.invert_images_in_FS()

    # ##### Option 2: image path
    # # ii2s.invert_images_in_W('input/face/28.png')
    # # ii2s.invert_images_in_FS('input/face/28.png')
    #
    ##### Option 3: image path list

    # im_path1 = 'input/face/90.png'
    # im_path2 = 'input/face/15.png'
    # im_path3 = 'input/face/117.png'

    # im_path1 = os.path.join(args.input_dir, args.im_path1)
    # im_path2 = os.path.join(args.input_dir, args.im_path2)
    # im_path3 = os.path.join(args.input_dir, args.im_path3)
    im_path1 = face_path
    im_path2 = hair_path
    im_path3 = hair_path

    im_set = {im_path1, im_path2, im_path3}
    ii2s.invert_images_in_W([*im_set])
    ii2s.invert_images_in_FS([*im_set])

    align = Alignment(args)
    res = align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)
    if im_path2 != im_path3:
        align.align_images(im_path1, im_path3, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)

    blend = Blending(args)
    res_all = blend.blend_images(im_path1, im_path2, im_path3, sign=args.sign)
    # res_cv =cv2.cvtColor(np.asarray(res),cv2.COLOR_RGB2BGR)
    res_all_cv =cv2.cvtColor(np.asarray(res_all),cv2.COLOR_RGB2BGR)
    cv.imshow('dadada', res_all_cv)
    input()
    # cv2.imwrite(r"../results/output/a.png",res_cv)
    cv2.imwrite("../results/output/aaaaaa.png",res_all_cv)
    # cv2.imshow('res', res_cv)


    return res_all_cv






if __name__ == "__main__":

    face_path = '../images/face/face.png'

    hair_path = '../images/face/hair.png'

    a = hairstyle_transfer(face_path, hair_path)