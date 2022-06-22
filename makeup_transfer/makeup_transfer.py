import cv2
import os
import sys
import argparse

sys.path.insert(0, "./makeup_transfer")

from align_face import extract_head
from face_parse import face_parse
from ssat import ssat

import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def args_parameter():
    parser = argparse.ArgumentParser(description='makeup_transfer')

    # extract head
    parser.add_argument('-unprocessed_dir', type=str, default='./images/src',
                        help='directory with unprocessed images')
    parser.add_argument('-output_dir', type=str, default='./images/head', help='output directory')
    parser.add_argument('-output_size', type=int, default=1024,
                        help='size to downscale the input images to, must be power of 2')
    parser.add_argument('-seed', type=int, help='manual seed to use')
    parser.add_argument('-cache_dir', type=str, default='./checkpoint/face_detect',
                        help='cache directory for model weights')
    parser.add_argument('-inter_method', type=str, default='bicubic')

    # face parse
    parser.add_argument('-model_path', type=str, default='./checkpoint/face_parse/79999_iter.pth')
    parser.add_argument('-save_direct', type=str, default='./images/seg/')

    # makeup transfer (SSAT)
    # data loader related
    parser.add_argument('--dataroot', type=str, default='./images/head/', help='path of data')
    parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
    parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
    parser.add_argument('--semantic_dim', type=int, default=18, help='output_dim')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
    parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
    parser.add_argument('--flip', type=bool, default=False, help='specified if  flipping')
    parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')

    # ouptput related
    parser.add_argument('--name', type=str, default='makeup', help='folder name to save outputs')
    parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    parser.add_argument('--result_dir', type=str, default='./results/makeup_transfer',
                        help='path for saving result images and models')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/SSAT',
                        help='path for saving result images ')

    parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    parser.add_argument('--model_save_freq', type=int, default=100, help='freq (epoch) of saving models')

    # training related
    parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    parser.add_argument('--dis_norm', type=str, default='None',
                        help='normalization layer in discriminator [None, Instance]')
    parser.add_argument('--dis_spectral_norm', type=bool, default=True,
                        help='use spectral normalization in discriminator')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    parser.add_argument('--n_ep', type=int, default=600, help='number of epochs')  # 400 * d_iter
    parser.add_argument('--n_ep_decay', type=int, default=300,
                        help='epoch start decay learning rate, set -1 if no decay')  # 200 * d_iter
    parser.add_argument('--resume', type=str, default=None,
                        help='specified the dir of saved models for resume the training')
    parser.add_argument('--num_residule_block', type=int, default=4, help='num_residule_block')
    parser.add_argument('--lr', type=float, default=0.0002, help='lr')
    parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')

    args = parser.parse_args()

    return args


def makeup_transfer(non_makeup_image=None, makeup_image=None):
    args = args_parameter()

    # 全身照分割出人头部分
    non_makeup_head = extract_head(args, non_makeup_image, flag='non_makeup')
    makeup_head = extract_head(args, makeup_image, flag='makeup')
    print("finish extract head")

    # 解析人脸属性
    non_makeup_seg_image = face_parse(img=non_makeup_head, model_path=args.model_path,
                                      save_path=args.save_direct + 'non_makeup/' + 'non_makeup.png')
    # cv2.imwrite("mask_image.png", non_makeup_mask_image)
    makeup_seg_image = face_parse(img=makeup_head, model_path=args.model_path,
                                  save_path=args.save_direct + 'makeup/' + 'makeup.png')
    print("finish parse face")
    # cv2.imwrite("mask_image.png", makeup_seg_image)

    # SSAT从文件夹读取图片,目录为head和seg
    result_img = ssat(args)
    # cv2.imwrite("result_img.png", result_img)

    return result_img


if __name__ == '__main__':
    non_makeup_image = cv2.imread('../images/src/non_makeup/target_image.jpg')
    makeup_image = cv2.imread('../images/src/makeup/vFG112.png')
    makeup_transfer(non_makeup_image, makeup_image)
