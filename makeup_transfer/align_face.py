import os
import numpy as np
import cv2
import dlib
from pathlib import Path
import argparse
import torchvision
from utils.drive import open_url
from utils.shape_predictor import align_face, align_face_from_direct
import PIL
import shutil


def extract_head(args, img, flag='non_makeup'):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir + '/' + flag)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Shape Predictor")
    f = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
    predictor = dlib.shape_predictor(f)

    faces = align_face(img, predictor)

    for i, face in enumerate(faces):
        if args.output_size:
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
        if len(faces) > 1:
            face.save(Path(args.output_dir + '/' + flag) / (flag + f"_{i}.png"))
            print("ERROR: more than one person detected!")
        else:
            face.save(Path(args.output_dir + '/' + flag) / (flag + f".png"))

    head = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2BGR)

    return head


def extract_head_from_direct(args):

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Shape Predictor")
    f = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
    predictor = dlib.shape_predictor(f)

    for dirs in os.scandir(args.unprocessed_dir):

        output_dir = Path(args.output_dir + '/' + dirs.name)
        output_dir.mkdir(parents=True, exist_ok=True)

        for im in Path(dirs.path).glob("*.*"):

            faces = align_face_from_direct(str(im), predictor)

            for i, face in enumerate(faces):
                if args.output_size:
                    factor = 1024//args.output_size
                    assert args.output_size*factor == 1024
                    face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
                    face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
                    face = torchvision.transforms.ToPILImage()(face_tensor_lr)
                    if factor != 1:
                        face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
                if len(faces) > 1:
                    face.save(Path(args.output_dir + '/' + dirs.name) / (im.stem+f"_{i}.png"))
                else:
                    face.save(Path(args.output_dir + '/' + dirs.name) / (im.stem + f".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align_face')

    parser.add_argument('-unprocessed_dir', type=str, default='./images/src', help='directory with unprocessed images')
    parser.add_argument('-output_dir', type=str, default='./images/head', help='output directory')

    parser.add_argument('-output_size', type=int, default=1024, help='size to downscale the input images to, must be power of 2')
    parser.add_argument('-seed', type=int, help='manual seed to use')
    parser.add_argument('-cache_dir', type=str, default='./checkpoint/face_detect', help='cache directory for model weights')
    parser.add_argument('-inter_method', type=str, default='bicubic')

    args = parser.parse_args()

    img = cv2.imread("./images/src/makeup/vFG112.png")
    extract_head(args, img, flag="makeup")
    img = cv2.imread("./images/src/non_makeup/target_image.jpg")
    extract_head(args, img, flag="non_makeup")

    extract_head_from_direct(args)
