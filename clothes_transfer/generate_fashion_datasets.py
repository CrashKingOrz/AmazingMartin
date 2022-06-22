import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, new_root='./clothes_transfer/fashion_data', new_size=(176, 256), crop_size=(750, 1101),
                 crop_root='./clothes_transfer/fashion_image_crop',mode=1):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    test_root = new_root + '/test'
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    test_images = []
    test_f = open(new_root + '/test.lst', 'r')
    for lines in test_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)

    crop_test_root = crop_root + '/test'
    if not os.path.exists(crop_test_root):
        os.makedirs(crop_test_root)

    print(test_images)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                print("Load Image:", path)
                path_names = path.split('/')
                path_names = path_names[len(dir.split("/")):]
                path_names = "".join(path_names)

                # resize and crop
                img = Image.open(path)

                w, h = img.width, img.height

                h1 = crop_size[1]
                w1 = int(h1 / h * w)
                if w1 > crop_size[0]:
                    w1 = crop_size[0]
                    h1 = int(w1 / w * h)

                img = img.resize((w1, h1), Image.ANTIALIAS)

                w, h = w1, h1

                lefttop_x = int((w - crop_size[0]) / 2)
                lefttop_y = int((h - crop_size[1]) / 2)

                rightbottom_x = lefttop_x + crop_size[0]
                rightbottom_y = lefttop_y + crop_size[1]

                imgcrop = img.crop((lefttop_x, lefttop_y, rightbottom_x, rightbottom_y))  # 750*1101
                imgcrop.save(os.path.join(crop_test_root, path_names))
                if mode==1:
                    imgcrop = imgcrop.resize(new_size, Image.ANTIALIAS)  # 176*256

                if path_names in test_images:
                    imgcrop.save(os.path.join(test_root, path_names))


if __name__ == "__main__":
    make_dataset('fashion_image_src')
