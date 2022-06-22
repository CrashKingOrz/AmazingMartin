from generate_fashion_datasets import make_dataset
from compute_coordinates import get_keypoints
from human_parsing import human_parsing


def preprocess():
    # 关键点提取采用176*256，人体分割采用750*1101
    make_dataset('./clothes_transfer/fashion_image_src', new_root='./clothes_transfer/fashion_data',
                 new_size=(176, 256),
                 crop_root='./clothes_transfer/fashion_image_crop', crop_size=(750, 1101))
    get_keypoints(input_folder='./clothes_transfer/fashion_data/test',
                  output_path='./clothes_transfer/fashion_data/fasion-annotation-test.csv')
    human_parsing(input_dir='./clothes_transfer/fashion_image_crop/test',
                  output_dir='./clothes_transfer/fashion_data/testM_lip')


if __name__ == "__main__":
    preprocess()
