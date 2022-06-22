import shutil
import sys
import torch
import os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, "./clothes_transfer")

import cv2
import numpy as np
import matplotlib.pyplot as plt


def dress_in_orders(pid, gids, ogids, name='test', mode=2):
    dataroot = './clothes_transfer/fashion_data'
    exp_name = 'DIOR_64'  # DIORv1_64
    epoch = 'latest'
    netG = 'dior'  # diorv1
    ngf = 64

    ## this is a dummy "argparse"
    class Opt:
        def __init__(self):
            pass

    if True:
        opt = Opt()
        opt.dataroot = dataroot
        opt.isTrain = False
        opt.phase = 'test'
        opt.n_human_parts = 8;
        opt.n_kpts = 18;
        opt.style_nc = 64
        opt.n_style_blocks = 4;
        opt.netG = netG;
        opt.netE = 'adgan'
        opt.ngf = ngf
        opt.norm_type = 'instance';
        opt.relu_type = 'leakyrelu'
        opt.init_type = 'orthogonal';
        opt.init_gain = 0.02;
        opt.gpu_ids = [0]
        opt.frozen_flownet = True;
        opt.random_rate = 1;
        opt.perturb = False;
        opt.warmup = False
        opt.name = exp_name
        opt.vgg_path = '';
        opt.flownet_path = './checkpoint/DIOR_64/latest_net_Flow.pth'
        opt.checkpoints_dir = './checkpoint'
        opt.frozen_enc = True
        opt.load_iter = 0
        opt.epoch = epoch
        opt.verbose = False

    # PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
    # GID = [2,5,1,3] # hair头发, top上衣, bottom下衣, jacket外套
    # create model
    from clothes_transfer.models.dior_model import DIORModel
    model = DIORModel(opt)
    model.setup(opt)

    # # --------------------------Set up------------------------------------------
    # load data
    from clothes_transfer.datasets.deepfashion_datasets import DFVisualDataset

    Dataset = DFVisualDataset
    ds = Dataset(dataroot=dataroot, dim=(256, 176), n_human_part=8)

    # preload a set of pre-selected models defined in "standard_test_anns.txt" for quick visualizations
    inputs = dict()
    for attr in ds.attr_keys:
        inputs[attr] = ds.get_attr_visual_input(attr)

    # define some tool functions for I/O
    def load_img(pid, ds):
        if isinstance(pid, str):  # load pose from scratch
            return None, None, load_pose_from_json(pid)
        if len(pid[0]) < 10 and pid[0] != 'user.jpg':  # load pre-selected models
            person = inputs[pid[0]]
            person = (i.cuda() for i in person)
            # 原图 mask 姿势
            pimg, parse, to_pose = person
            pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
        else:  # load model from scratch
            person = ds.get_inputs_by_key(pid[0])
            person = (i.cuda() for i in person)
            pimg, parse, to_pose = person
        return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

    def load_pose_from_json(ani_pose_dir):
        with open(ani_pose_dir, 'r') as f:
            anno = json.load(f)
        len(anno['people'][0]['pose_keypoints_2d'])
        anno = list(anno['people'][0]['pose_keypoints_2d'])
        x = np.array(anno[1::3])
        y = np.array(anno[::3])

        coord = np.concatenate([x[:, None], y[:, None]], -1)
        # import pdb; pdb.set_trace()
        # coord = (coord * 1.1) - np.array([10,30])[None, :]
        pose = pose_utils.cords_to_map(coord, (256, 176), (256, 256))
        pose = np.transpose(pose, (2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    # pimg 原始图像 gimg 衣服照片 oimg 外面衣服 genimg 生成图像 pose 骨架
    def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None, name=None):
        if pose is not None:
            import clothes_transfer.utils.pose_utils as pose_utils
            print(pose.size())
            kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1, 2, 0), radius=6)
            kpt = kpt[0]
        if not isinstance(pimg, list):
            pimg = [pimg]
        if not isinstance(gen_img, list):
            gen_img = [gen_img]
        out = pimg + gimgs + oimgs + gen_img
        if out:
            out = torch.cat(out, 2).float().cpu().detach().numpy()
            out = (out + 1) / 2  # denormalize
            out = np.transpose(out, [1, 2, 0])

            pimg1 = pimg[0].float().cpu().detach().numpy()
            pimg1 = 255 * (pimg1 + 1) / 2  # denormalize
            pimg1 = np.transpose(pimg1, [1, 2, 0])
            pimg1 = cv2.cvtColor(pimg1, cv2.COLOR_RGB2BGR)

            gen_img1 = gen_img[0].float().cpu().detach().numpy()
            gen_img1 = 255 * (gen_img1 + 1) / 2  # denormalize
            gen_img1 = np.transpose(gen_img1, [1, 2, 0])
            gen_img1 = cv2.cvtColor(gen_img1, cv2.COLOR_RGB2BGR)

            gimgs_img = torch.cat(gimgs, 2).float().cpu().detach().numpy()
            gimgs_img = 255 * (gimgs_img + 1) / 2  # denormalize
            gimgs_img = np.transpose(gimgs_img, [1, 2, 0])
            gimgs_img = cv2.cvtColor(gimgs_img, cv2.COLOR_RGB2BGR)

            if pose is not None:
                out = np.concatenate((kpt, out), 1)
        else:
            out = kpt
        fig = plt.figure(figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
        plt.axis('off')

        plt.imshow(out)
        plt.savefig('{}.jpg'.format(name))

        # fig = plt.figure(figsize=(2, 1), dpi=128, facecolor='w', edgecolor='k')
        # plt.axis('off')
        cv2.imwrite('{}-pimg.jpg'.format(name), pimg1)
        cv2.imwrite('{}-gen.jpg'.format(name), gen_img1)
        cv2.imwrite('{}-cloth.jpg'.format(name), gimgs_img)
        gen_img2 = cv2.imread('{}-gen.jpg'.format(name))
        gimg_img1 = cv2.imread('{}-cloth.jpg'.format(name))
        # cv2.imshow("img",gen_img2)
        # cv2.waitKey(0)
        # plt.imshow(pimg1)
        # plt.savefig('{}-pimg.jpg'.format(name))
        # plt.imshow(gen_img1)
        # plt.savefig('{}-gen.jpg'.format(name))
        return gen_img2, gimg_img1

    # define dressing_in_order function (the pipeline)
    # PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
    # GID = [2,5,1,3] # hair, top, bottom, jacket
    #  原图像pid, 姿势图像pose_id=None, gids=[], ogids=[]
    def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5, 1, 3, 2], perturb=False):
        PID = [0, 4, 6, 7]
        GID = [2, 5, 1, 3]
        # encode person
        pimg, parse, from_pose = load_img(pid, ds)
        if perturb:
            pimg = perturb_images(pimg[None])[0]
        if not pose_id:
            to_pose = from_pose
        else:
            to_img, _, to_pose = load_img(pose_id, ds)
        psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

        # encode base garments
        gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])

        # swap base garment if any
        gimgs = []
        gimgs_list = []
        for gid in gids:
            _, _, k = gid
            gimg, gparse, pose = load_img(gid, ds)
            seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
            gsegs[gid[2]] = seg
            # gimgs += [gimg * (gparse == gid[2])]
            result = [gimg * (gparse == gid[2]).float()]
            if result[0].max() != 0:
                gimgs += result
                gimgs_list.append(gid[2])
        # encode garment (overlay)
        garments = []
        over_gsegs = []
        oimgs = []
        for gid in ogids:
            oimg, oparse, pose = load_img(gid, ds)
            oimgs += [oimg * (oparse == gid[2]).float()]
            seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
            over_gsegs += [seg]

        gsegs = [gsegs[i] for i in order] + over_gsegs
        gen_img = model.netG(to_pose[None], psegs, gsegs)

        return pimg, gimgs, oimgs, gen_img[0], to_pose, gimgs_list

    # Layering - Multiple (dress in order)
    # person id
    # pid = ("fashionWOMENBlouses_Shirtsid0000637003_1front.jpg", None, None)  # load person from the file
    #
    # # garments to try on (ordered)
    # gids = [
    #     ("gfla", 2, 5),
    #     ("strip", 3, 1),
    # ]
    #
    # # garments to lay over (ordered)
    # ogids = [
    #     ("fashionWOMENTees_Tanksid0000159006_1front.jpg", None, 5),
    #     ('fashionWOMENJackets_Coatsid0000645302_1front.jpg', None, 3),
    # ]

    # dressing in order
    pimg, gimgs, oimgs, gen_img, pose, gimgs_list = dress_in_order(model, pid=pid, gids=gids, ogids=ogids)
    # trans
    if mode == 2:
        gen_img1, _ = plot_img(pimg, gimgs, oimgs, gen_img, pose, name=name)
        return gen_img1
    # cloth
    if mode == 1:
        _, cloth_img1 = plot_img(pimg, gimgs, oimgs, gen_img, pose, name=name)
        return cloth_img1, gimgs_list


def img_move(src1, src2, name):
    root = os.getcwd()
    print(root)
    shutil.copy(src1, os.path.join(src2, name))
    print(os.path.join(src2, name))
    root = os.getcwd()
    print(root)


def preprocess():
    from generate_fashion_datasets import make_dataset
    # 关键点提取采�?76*256，人体分割采�?50*1101
    make_dataset('./clothes_transfer/fashion_image_src', new_root='./clothes_transfer/fashion_data',
                 new_size=(176, 256),
                 crop_root='./clothes_transfer/fashion_image_crop', crop_size=(750, 1101))
    from compute_coordinates import get_keypoints

    get_keypoints(input_folder='./clothes_transfer/fashion_data/test',
                  output_path='./clothes_transfer/fashion_data/fasion-annotation-test.csv')
    from human_parsing import human_parsing
    human_parsing(input_dir='./clothes_transfer/fashion_image_crop/test',
                  output_dir='./clothes_transfer/fashion_data/testM_lip')

    make_dataset('./clothes_transfer/fashion_image_src', new_root='./clothes_transfer/fashion_data',
                 new_size=(176, 256),
                 crop_root='./clothes_transfer/fashion_image_crop', crop_size=(750, 1101), mode=11)


# result_image:(ndarray)  = clothes_transfer(user_image:(ndarray),clothes_class:(list))

def clothes_transfer(raw_path, clothes_path, ids=[5, 1, 3]):
    # 移动文件路径
    img_move(raw_path, './clothes_transfer/fashion_image_src', 'user.jpg')
    img_move(clothes_path, './clothes_transfer/fashion_image_src', 'model_template.jpg')

    preprocess()

    # 用户图像
    pid = ("user.jpg", None, None)  # load person from the file
    # 模特图像
    # garments to try on (ordered) 根据需要更�?
    gids = []
    for id in ids:
        gids.append(("model_template.jpg", None, id))
    # gids = [
    #     ("model_template.jpg", None, 5),  # top 上衣
    #     ("model_template.jpg", None, 1),  # bottom 下装
    #     ('model_template.jpg', None, 3),  # jacket 外套
    #     # ("model_template.jpg", None, 2),# hair 头发
    # ]
    # garments to lay over (ordered)
    ogids = [
        # ("model_template.jpg", None, 5),  # top 上衣
        # ("model_template.jpg", None, 1),  # bottom 下装

    ]
    gen_img = dress_in_orders(pid, gids, ogids, name='./images/clothes/trans_img', mode=2)

    return gen_img


# clothes_class:(list), clothes_image:(ndarray) = parse_clothes(model_image:(ndarray))
def parse_clothes(clothes_path, ids=[5, 1, 3]):
    # 移动文件路径
    img_move(clothes_path, './clothes_transfer/fashion_image_src', 'user.jpg')
    img_move(clothes_path, './clothes_transfer/fashion_image_src', 'model_template.jpg')

    preprocess()

    # 用户图像
    pid = ("user.jpg", None, None)  # load person from the file
    # 模特图像
    # garments to try on (ordered) 根据需要更�?
    gids = []
    for id in ids:
        gids.append(("model_template.jpg", None, id))
    # gids = [
    #     ("model_template.jpg", None, 5),  # top 上衣
    #     ("model_template.jpg", None, 1),  # bottom 下装
    #     ('model_template.jpg', None, 3),  # jacket 外套
    #     # ("model_template.jpg", None, 2),# hair 头发
    # ]
    # garments to lay over (ordered)
    ogids = [
        # ("model_template.jpg", None, 5),  # top 上衣
        # ("model_template.jpg", None, 1),  # bottom 下装

    ]

    cloth_img, cloth_list = dress_in_orders(pid, gids, ogids, name='./images/clothes/cloth_img', mode=1)

    return cloth_img, cloth_list


# PID = [0,4,6,7] # bg, face, arm, leg (the first has to be bg and the second has to be face.)
# GID = [2,5,1,3] # hair, top, bottom, jacket
if __name__ == "__main__":
    # preprocess()
    # # 用户图像
    # pid = ("user.jpg", None, None)  # load person from the file
    # # 模特图像
    # # garments to try on (ordered) 根据需要更�?
    # gids = [
    #     # ("model_template.jpg", None, 5),  # top 上衣
    #     # ("model_template.jpg", None, 1),  # bottom 下装
    #     # ("model_template.jpg", None, 2),# hair 头发
    # ]
    # # garments to lay over (ordered)
    # ogids = [
    #     ("model_template.jpg", None, 5),  # top 上衣
    #     ("model_template.jpg", None, 1),  # bottom 下装
    #     ('model_template.jpg', None, 3),# jacket 外套
    # ]
    # dress_in_orders(pid, gids, ogids, name='test2')
    # print(using_clothes('./test-datasets/test/fashionMENShirts_Polosid0000113801_1front.jpg',
    #                     './test-datasets/test/fashionWOMENShortsid0000628101_1front.jpg'))
    img, gimg_list = parse_clothes(clothes_path='test-datasets/test/fashionMENDenimid0000056501_7additional.jpg')
    # imgs=clothes_transfer('./test-datasets/test/fashionMENShirts_Polosid0000113801_1front.jpg',
    #                    './test-datasets/test/fashionWOMENShortsid0000628101_1front.jpg',gimg_list
    # )
