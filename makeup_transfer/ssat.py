import torch
import os
from SSAT.dataset_makeup import MakeupDataset
from SSAT.model import MakeupGAN
from SSAT.options import MakeupOptions
from SSAT.saver import Saver


def ssat(opts):
    # parse options
    # parser = MakeupOptions()
    # opts = parser.parse()
    opts.phase = 'test_pair'
    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(train_loader))

    # model
    print('\n--- load model ---')
    model = MakeupGAN(opts)
    #ep0, total_it = model.resume(opts.resume)

    ep0, total_it = model.resume(os.path.join(opts.checkpoint_dir, 'SSAT.pth'), False)
    model.eval()
    print('start pair test')
    # saver for display and output
    saver = Saver(opts)
    for iter, data in enumerate(train_loader):
        with torch.no_grad():
            img = saver.write_test_pair_img(iter, model, data)

    return img


def ssat_from_directs():
    # parse options
    parser = MakeupOptions()
    opts = parser.parse()
    opts.phase = 'test_pair'
    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(train_loader))

    # model
    print('\n--- load model ---')
    model = MakeupGAN(opts)
    #ep0, total_it = model.resume(opts.resume)

    ep0, total_it = model.resume(os.path.join(opts.checkpoint_dir, 'SSAT.pth'), False)
    model.eval()
    print('start pair test')
    # saver for display and output
    saver = Saver(opts)
    for iter, data in enumerate(train_loader):
        with torch.no_grad():
            saver.write_test_pair_img(iter, model, data)


if __name__ == '__main__':
    ssat_from_directs()

