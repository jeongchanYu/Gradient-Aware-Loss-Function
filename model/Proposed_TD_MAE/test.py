import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import time, datetime, math, random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torchmetrics.audio.pesq import perceptual_evaluation_speech_quality as pesq

import shared.dataset as dataset
import shared.util_func as util
import shared.sig_proc_func as sp
from config import *
import model


def test(rank, num_gpus):
    util.raise_issue(f"Process initiated.(Rank{rank})")
    if num_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:54321', world_size=num_gpus, rank=rank)

    # seed and device setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # calc batch size
    if batch_size % num_gpus != 0:
        util.raise_error("batch_size must be multiple of num_gpus.")

    # init model
    se_model = model.SE(kernel_size, channel_size, hidden_size).to(device)
    sin_window = sp.sin_window(fft_size, device=device)

    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        load_checkpoint_path = os.path.join('./checkpoint', f"{load_checkpoint_name}.pth")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        se_model.load_state_dict(checkpoint['se_model'], strict=False)
    else:
        util.raise_error("Checkpoint did not loaded.")

    # multi gpu model upload
    if num_gpus > 1:
        se_model = DistributedDataParallel(se_model, device_ids=[rank])

    # dataloader
    test_set = dataset.test_dataset(dataset_path, sample_rate)
    test_sampler = DistributedSampler(test_set, shuffle=False) if num_gpus > 1 else None
    test_loader = DataLoader(test_set, shuffle=False, sampler=test_sampler, batch_size=1, pin_memory=True, drop_last=False)

    # run
    se_model.eval()
    with torch.no_grad():
        if rank == 0:
            start_time = time.time()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            test_loss = 0

            # pesq lsd metric
            os.makedirs(os.path.join('./test_result', load_checkpoint_name), exist_ok=True)
            f_test = open(os.path.join('./test_result', load_checkpoint_name, 'obj_pesq_lsd.csv'), 'w')
            f_test.write('pesq,lsd1,lsd2,lsd4\n')

        for i, batch in enumerate(test_loader):
            if rank == 0:
                print(f"\r({now_time})", end=" ")
                print(util.change_font_color('bright cyan', 'Test:'), end=" ")
                print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{saved_epoch},"), end=" ")
                print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{i + 1}/{len(test_loader)}"), end=" ")

            file_name, clean, noisy = batch
            if clean.shape[0] != 0:
                clean = clean.to(device, non_blocking=True)
                noisy = noisy.to(device, non_blocking=True)

                pad_len = (fft_size - (noisy.shape[1] % fft_size)) % fft_size
                noisy_pad = F.pad(noisy, (fft_size, pad_len + fft_size))

                clean_pred = se_model(noisy_pad.unsqueeze(0)).squeeze(0)[:, fft_size:fft_size + noisy.shape[1]]\

                save_path = os.path.join('./test_result', load_checkpoint_name, file_name[0])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                sf.write(save_path, clean_pred[0].detach().cpu(), sample_rate)

                pesq_ = torch.sum(pesq(clean_pred, clean, 16000, 'wb', keep_same_device=True))
                lsd1_ = sp.lsd(clean, clean_pred, 1, sin_window, 0.5, 1)
                lsd2_ = sp.lsd(clean, clean_pred, 2, sin_window, 0.5, 1)
                lsd4_ = sp.lsd(clean, clean_pred, 4, sin_window, 0.5, 1)
            else:
                pesq_ = torch.zeros(1, device=device)
                lsd1_ = torch.zeros(1, device=device)
                lsd2_ = torch.zeros(1, device=device)
                lsd4_ = torch.zeros(1, device=device)

            if num_gpus > 1:
                dist.all_reduce(pesq_)
                dist.all_reduce(lsd1_)
                dist.all_reduce(lsd2_)
                dist.all_reduce(lsd4_)
            if rank == 0:
                test_loss += np.array([pesq_.item(), lsd1_.item(), lsd2_.item(), lsd4_.item()])

        if rank == 0:
            test_loss /= len(test_set)
            f_test.write(f'{test_loss[0]},{test_loss[1]},{test_loss[2]},{test_loss[3]},\n')
            f_test.close()

            end_time = util.second_to_dhms_string(time.time() - start_time)
            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{test_loss[0]:.4E}"), end=" ")
            print(f"({end_time})")


if __name__ == '__main__':
    # train var preset
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0

    # run
    if num_gpus > 1:
        mp.spawn(test, nprocs=num_gpus, args=(num_gpus,))
    else:
        test(0, num_gpus)
