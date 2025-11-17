import os, sys, time, datetime, math, random

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torchmetrics.audio.pesq import perceptual_evaluation_speech_quality as pesq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import shared.dataset as dataset
import shared.util_func as util
import shared.sig_proc_func as sp
from config import *
import model


def train(rank, num_gpus):
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
    batch_per_gpu = batch_size // num_gpus

    # init model
    se_model = model.SE(kernel_size, channel_size, hidden_size).to(device)
    optimizer = torch.optim.AdamW(se_model.parameters(), learning_rate)
    sin_window = sp.sin_window(fft_size, device=device)

    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        load_checkpoint_path = os.path.join('./checkpoint', f"{load_checkpoint_name}.pth")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        se_model.load_state_dict(checkpoint['se_model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        saved_epoch = 0

    # multi gpu model upload
    if num_gpus > 1:
        se_model = DistributedDataParallel(se_model, device_ids=[rank])

    # dataloader
    train_set = dataset.train_dataset(dataset_path, frame_size, sample_rate)
    train_sampler = DistributedSampler(train_set) if num_gpus > 1 else None
    train_loader = DataLoader(train_set, shuffle=False if num_gpus > 1 else True, sampler=train_sampler, batch_size=batch_per_gpu, pin_memory=True, drop_last=True)

    test_set = dataset.test_dataset(dataset_path, sample_rate)
    test_sampler = DistributedSampler(test_set, shuffle=False) if num_gpus > 1 else None
    test_loader = DataLoader(test_set, shuffle=False, sampler=test_sampler, batch_size=1, pin_memory=True, drop_last=False)

    # learning curve
    if rank == 0:
        os.makedirs(os.path.join('./learning_curve', save_checkpoint_name), exist_ok=True)
        train_curve_path = os.path.join('./learning_curve', save_checkpoint_name, 'train.csv')
        test_curve_path = os.path.join('./learning_curve', save_checkpoint_name, 'test.csv')
        if load_checkpoint_name != "":
            with open(train_curve_path, 'r') as f_train:
                train_curve = f_train.readlines()
            with open(train_curve_path, 'w') as f_train:
                for i in range(saved_epoch + 1):
                    f_train.write(train_curve[i])
            with open(test_curve_path, 'r') as f_test:
                test_curve = f_test.readlines()
            with open(test_curve_path, 'w') as f_test:
                for i in range(saved_epoch + 1):
                    f_test.write(test_curve[i])
        else:
            with open(train_curve_path, 'w') as f_train:
                f_train.write('epoch,loss,\n')
            with open(test_curve_path, 'w') as f_test:
                f_test.write('epoch,pesq,lsd1,lsd2,lsd4,\n')

    # run
    for epoch in range(saved_epoch + 1, epochs + 1):
        # train
        se_model.train()
        random.seed(seed+epoch)
        np.random.seed(seed+epoch)
        torch.manual_seed(seed+epoch)
        if rank == 0:
            train_start_time = time.time()
            current_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            train_loss = 0
        for i, batch in enumerate(train_loader):
            if rank == 0:
                print(f"\r({current_time})", end=" ")
                print(util.change_font_color('bright cyan', 'Train:'), end=" ")
                print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{epochs},"), end=" ")
                print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{i + 1}/{len(train_loader)}"), end=" ")

            file_name, clean, noisy = batch
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)

            clean_pred = se_model(noisy.unsqueeze(1)).squeeze(1)

            clean_mag = torch.abs(sp.stft(clean, sin_window, 0.5, False))
            clean_pred_mag = torch.abs(sp.stft(clean_pred, sin_window, 0.5, False))
            loss = sp.ga_lsdp_loss(clean_mag, clean_pred_mag, 4, eps, batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_gpus > 1:
                dist.all_reduce(loss)
            if rank == 0:
                train_loss += np.array([loss.item(), ])

        if rank == 0:
            train_loss /= len(train_loader)
            with open(train_curve_path, 'a') as f_train:
                f_train.write(f'{epoch},{train_loss[0]},\n')

            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{train_loss[0]:.4E}"), end=" ")
            print(f"({util.second_to_dhms_string(time.time() - train_start_time)})")

        # test
        se_model.eval()
        with torch.no_grad():
            if rank == 0:
                test_start_time = time.time()
                now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
                test_loss = 0
            for i, batch in enumerate(test_loader):
                if rank == 0:
                    print(f"\r({now_time})", end=" ")
                    print(util.change_font_color('bright cyan', 'Test:'), end=" ")
                    print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{epochs},"), end=" ")
                    print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{i + 1}/{len(test_loader)}"), end=" ")

                file_name, clean, noisy = batch
                if clean.shape[0] != 0:
                    clean = clean.to(device, non_blocking=True)
                    noisy = noisy.to(device, non_blocking=True)

                    pad_len = (fft_size - (noisy.shape[1] % fft_size)) % fft_size
                    noisy_pad = F.pad(noisy, (fft_size, pad_len+fft_size))

                    clean_pred = se_model(noisy_pad.unsqueeze(0)).squeeze(0)[:, fft_size:fft_size + noisy.shape[1]]

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
                with open(test_curve_path, 'a') as f_test:
                    f_test.write(f'{epoch},{test_loss[0]},{test_loss[1]},{test_loss[2]},{test_loss[3]},\n')

                print(util.change_font_color('bright black', '|'), end=" ")
                print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{test_loss[0]:.4E}"), end=" ")
                print(f"({util.second_to_dhms_string(time.time() - test_start_time)})")

        # checkpoint save
        if rank == 0:
            if epoch % save_checkpoint_period == 0 or epoch % save_optimizer_period == 0:
                checkpoint = {}
                checkpoint['se_model'] = (se_model.module if num_gpus > 1 else se_model).state_dict()
                if epoch % save_optimizer_period == 0:
                    checkpoint['optimizer'] = optimizer.state_dict()
                os.makedirs('./checkpoint', exist_ok=True)
                torch.save(checkpoint, f"./checkpoint/{save_checkpoint_name}_{epoch}.pth")

        # calc end time
        if rank == 0:
            total_time = time.time() - train_start_time
            left_time = (epochs - epoch) * total_time
            now_time = datetime.datetime.now()
            end_time = now_time+datetime.timedelta(seconds=left_time)
            now_time = now_time.strftime("%Y-%m-%d/%H:%M:%S")
            end_time = end_time.strftime("%Y-%m-%d/%H:%M:%S")

            print(f"\r({now_time})", end=" ")
            print(util.change_font_color('bright cyan', 'Left:'), f"{util.second_to_dhms_string(left_time)}", end=" ")
            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'End:'), f"{end_time}")


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
        mp.spawn(train, nprocs=num_gpus, args=(num_gpus,))
    else:
        train(0, num_gpus)
