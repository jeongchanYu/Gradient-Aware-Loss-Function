import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import librosa
import torch
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import shared.sig_proc_func as sp

fft_size= 512
overlap = 0.75
eps = 1e-12

clean = librosa.load("./clean.wav", sr=16000)[0]
noisy = librosa.load("./noisy.wav", sr=16000)[0]
noise = (noisy - clean) * 5
noisy = clean + noise

clean = torch.from_numpy(clean).unsqueeze(0).requires_grad_(True)
noise = torch.from_numpy(noise).unsqueeze(0).requires_grad_(True)
noisy = torch.from_numpy(noisy).unsqueeze(0).requires_grad_(True)

sin_window = sp.sin_window(fft_size, device=None)
clean_mag, clean_pha = sp.cpx_to_mp(sp.stft(clean, sin_window, overlap, False))
noise_mag, noise_pha = sp.cpx_to_mp(sp.stft(noise, sin_window, overlap, False))
noisy_mag, noisy_pha = sp.cpx_to_mp(sp.stft(noisy, sin_window, overlap, False))
clean_log = torch.log(clean_mag + eps)
noise_log = torch.log(noise_mag + eps)
noisy_log = torch.log(noisy_mag + eps)
lsd = torch.abs(clean_log - noisy_log)
clean_pred = sp.istft(sp.mp_to_cpx(noisy_mag, noisy_pha), sin_window, overlap, False)

scale = clean_mag.shape[1]*clean_mag.shape[2] # for ignore mean operation
time_l1 = sp.mae_loss(clean[:, :clean_pred.shape[1]], clean_pred, 1) * scale
time_l2 = sp.mse_loss(clean[:, :clean_pred.shape[1]], clean_pred, 1) * scale
freq_l1 = sp.mae_loss(clean_mag, noisy_mag, 1) * scale
freq_l2 = sp.mse_loss(clean_mag, noisy_mag, 1) * scale
log_l1 = sp.log_mae_loss(clean_mag, noisy_mag, eps, 1) * scale
log_l2 = sp.log_mse_loss(clean_mag, noisy_mag, eps, 1) * scale
glsd1 = sp.ga_lsd1_loss(clean_mag, noisy_mag, eps, 1) * scale
glsd2 = sp.ga_lsd2_loss(clean_mag, noisy_mag, eps, 1) * scale
glsd4 = sp.ga_lsdp_loss(clean_mag, noisy_mag, 4, eps, 1) * scale

grad_time_l1 = torch.abs(torch.autograd.grad(time_l1, noisy_mag, retain_graph=True)[0])
grad_time_l2 = torch.abs(torch.autograd.grad(time_l2, noisy_mag, retain_graph=True)[0])
grad_freq_l1 = torch.abs(torch.autograd.grad(freq_l1, noisy_mag, retain_graph=True)[0])
grad_freq_l2 = torch.abs(torch.autograd.grad(freq_l2, noisy_mag, retain_graph=True)[0])
grad_log_l1 = torch.abs(torch.autograd.grad(log_l1, noisy_mag, retain_graph=True)[0])
grad_log_l2 = torch.abs(torch.autograd.grad(log_l2, noisy_mag, retain_graph=True)[0])
grad_glsd1 = torch.abs(torch.autograd.grad(glsd1, noisy_mag, retain_graph=True)[0])
grad_glsd2 = torch.abs(torch.autograd.grad(glsd2, noisy_mag, retain_graph=True)[0])
grad_glsd4 = torch.abs(torch.autograd.grad(glsd4, noisy_mag, retain_graph=True)[0])


def imshow(x, title, vmin, vmax, cmap=None):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.title(title)
    plt.imshow(x.detach().squeeze(0), origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Time (s)")
    xtick_positions = [0, 125, 250, 375, 500, 625, 750]  # 0, 2, 4, 8 kHz에 해당하는 픽셀 위치
    xtick_labels = ["0", "1", "2", "3", "4", "5", "6"]  # 원하는 라벨
    plt.xticks(xtick_positions, xtick_labels)

    plt.ylabel("Frequency (kHz)")
    ytick_positions = [0, 64, 128, 192, 255]  # 0, 2, 4, 8 kHz에 해당하는 픽셀 위치
    ytick_labels = ["0", "2", "4", "6", "8"]  # 원하는 라벨
    plt.yticks(ytick_positions, ytick_labels)

    cbar = plt.colorbar()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # 모든 값에 대해 지수 표기법 강제 적용
    cbar.ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
    plt.close()

imshow(clean_log, r'Spectrogram ($x_pred$)', -6, 3, 'inferno')
imshow(noisy_log, r'Spectrogram ($\hat{x_pred}$)', -6, 3, 'inferno')
imshow(lsd, 'LSD', 0, 10)
imshow(grad_time_l1, 'Gradient (time MAE)', 0, .1)
imshow(grad_time_l2, 'Gradient (time MSE)', 0, .1)
imshow(grad_freq_l1, 'Gradient (frequency MAE)', 0, 10)
imshow(grad_freq_l2, 'Gradient (frequency MSE)', 0, 100)
imshow(grad_log_l1, 'Gradient (log MAE)', 0, 100)
imshow(grad_log_l2, 'Gradient (log MSE)', 0, 1000)
imshow(grad_glsd1, 'Gradient (GA-LSD1)', 0, 10)
imshow(grad_glsd2, 'Gradient (GA-LSD2)', 0, 100)
imshow(grad_glsd4, 'Gradient (GA-LSD4)', 0, 10000)