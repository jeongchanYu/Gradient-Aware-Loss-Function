import torch
import torch.nn.functional as F


def sin_window(length, device, eps=1e-3):
    return torch.sin(torch.arange(length, dtype=torch.float32, device=device)*torch.pi/length) + eps

def stft(x, window, overlap=0.5, ignore_zero_bin=True):
    win_len = len(window)
    hop_len = win_len  - round(win_len * overlap)

    s = torch.stft(x, win_len, hop_len, win_len, window, False, return_complex=True)
    s = s[..., 1:, :] if ignore_zero_bin else s
    return s

def istft(s, window, overlap=0.5, ignore_zero_bin=True):
    win_len = len(window)
    hop_len = win_len  - round(win_len * overlap)
    x_length = (s.shape[-1] - 1) * hop_len + win_len

    s = F.pad(s, (0, 0, 1, 0)) if ignore_zero_bin else s
    x = torch.istft(s, win_len, hop_len, win_len, window,False, length=x_length)
    return x

def cpx_to_mp(cpx):
    mag = torch.abs(cpx)
    pha = torch.angle(cpx)
    return mag, pha

def mp_to_cpx(mag, pha):
    cpx = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    return cpx


def mae_loss(x_true, x_pred, batch_size):
    x_true = x_true.to(dtype=torch.double)
    x_pred = x_pred.to(dtype=torch.double)
    return torch.sum(torch.mean(torch.abs(x_true - x_pred), tuple(range(1, x_true.ndim)))) / batch_size

def mse_loss(x_true, x_pred, batch_size):
    x_true = x_true.to(dtype=torch.double)
    x_pred = x_pred.to(dtype=torch.double)
    return torch.sum(torch.mean(torch.square(x_true - x_pred), tuple(range(1, x_true.ndim)))) / batch_size

def log_mae_loss(x_true, x_pred, eps, batch_size):
    x_true = x_true.to(dtype=torch.double) + eps
    x_pred = x_pred.to(dtype=torch.double) + eps
    return torch.sum(torch.mean(torch.abs(torch.log(x_true) - torch.log(x_pred)), tuple(range(1, x_true.ndim)))) / batch_size

def log_mse_loss(x_true, x_pred, eps, batch_size):
    x_true = x_true.to(dtype=torch.double) + eps
    x_pred = x_pred.to(dtype=torch.double) + eps
    return torch.sum(torch.mean(torch.square(torch.log(x_true) - torch.log(x_pred)), tuple(range(1, x_true.ndim)))) / batch_size

def ga_lsd1_loss(x_true, x_pred, eps, batch_size):
    x_true = x_true.to(dtype=torch.double) + eps
    x_pred = x_pred.to(dtype=torch.double) + eps
    return torch.sum(torch.mean(- x_pred * (torch.log(x_true) - torch.log(x_pred) + 1) + x_true, tuple(range(1, x_true.ndim)))) / batch_size

def ga_lsd2_loss(x_true, x_pred, eps, batch_size):
    x_true = x_true.to(dtype=torch.double) + eps
    x_pred = x_pred.to(dtype=torch.double) + eps
    loss = - x_pred * (torch.square(torch.log(x_true) - torch.log(x_pred) + 1) + 1) + 2 * x_true
    return torch.sum(torch.mean(torch.where(x_true >= x_pred, loss, -loss), tuple(range(1, x_true.ndim)))) / batch_size

class Log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.log(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def log_dagger(x):
    return Log.apply(x)

def ga_lsdp_loss(x_true, x_pred, p, eps, batch_size):
    x_true = x_true.to(dtype=torch.double) + eps
    x_pred = x_pred.to(dtype=torch.double) + eps
    lsd = torch.abs(log_dagger(x_true) - log_dagger(x_pred))
    return torch.sum(torch.mean(torch.pow(lsd, p + 1) / (p + 1), tuple(range(1, x_true.ndim)))) / batch_size

def lsd(references, estimates, p, window, overlap, batch_size):
    references = torch.abs(stft(references, window, overlap, False))
    estimates = torch.abs(stft(estimates, window, overlap, False))
    scores = torch.sum(torch.pow(torch.mean(torch.pow(torch.abs(torch.log(references + 1e-12) - torch.log(estimates + 1e-12)), p), (-2, -1)), 1/p)) / batch_size
    return scores

def sdr(references, estimates, batch_size):
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), axis=-1)
    den = torch.sum(torch.square(references - estimates), axis=-1)
    num += delta
    den += delta
    scores = torch.sum(10 * torch.log10(num / den)) / batch_size
    return scores