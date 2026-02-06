import torch

def _mask_rfft_1d(N: int, device, dtype,
                 cutoff: float = 0.25,
                 kind: str = "gaussian",
                 order: int = 4):
    f = torch.fft.rfftfreq(N, d=1.0, device=device)  # [N//2+1], [0,0.5]
    fc = max(1e-6, float(cutoff))

    if kind == "gaussian":
        M = torch.exp(-0.5 * (f / fc) ** 2)
    elif kind == "butterworth":
        M = 1.0 / torch.sqrt(1.0 + (f / fc) ** (2 * order))
    else:
        raise ValueError(f"kind must be gaussian/butterworth, got {kind}")

    return M.to(dtype=dtype)  # [N//2+1]


def lf_tied_hf_indep_fft(x_like: torch.Tensor,
                        fft_dim: int,
                        share_dim: int,
                        cutoff: float = 0.25,
                        kind: str = "gaussian",
                        order: int = 4):
    """
    通用版：在 fft_dim 上做 rFFT/irFFT,实现
    - share_dim 维共享低频
    - share_dim 维独立高频
    输出 shape 与 x_like 一样。

    关键混合:Z = M*U + sqrt(1-M^2)*V
    """
    device, dtype = x_like.device, x_like.dtype
    N = x_like.shape[fft_dim]

    # 共享噪声 u：share_dim 上置为 1，再 expand 回去
    shape_shared = list(x_like.shape)
    shape_shared[share_dim] = 1
    u = torch.randn(shape_shared, device=device, dtype=dtype).expand_as(x_like)

    # 独立噪声 v
    v = torch.randn_like(x_like)

    U = torch.fft.rfft(u, dim=fft_dim)
    V = torch.fft.rfft(v, dim=fft_dim)

    M = _mask_rfft_1d(N, device=device, dtype=U.real.dtype,
                     cutoff=cutoff, kind=kind, order=order)
    view = [1] * U.ndim
    view[fft_dim] = -1
    M = M.view(*view)

    mix = torch.sqrt(torch.clamp(1.0 - M * M, min=0.0))
    Z = M * U + mix * V

    z = torch.fft.irfft(Z, n=N, dim=fft_dim)
    return z
