# Credits: This code is adapted from the DDIP3D here https://github.com/hyungjin-chung/DDIP3D/blob/main/physics/mri.py
import torch
from physics.fastmri_utils import fft2_m, ifft2_m


class SinglecoilMRI_real:
    def __init__(self, mask):
        self.mask = mask

    def _A(self, x):
        return fft2_m(x) * self.mask

    def _Adagger(self, x):
        return torch.real(ifft2_m(x))

    def _AT(self, x):
        return self._Adagger(x)


class SinglecoilMRI_comp:
    def __init__(self, mask):
        self.mask = mask

    def _A(self, x):
        return fft2_m(x) * self.mask

    def _Adagger(self, x):
        return ifft2_m(x)

    def _AT(self, x):
        return self._Adagger(x)


class MulticoilMRI():
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def _A(self, x, mps, use_mask=True):
        if use_mask:
            return fft2_m(mps * x) * self.mask
        else:
            return fft2_m(mps * x)
    
    def _Adagger(self, x, mps, use_mask=True):
        if use_mask:
            return torch.sum(torch.conj(mps) * ifft2_m(x * self.mask), dim=1).unsqueeze(dim=1)
        else:
            return torch.sum(torch.conj(mps) * ifft2_m(x), dim=1).unsqueeze(dim=1)

    def _AT(self, x, mps, use_mask=True):
        return self._Adagger(x, mps, use_mask=use_mask)

    