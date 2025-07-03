from pprint import pprint
import os

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

import pandas as pd
from xml.etree import ElementTree as ET
import sigpy as sp
import torchvision
import torch.nn as nn

# Set the default device if cuda is enabled
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device: ", DEVICE)

# Run this setup phase only once.
# Otherwise, you may get multiple print statements
#setup_logger()
#logger = setup_logger("skm_tea")
#path_mgr = env.get_path_manager()

# Some general utilities

from typing import Union, Sequence


def get_scaled_image(
        x: Union[torch.Tensor, np.ndarray], percentile=0.99, clip=False
):
    """Scales image by intensity percentile (and optionally clips to [0, 1]).

    Args:
      x (torch.Tensor | np.ndarray): The image to process.
      percentile (float): The percentile of magnitude to scale by.
      clip (bool): If True, clip values between [0, 1]

    Returns:
      torch.Tensor | np.ndarray: The scaled image.
    """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.as_tensor(x)

    scale_factor = torch.quantile(x, percentile)
    x = x / scale_factor
    if clip:
        x = torch.clip(x, 0, 1)

    if is_numpy:
        x = x.numpy()

    return x


class SenseModelFastMRI(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation converts a complex image -> multi-coil kspace.
    The adjoint operation converts multi-coil kspace -> a complex image.

    This module also supports multiple sensitivity maps. This is useful if
    you would like to generate images from multiple estimated sensitivity maps.
    This module also works with single coil inputs as long as the #coils dimension
    is set to 1.

    Attributes:
        maps (torch.Tensor): Sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.
        weights (torch.Tensor, optional): Undersampling masks (if applicable).
            Shape ``(B, H, W)`` or ``(B, H, W, #coils, #coils)``.
    """

    def __init__(self, maps: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            maps (torch.Tensor): Sensitivity maps.
            mask (torch.Tensor): Undersampling masks.
                If ``None``, it is assumed that inputs are fully-sampled.
        """
        super().__init__()

        self.maps = maps  # [B, H, W, #coils]
        if mask is None:
            self.weights = 1.0
        else:
            self.weights = mask

    def AT(self, kspace, use_mask=True):
        """
        Args:
            kspace: Shape (B, C, H, W)
        Returns:
            image: Shape (B, C, H, W)
        """
        if use_mask:
            kspace = kspace * self.weights
        image = torch.fft.ifftshift(kspace, dim=(-2, -1))
        image = torch.fft.ifft2(image, dim=(-2, -1), norm="ortho")
        image = torch.fft.fftshift(image, dim=(-2, -1))
        image = image * torch.conj(self.maps)
        return image.sum(dim=1)


    def A(self, image, use_mask=True):
        """
        Args:
            image: Shape (B,H,W,#maps,[2])
        Returns:
            kspace: Shape (B,H,W,#coils,[2])
        """
        #if len(image.shape) == 3:
        #    image = image.unsqueeze(1)

        kspace = image * self.maps
        kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
        kspace = torch.fft.fft2(kspace, dim=(-2, -1), norm="ortho")
        kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
        if use_mask:
            kspace = self.weights * kspace
        return kspace

    def CG(self, kspace_us, x, mask, max_iter=10):
        """
        b is the undersampled image(!) and x is the initial guess
        """
        is_complex=True

        b_kspace = mask * kspace_us #+ ~mask * self.A(x, use_mask=False)
        b = self.AT(b_kspace, use_mask=False)
        r = b - self.forward_backward(x, use_mask=True)
        p = r.clone()
        rsold = torch.sum(r * torch.conj(r))
        for i in range(max_iter):
            Ap = self.forward_backward(p)
            alpha = rsold / torch.sum(p * torch.conj(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * torch.conj(r))
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if is_complex is False:
            x = torch.stack([x.real, x.imag], dim=1)
        return x


    def forward_backward(self, image, use_mask=False):
        """
        equivalent to A* A
        """
        kspace = self.A(image, use_mask=use_mask)
        image = self.AT(kspace, use_mask=use_mask)
        return image

    def A_dagger(self, x):
        return self.AT(x)


def plot_images(
        images, processor=None, disable_ticks=True, titles: Sequence[str] = None,
        ylabel: str = None, xlabels: Sequence[str] = None, cmap: str = "gray",
        show_cbar: bool = False, overlay=None, opacity: float = 0.3,
        hsize=5, wsize=5, axs=None
):
    """Plot multiple images in a single row.

    Add an overlay with the `overlay=` argument.
    Add a colorbar with `show_cbar=True`.
    """

    def get_default_values(x, default=""):
        if x is None:
            return [default] * len(images)
        return x

    titles = get_default_values(titles)
    ylabels = get_default_values(images)
    xlabels = get_default_values(xlabels)

    N = len(images)
    if axs is None:
        fig, axs = plt.subplots(1, N, figsize=(wsize * N, hsize))
    else:
        assert len(axs) >= N
        fig = axs.flatten()[0].get_figure()

    for ax, img, title, xlabel in zip(axs, images, titles, xlabels):
        if processor is not None:
            img = processor(img)
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    if overlay is not None:
        for ax in axs.flatten():
            im = ax.imshow(overlay, alpha=opacity)

    if show_cbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    if disable_ticks:
        for ax in axs.flatten():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    return axs


def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

class MVU_Estimator_Knees(torch.utils.data.Dataset):
    # from yalal et al
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320, 320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical',
                 annotation_dir=None):
        # Attributes
        self.project_dir = project_dir
        self.file_list    = file_list
        self.acs_size     = acs_size
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.R = R
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation
        self.annotation_dir = annotation_dir

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            raw_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
                value = data['ismrmrd_header'][()]
                value = ET.fromstring(value)
                self.num_slices[idx] = int(value[4][2][3][1].text) + 1

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask Pattern not implemented yet...')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    def _knees_remove_zeros(self, kimage):
        # Compute sum-energy of lines
        # !!! This is because some lines are near-empty
        line_energy = np.sum(np.square(np.abs(kimage)),
                             axis=(0, 1))
        dead_lines  = np.where(line_energy < 1e-12)[0] # Sufficient for FP32
        # Always remove an even number of lines
        dead_lines_front = np.sum(dead_lines < 160)
        dead_lines_back  = np.sum(dead_lines > 160)
        if np.mod(dead_lines_front, 2):
            dead_lines = np.delete(dead_lines, 0)
        if np.mod(dead_lines_back, 2):
            dead_lines = np.delete(dead_lines, -1)
        # Remove dead lines completely
        k_image = np.delete(kimage, dead_lines, axis=-1)
        return k_image

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        if slice_idx > 5 and slice_idx < self.num_slices[scan_idx] - 5:
            in_the_middle = 1
        else:
            in_the_middle = 0

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])
        #gt_ksp = self._knees_remove_zeros(gt_ksp)

        # Crop extra lines and reduce FoV by half in readout
        gt_ksp = sp.resize(gt_ksp, (
            gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV by half in readout
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # new: we need to normalize the maps such that sum(S*torch.conj(S)) = 1
        factor = np.sqrt(np.sum(np.abs(maps)**2, axis=(0)))
        maps = maps / factor

        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)

        # Mask k-space
        if self.orientation == 'vertical':
            gt_ksp *= mask[None, None, :]
        elif self.orientation == 'horizontal':
            gt_ksp *= mask[None, :, None]
        else:
            raise NotImplementedError

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file,
                  'in_the_middle': in_the_middle}
        return sample

    def get_idx_with_annotations(self):
        # TODO: adjust this for this loader
        idxs = []
        annotation_files = [x for x in os.listdir(self.annotation_dir) if x.endswith('.txt')]
        for annotation_file in annotation_files:
            volume_name = annotation_file.split('_')[0]
            slice_idx = int(annotation_file.split('_')[1].split('.')[0])
            if volume_name in self.volume_names:
                # to filter for train or val sets
                idx = self.get_idx_from_volume_and_slice(volume_name, slice_idx)
                idxs.append(idx)
        return idxs


class MVU_Estimator_Origsize(MVU_Estimator_Knees):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320, 320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical',
                 annotation_dir=None):
        super().__init__(file_list, maps_dir, input_dir, project_dir, R, image_size, acs_size, pattern, orientation, annotation_dir)

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        if slice_idx > 5 and slice_idx < self.num_slices[scan_idx] - 5:
            in_the_middle = 1
        else:
            in_the_middle = 0

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])

        # create mvue file:
        A = SenseModelFastMRI(torch.from_numpy(maps).unsqueeze(0))
        image = A.AT(torch.from_numpy(gt_ksp).unsqueeze(0))

        # center crop image:
        mvue = torchvision.transforms.functional.center_crop(image, (320, 320))

        #gt_ksp = self._knees_remove_zeros(gt_ksp)

        # Crop extra lines and reduce FoV by half in readout
        #gt_ksp = sp.resize(gt_ksp, (
        #    gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        #gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        #gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
        #                            gt_ksp.shape[2]))
        #gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space


        # Crop extra lines and reduce FoV by half in readout
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # new: we need to normalize the maps such that sum(S*torch.conj(S)) = 1
        factor = np.sqrt(np.sum(np.abs(maps)**2, axis=(0)))
        maps = maps / factor

        # find mvue image
        #mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        #scale_factor = 1.

        # Scale data
        #mvue   = mvue / scale_factor
        #gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        #total_lines = gt_ksp.shape[-1]
        #if 1 < self.R <= 6:
        #    # Keep 8% of center samples
        #    acs_lines = np.floor(0.08 * total_lines).astype(int)
        #else:
        #    # Keep 4% of center samples
        #    acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        #mask = self._get_mask(acs_lines, total_lines,
        #                      self.R, self.pattern)

        # Mask k-space
        #if self.orientation == 'vertical':
        #    gt_ksp *= mask[None, None, :]
        #elif self.orientation == 'horizontal':
        #    gt_ksp *= mask[None, :, None]
        #else:
        #    raise NotImplementedError

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  #'mask': mask,
                  #'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file,
                  'in_the_middle': in_the_middle}
        return sample


class GetMRI_Fastmri(torch.utils.data.Dataset):
    def __init__(self, dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee', split='train', acc=0.0, normalize=False,
                 annotation_dir=None, only_annotations=False, percentile=0.99, image_size=[320, 320]):
        """
        normalize: divide by 99th percentile
        """
        self.image_size = image_size
        self.split = split
        self.dataset_root = dataset_root
        self.acc = acc
        self.normalize = normalize
        self.annotation_dir = annotation_dir
        self.data_dir = os.path.join(self.dataset_root, f"multicoil_{self.split}")
        self.mask_dir = os.path.join(self.dataset_root, f'multicoil_{self.split}_maps')
        # list only .h5 files in directory:
        self.volumes = [x for x in os.listdir(self.data_dir) if x.endswith('.h5')]  #file1000033
        self.volume_names = [x.split('.')[0] for x in self.volumes]
        self.masks = [x for x in os.listdir(self.mask_dir) if x.endswith('.h5')]
        self.slice_counts = pd.read_csv(os.path.join(f'./data/volume_info_{split}.csv'))
        self.cumulative_slices = []
        self.percentile = percentile
        cumulative = 0
        for vol in self.volumes:
            num_slices = self.slice_counts.loc[self.slice_counts['volume_id'] == vol.split('.')[0], 'number_of_slices'].values[0]
            cumulative += num_slices
            self.cumulative_slices.append(cumulative)
        print('Number of slices in dataset: ', self.cumulative_slices[-1])


    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, idx):
        volume_idx = next(i for i, v in enumerate(self.cumulative_slices) if idx < v)
        if volume_idx > 0:
            slice_idx = idx - self.cumulative_slices[volume_idx - 1]
            slices_in_volume = self.cumulative_slices[volume_idx] - self.cumulative_slices[volume_idx - 1]
        else:
            slice_idx = idx
            slices_in_volume = self.cumulative_slices[volume_idx]
        if slice_idx > 5 and slice_idx < slices_in_volume - 5:
            in_the_middle = 1
        else:
            in_the_middle = 0
        filename = os.path.join(self.data_dir, self.volumes[volume_idx])
        with h5py.File(filename, "r") as f:
            # scan has keys: ['ismrmrd_header', 'kspace', 'reconstruction_rss']
            kspace = f['kspace'][slice_idx, :, :, :]
            #attrs = dict(f.attrs)
            rec_rss = f['reconstruction_rss'][slice_idx, :, :]
            #hdr = ismrmrd.xsd.CreateFromDocument(
            #    f['ismrmrd_header'][()])
        # cast to torch.complex128:

        # load maps:
        filename_mask = os.path.join(self.mask_dir, self.masks[volume_idx])
        with h5py.File(filename_mask, "r") as f:
            maps = f['s_maps'][slice_idx, :, :, :]
        # cast to torch.complex128:

        kspace = torch.tensor(kspace, dtype=torch.complex128).unsqueeze(0)
        maps_torch = torch.tensor(maps, dtype=torch.complex128).unsqueeze(0)
        # masks are already normalized luckily (S*S^* = 1)

        A = SenseModelFastMRI(maps_torch)
        image = A.AT(kspace)

        image = torchvision.transforms.functional.center_crop(image, self.image_size)

        if self.normalize:
            magn_image = torch.abs(image)
            norm_constant = torch.quantile(magn_image, self.percentile)# + 1e-8
            kspace = kspace / (norm_constant)
            image = image / (norm_constant)
        else:
            norm_constant = 1

        #maps = maps.numpy()[0]

        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # new: we need to normalize the maps such that sum(S*torch.conj(S)) = 1
        factor = np.sqrt(np.sum(np.abs(maps)**2, axis=(0)))
        maps = maps / factor

        maps = torch.from_numpy(maps)

        annotation = torch.zeros((1, 5))-1
        if self.annotation_dir is not None:
            annotation_name = f"{self.volumes[volume_idx].split('.')[0]}_{slice_idx:03}.txt"
            # read annotation txt file:
            complete_annotation_name = os.path.join(self.annotation_dir, annotation_name)
            if os.path.exists(complete_annotation_name):
                annotation = torch.from_numpy(np.loadtxt(complete_annotation_name))
                if annotation.dim() == 1:
                    annotation = annotation.unsqueeze(0)
        kspace = kspace.squeeze(0)
        maps = maps.squeeze(0)
        #mask = mask.squeeze(0)
        slice_name = f"{self.volumes[volume_idx].split('.')[0]}_{slice_idx:03}"

        return image, maps, norm_constant, in_the_middle, slice_idx, volume_idx, annotation, slice_name, rec_rss.shape, rec_rss

    def get_idx_from_volume_and_slice(self, volume_name, slice_idx):
        volume_idx = self.volume_names.index(volume_name)
        if volume_idx > 0:
            idx = slice_idx + self.cumulative_slices[volume_idx - 1]
        else:
            idx = slice_idx
        return idx

    def get_idx_with_annotations(self, patients=None, annotated_classes=None):
        idxs = []
        volume_names = []
        slice_idxs = []
        annotation_files = [x for x in os.listdir(self.annotation_dir) if x.endswith('.txt')]
        for annotation_file in annotation_files:
            if annotated_classes is not None:
                # load annotation file and check if it contains any of the annotated classes
                annotation = np.loadtxt(os.path.join(self.annotation_dir, annotation_file))
                if len(annotation.shape) == 1:
                    this_classes = [int(annotation[0])]
                else:
                    this_classes = [int(x) for x in annotation[:, 0]]
                if not any([x in annotated_classes for x in this_classes]):
                    continue
            volume_name = '_'.join(annotation_file.split('_')[:-1])
            slice_idx = int(annotation_file.split('_')[-1].split('.')[0])
            if patients is not None:
                # intersection of patients and volumes
                if volume_name in patients:
                    idx = self.get_idx_from_volume_and_slice(volume_name, slice_idx)
                    idxs.append(idx)
                    volume_names.append(volume_name)
                    slice_idxs.append(slice_idx)
            else:
                if volume_name in self.volume_names:
                    # to filter for train or val sets
                    idx = self.get_idx_from_volume_and_slice(volume_name, slice_idx)
                    idxs.append(idx)
                    volume_names.append(volume_name)
                    slice_idxs.append(slice_idx)
        return idxs, volume_names, slice_idxs

    def filter_dataset_by_vol_and_slice(self, file_names):
        idxs = []
        for file_name in file_names:
            volume_name = file_name.split('_')[0]
            slice_idx = int(file_name.split('_')[1].split('.')[0])
            if volume_name in self.volume_names:
                # to filter for train or val sets
                idx = self.get_idx_from_volume_and_slice(volume_name, slice_idx)
                idxs.append(idx)
        return idxs


def prepare_dataset(dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee', split='train', normalized=False,
                    outdir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_train_preprocessed'):
    """
    Prepare the dataset for training.
    """
    dataset = GetMRI_Fastmri(dataset_root=dataset_root, split=split, acc=0.0, normalize=normalized,
                             annotation_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/fastmri-plus/Annotations/labels_yolo')

    # dataset from 26100:
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (kspace, maps, mask, norm_constant, image, in_the_middle, slice_idx, volume_idx, annotation) in enumerate(loader):
        volume_name = dataset.volumes[volume_idx].split('.')[0]
        filename = os.path.join(outdir, f'{volume_name}_{slice_idx.item():03}_{in_the_middle.item()}.npy')
        #image = torch.concat((torch.real(image), torch.imag(image)), dim=1)
        #image = image.squeeze(0).numpy()
        if not os.path.exists(filename):
            image = image.squeeze(0).numpy()
            image = np.complex64(image)
            np.save(filename, image)
        if i % 100 == 0:
            print(f'Processed {i} slices')
    return dataset


class GetMRIBrain(GetMRI_Fastmri):
    def __init__(self, dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/brain', split='train', acc=0.0, normalize=False,
                 annotation_dir=None, only_annotations=False, percentile=0.99, image_size=[320, 320]):
        super().__init__(dataset_root=dataset_root, split=split, acc=acc, normalize=normalize,
                 annotation_dir=annotation_dir, only_annotations=only_annotations, percentile=percentile,
                         image_size=image_size)
        self.df = pd.read_csv('/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/fastmri-plus/Annotations/brain.csv')

        #import yaml file:
        import yaml
        with open('./data/label_map_brain.yaml') as file:
            self.label_map = yaml.load(file, Loader=yaml.FullLoader)

    def normalize_bbox(self, x, y, width, height, img_width, img_height):
        x_center = (x+ width / 2) / img_width
        y_center = (y+ height / 2) / img_height
        width_normalized = width / img_width
        height_normalized = height / img_height
        normalized_bboxes = torch.stack([x_center, y_center, width_normalized, height_normalized])
        return normalized_bboxes.T

    def adjust_normalized_bboxes(self, bboxes, orig_width, orig_height, crop_width, crop_height):
        # Calculate normalized crop offsets and sizes
        x_offset = (crop_width - orig_width) / 2
        y_offset = (crop_height - orig_height) / 2

        # Adjust bounding boxes
        adjusted_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox

            # Adjust center coordinates
            x_new = (x + x_offset)
            y_new = (y + y_offset)

            # Scale width and height
            w_new = w
            h_new = h

            # Ensure the bounding box is within the cropped region
            # if 0 <= x_new <= 1 and 0 <= y_new <= 1:
            adjusted_bboxes.append([x_new, y_new, w_new, h_new])

        return torch.tensor(adjusted_bboxes)

    def xywh2xyxy(self, boxes):
        new_boxes = []
        for box in boxes:
            x1, y1, w, h = box
            box = [x1, y1, x1 + w, y1 + h]
            new_boxes.append(box)
        return new_boxes

    def __getitem__(self, idx):
        volume_idx = next(i for i, v in enumerate(self.cumulative_slices) if idx < v)
        if volume_idx > 0:
            slice_idx = idx - self.cumulative_slices[volume_idx - 1]
            slices_in_volume = self.cumulative_slices[volume_idx] - self.cumulative_slices[volume_idx - 1]
        else:
            slice_idx = idx
            slices_in_volume = self.cumulative_slices[volume_idx]
        if slice_idx > 5 and slice_idx < slices_in_volume - 5:
            in_the_middle = 1
        else:
            in_the_middle = 0
        filename = os.path.join(self.data_dir, self.volumes[volume_idx])
        with h5py.File(filename, "r") as f:
            # scan has keys: ['ismrmrd_header', 'kspace', 'reconstruction_rss']
            kspace = f['kspace'][slice_idx, :, :, :]
            #attrs = dict(f.attrs)
            rec_rss = f['reconstruction_rss'][slice_idx, :, :]
            #hdr = ismrmrd.xsd.CreateFromDocument(
            #    f['ismrmrd_header'][()])
        # cast to torch.complex128:

        # load maps:
        filename_mask = os.path.join(self.mask_dir, self.masks[volume_idx])
        with h5py.File(filename_mask, "r") as f:
            maps = f['s_maps'][slice_idx, :, :, :]
        # cast to torch.complex128:

        kspace = torch.tensor(kspace, dtype=torch.complex128).unsqueeze(0)
        maps_torch = torch.tensor(maps, dtype=torch.complex128).unsqueeze(0)
        # masks are already normalized luckily (S*S^* = 1)

        A = SenseModelFastMRI(maps_torch)
        image = A.AT(kspace)

        image = torchvision.transforms.functional.center_crop(image, self.image_size)

        if self.normalize:
            magn_image = torch.abs(image)
            norm_constant = torch.quantile(magn_image, self.percentile)# + 1e-8
            kspace = kspace / (norm_constant)
            image = image / (norm_constant)
        else:
            norm_constant = 1

        #maps = maps.numpy()[0]

        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # new: we need to normalize the maps such that sum(S*torch.conj(S)) = 1
        factor = np.sqrt(np.sum(np.abs(maps)**2, axis=(0)))
        maps = maps / factor

        maps = torch.from_numpy(maps)



        #if self.acc > 0:
        #    mask = get_mask(image.unsqueeze(0), type='uniform1d', size=kspace.shape[-1], batch_size=1, acc_factor=self.acc)
        #else:
        #    mask = torch.ones_like(kspace[:, :1])
        #mask = mask.abs().int()

        #annotation = torch.zeros((1, 5))-1
        #if self.annotation_dir is not None:
        #    annotation_name = f"{self.volumes[volume_idx].split('.')[0]}_{slice_idx:03}.txt"
        #    # read annotation txt file:
        #    complete_annotation_name = os.path.join(self.annotation_dir, annotation_name)
        #    if os.path.exists(complete_annotation_name):
        #        annotation = torch.from_numpy(np.loadtxt(complete_annotation_name))
        #        if annotation.dim() == 1:
        #            annotation = annotation.unsqueeze(0)
        # filter self.df for volume_name and slice_idx:
        volume_name = self.volumes[volume_idx].split('.')[0]
        df_here = self.df[(self.df['file'] == volume_name) & (self.df['slice'] == slice_idx)]

        boxes = df_here.iloc[:][['x', 'y', 'width', 'height']].values

        new_boxes = self.adjust_normalized_bboxes(boxes, orig_width=rec_rss.shape[1], orig_height=rec_rss.shape[0],
                                 crop_width=self.image_size[1], crop_height=self.image_size[0])

        normalized_boxes = self.normalize_bbox(new_boxes[:, 0], new_boxes[:, 1], new_boxes[:, 2], new_boxes[:, 3], self.image_size[1], self.image_size[0])

        pathology_class = df_here['label'].values
        pathology_class = [self.label_map[x] for x in pathology_class]

        # concat to normalized_boxes:
        annotation = torch.cat((torch.tensor(pathology_class).unsqueeze(1), normalized_boxes), dim=1)


        kspace = kspace.squeeze(0)
        maps = maps.squeeze(0)
        #mask = mask.squeeze(0)
        slice_name = f"{self.volumes[volume_idx].split('.')[0]}_{slice_idx:03}"

        return image, maps, norm_constant, in_the_middle, slice_idx, volume_idx, annotation, slice_name, rec_rss.shape, rec_rss


def prepare_dataset_320(dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee', split='train', normalized=False,
                    outdir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_train_preprocessed_320'):
    """
    Prepare the dataset for training.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_dir = os.path.join(dataset_root, f"multicoil_{split}")
    map_dir = os.path.join(dataset_root, f'multicoil_{split}_maps')
    files = [x for x in os.listdir(data_dir) if x.endswith('.h5')]
    dataset = MVU_Estimator_Knees(file_list=files, maps_dir=map_dir, input_dir=data_dir, R=1, image_size=(320, 320))


    # dataset from 26100:
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, item in enumerate(loader):
        volume_name = item['mvue_file'][0].split('.')[0].split('/')[-1]
        filename = os.path.join(outdir, f"{volume_name}_{item['slice_idx'].item():03}_{item['in_the_middle'].item()}.npy")
        image = item['mvue']
        #image = torch.concat((torch.real(image), torch.imag(image)), dim=1)
        #image = image.squeeze(0).numpy()
        if not os.path.exists(filename):
            image = image.squeeze(0).numpy()
            image = np.complex64(image)
            np.save(filename, image)
        if i % 100 == 0:
            print(f'Processed {i} slices')
    return dataset


def get_info_about_slices(dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee', split='train'):
    """
    This function returns a pandas dataframe with the following columns:
    - volume_id
    - number_of_slices
    - slice_shape
    - 99th_percentile
    """
    data_dir = os.path.join(dataset_root, f"multicoil_{split}")
    volumes = [x for x in os.listdir(data_dir) if x.endswith('.h5')]
    data = []
    for volume_id, volume in enumerate(volumes):
        #if volume_id < 4000:
        #    continue
        scan = h5py.File(os.path.join(data_dir, volume))
        kspace = scan['kspace'][:, 0, :, :]
        shape_rss = scan['reconstruction_rss'][0, :, :].shape

        data.append({
            'volume_id': volume.split('.')[0],
            'number_of_slices': kspace.shape[0],
            #'slice_shape': kspace.shape[-2:],
            'slice_height': int(kspace.shape[-2]),
            'slice_width': int(kspace.shape[-1]),
            'reconstruction_height': shape_rss[0],
            'reconstruction_width': shape_rss[1],
        })
        print(f'Volume {volume_id} has {kspace.shape[0]} slices')
        if (volume_id+1) % 2000 == 0:
            print(f'Processed {volume_id+1} volumes')
            pd.DataFrame(data).to_csv(f'{dataset_root}/multicoil_{split}/volume_info_{volume_id}.csv')
            data = []
    return pd.DataFrame(data)

"""
class TestGetMRI(unittest.TestCase):
    def test_get_mri(self):
        dataset = GetMRI(split='test', acc=6.0, echo=1, normalize=True, seg_dir='/mnt/lustre/work/baumgartner/jmorshuis45/projects/guided-diffusion/predictions/seg_test_ddim100_acc8.0_fulldcTrue')
        item = dataset[0]

    def test_get_mri_no_seg(self):
        dataset = GetMRI(split='train', acc=6.0, echo=1, normalize=True,
                         seg_dir='/mnt/lustre/work/baumgartner/jmorshuis45/projects/guided-diffusion/predictions/seg_test_ddim100_acc8.0_fulldcTrue')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)
        for i, data in enumerate(dataloader):
            kspace, kspace_us, maps, mask, target, segmentation, norm_constant, scan_name = data
            print(kspace.shape, kspace_us.shape, maps.shape, mask.shape, target.shape, segmentation, norm_constant,
                  scan_name)
            self.assertEqual(kspace.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(kspace_us.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(maps.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(mask.shape, torch.Size([1, 512, 160, 1]))
            self.assertEqual(target.shape, torch.Size([1, 512, 160, 1]))
"""

def copy_annotated_files(annotation_dir, data_dir, output_dir):
    """
    Copy annotated files to output_dir
    """
    annotation_files = [x for x in os.listdir(annotation_dir) if x.endswith('.txt')]
    for annotation_file in annotation_files:
        volume_name = annotation_file.split('_')[0]
        slice_idx = int(annotation_file.split('_')[1].split('.')[0])
        filename1 = f"{volume_name}_{slice_idx:03}_1.npy"
        filename0 = f"{volume_name}_{slice_idx:03}_0.npy"
        if os.path.exists(os.path.join(data_dir, filename0)) and not os.path.exists(os.path.join(output_dir, filename0)):
            os.system(f'cp {os.path.join(data_dir, filename0)} {output_dir}')
        if os.path.exists(os.path.join(data_dir, filename1)) and not os.path.exists(os.path.join(output_dir, filename1)):
            os.system(f'cp {os.path.join(data_dir, filename1)} {output_dir}')

"""
class TestDatasetMRI(unittest.TestCase):
    def test_prepare_dataset(self):
        dataset = GetMRI_Fastmri(dataset_root='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee', split='val', acc=0.0, normalize=True,
                                 annotation_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/fastmri-plus/Annotations/labels_yolo')
        item = dataset[15]
        idxs = dataset.get_idx_with_annotations(annotated_classes=[6])
        print('ok')



class TestDataset(unittest.TestCase):
    def test_prepare_dataset(self):
        dataset = MVU_Estimator_Knees(file_list=['/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_val/file1000000.h5'],
                                        maps_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_val_maps',
                                        input_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_val',
                                        project_dir='./',
                                        R=1,
                                        image_size=(320, 320),
                                        acs_size=26,
                                        pattern='random',
                                        orientation='vertical')
        item = dataset[15]
        print('ok')

"""
if __name__ == "__main__":
    #prepare_dataset_320()
    #copy_annotated_files(annotation_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/fastmri-plus/Annotations/labels_yolo',
    #                     data_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_val_preprocessed',
    #                     output_dir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastmri/knee/multicoil_val_preprocessed_annotated'
    #                     )
    #prepare_dataset(split='val', outdir='/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/knee/multicoil_val_preprocessed')
    dataset_root = '/mnt/lustre/work/baumgartner/jmorshuis45/data/fastMRI/brain'
    split = 'val'
    df = get_info_about_slices(dataset_root=dataset_root, split=split)
    # save df to csv:
    df.to_csv(f'{dataset_root}/multicoil_{split}/volume_info.csv')
    print('ok')
