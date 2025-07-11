from pathlib import Path

import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from physics.fastmri_utils import fft2c_new, ifft2c_new
from statistics import mean, stdev
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sporco.metric import gmsd, mse
from scipy.ndimage import gaussian_laplace
from sigpy.mri import poisson
import functools


def get_sigma(t, sde):
	""" VE-SDE """
	sigma_t = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
	return sigma_t


def pred_x0_from_s(xt, s, t, sde):
	""" Tweedie's formula for denoising. Assumes VE-SDE """
	sigma_t = get_sigma(t, sde)
	tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
	pred_x0 = xt + (tmp ** 2) * s
	return pred_x0


def recover_xt_from_x0(x0_t, s, t, sde):
	sigma_t = get_sigma(t, sde)
	tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
	xt = x0_t - (tmp ** 2) * s
	return xt


def pred_eps_from_s(s, t, sde):
	sigma_t = get_sigma(t, sde)
	tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
	pred_eps = -tmp * s
	return pred_eps


def CG(A, b, x, n_inner=5, eps=1e-5):
	r = b - A(x)
	p = r
	rsold = torch.matmul(r.view(1, -1), r.view(1, -1).T)

	for i in range(n_inner):
		Ap = A(p)
		a = rsold / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

		# x += a * p
		# r -= a * Ap
		x = x + a * p
		r = r - a * Ap

		rsnew = torch.matmul(r.view(1, -1), r.view(1, -1).T)
		if torch.abs(torch.sqrt(rsnew)) < eps:
			break
		p = r + (rsnew / rsold) * p
		rsold = rsnew
	return x


def shrink(src, lamb):
	return torch.sign(src) * torch.max(torch.abs(src)-lamb, torch.zeros_like(src))


def clear_color(x):
	x = x.detach().cpu().squeeze().numpy()
	return np.transpose(x, (1, 2, 0))


def clear(x):
	x = x.detach().cpu().squeeze().numpy()
	return x


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
	ckpt_dir = Path(ckpt_dir)
	if not ckpt_dir.exists():
		ckpt_dir.mkdir(parents=True)
		logging.error(f"No checkpoint found at {ckpt_dir}. "
					  f"Returned the same state as input")
		FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
		return state
	else:
		loaded_state = torch.load(ckpt_dir, map_location=device)
		if not skip_optimizer:
			state['optimizer'].load_state_dict(loaded_state['optimizer'])
		loaded_model_state = loaded_state['model']
		if skip_sigma:
			loaded_model_state.pop('module.sigmas')

		state['model'].load_state_dict(loaded_model_state, strict=False)
		state['ema'].load_state_dict(loaded_state['ema'])
		state['step'] = loaded_state['step']
		print(f'loaded checkpoint dir from {ckpt_dir}')
		return state


def save_checkpoint(ckpt_dir, state):
	saved_state = {
		'optimizer': state['optimizer'].state_dict(),
		'model': state['model'].state_dict(),
		'ema': state['ema'].state_dict(),
		'step': state['step']
	}
	torch.save(saved_state, ckpt_dir)


"""
Helper functions for new types of inverse problems
"""


def fft2(x):
	""" FFT with shifting DC to the center of the image"""
	return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
	""" IFFT with shifting DC to the corner of the image prior to transform"""
	return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
	""" FFT for multi-coil """
	return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
	""" IFFT for multi-coil """
	return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def crop_center(img, cropx, cropy):
	c, y, x = img.shape
	startx = x // 2 - (cropx // 2)
	starty = y // 2 - (cropy // 2)
	return img[:, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
	""" Normalize img in arbitrary range to [0, 1] """
	img -= torch.min(img)
	img /= torch.max(img)
	return img


def normalize_np(img):
	""" Normalize img in arbitrary range to [0, 1] """
	img -= np.min(img)
	img /= np.max(img)
	return img


def normalize_np_kwarg(img, maxv=1.0, minv=0.0):
	""" Normalize img in arbitrary range to [0, 1] """
	img -= minv
	img /= maxv
	return img


def normalize_complex(img):
	""" normalizes the magnitude of complex-valued image to range [0, 1] """
	abs_img = normalize(torch.abs(img))
	# ang_img = torch.angle(img)
	ang_img = normalize(torch.angle(img))
	return abs_img * torch.exp(1j * ang_img)


def batchfy(tensor, batch_size):
	n = len(tensor)
	num_batches = n // batch_size + 1
	return tensor.chunk(num_batches, dim=0)


def img_wise_min_max(img):
	img_flatten = img.view(img.shape[0], -1)
	img_min = torch.min(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)
	img_max = torch.max(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)

	return (img - img_min) / (img_max - img_min)


def patient_wise_min_max(img):
	std_upper = 3
	img_flatten = img.view(img.shape[0], -1)

	std = torch.std(img)
	mean = torch.mean(img)

	img_min = torch.min(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)
	img_max = torch.max(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)

	min_max_scaled = (img - img_min) / (img_max - img_min)
	min_max_scaled_std = (std - img_min) / (img_max - img_min)
	min_max_scaled_mean = (mean - img_min) / (img_max - img_min)

	min_max_scaled[min_max_scaled > min_max_scaled_mean +
				   std_upper * min_max_scaled_std] = 1

	return min_max_scaled


def create_sphere(cx, cy, cz, r, resolution=256):
	'''
	create sphere with center (cx, cy, cz) and radius r
	'''
	phi = np.linspace(0, 2 * np.pi, 2 * resolution)
	theta = np.linspace(0, np.pi, resolution)

	theta, phi = np.meshgrid(theta, phi)

	r_xy = r * np.sin(theta)
	x = cx + np.cos(phi) * r_xy
	y = cy + np.sin(phi) * r_xy
	z = cz + r * np.cos(theta)

	return np.stack([x, y, z])


class lambda_schedule:
	def __init__(self, total=2000):
		self.total = total

	def get_current_lambda(self, i):
		pass


class lambda_schedule_linear(lambda_schedule):
	def __init__(self, start_lamb=1.0, end_lamb=0.0):
		super().__init__()
		self.start_lamb = start_lamb
		self.end_lamb = end_lamb

	def get_current_lambda(self, i):
		return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
	def __init__(self, lamb=1.0):
		super().__init__()
		self.lamb = lamb

	def get_current_lambda(self, i):
		return self.lamb


def image_grid(x, sz=32):
	size = sz
	channels = 3
	img = x.reshape(-1, size, size, channels)
	w = int(np.sqrt(img.shape[0]))
	img = img.reshape((w, w, size, size, channels)).transpose(
		(0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
	return img


def show_samples(x, sz=32):
	x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
	img = image_grid(x, sz)
	plt.figure(figsize=(8, 8))
	plt.axis('off')
	plt.imshow(img)
	plt.show()


def image_grid_gray(x, size=32):
	img = x.reshape(-1, size, size)
	w = int(np.sqrt(img.shape[0]))
	img = img.reshape((w, w, size, size)).transpose(
		(0, 2, 1, 3)).reshape((w * size, w * size))
	return img


def show_samples_gray(x, size=32, save=False, save_fname=None):
	x = x.detach().cpu().numpy()
	img = image_grid_gray(x, size=size)
	plt.figure(figsize=(8, 8))
	plt.axis('off')
	plt.imshow(img, cmap='gray')
	plt.show()
	if save:
		plt.imsave(save_fname, img, cmap='gray')


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
	mux_in = size ** 2
	if type.endswith('2d'):
		Nsamp = mux_in // acc_factor
	elif type.endswith('1d'):
		Nsamp = size // acc_factor
		Nsamp_center = int(size * center_fraction)
		c_from = size // 2 - Nsamp_center // 2
		p = np.ones(size)
		p[c_from:c_from + Nsamp_center] = 0
		p /= p.sum()
	if type == 'gaussian2d':
		mask = torch.zeros_like(img)
		cov_factor = size * (1.5 / 128)
		mean = [size // 2, size // 2]
		cov = [[size * cov_factor, 0], [0, size * cov_factor]]
		if fix:
			samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
			int_samples = samples.astype(int)
			int_samples = np.clip(int_samples, 0, size - 1)
			mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
		else:
			for i in range(batch_size):
				# sample different masks for batch
				samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
				int_samples = samples.astype(int)
				int_samples = np.clip(int_samples, 0, size - 1)
				mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
	elif type == 'uniformrandom2d':
		mask = torch.zeros_like(img)
		if fix:
			mask_vec = torch.zeros([1, size * size])
			samples = np.random.choice(size * size, int(Nsamp))
			mask_vec[:, samples] = 1
			mask_b = mask_vec.view(size, size)
			mask[:, ...] = mask_b
		else:
			for i in range(batch_size):
				# sample different masks for batch
				mask_vec = torch.zeros([1, size * size])
				samples = np.random.choice(size * size, int(Nsamp))
				mask_vec[:, samples] = 1
				mask_b = mask_vec.view(size, size)
				mask[i, ...] = mask_b
	elif type == 'gaussian1d':
		mask = torch.zeros_like(img)
		mean = size // 2
		std = size * (15.0 / 128)
		Nsamp_center = int(size * center_fraction)
		if fix:
			samples = np.random.normal(
				loc=mean, scale=std, size=int(Nsamp * 1.2))
			int_samples = samples.astype(int)
			int_samples = np.clip(int_samples, 0, size - 1)
			mask[..., int_samples] = 1
			c_from = size // 2 - Nsamp_center // 2
			mask[..., c_from:c_from + Nsamp_center] = 1
		else:
			for i in range(batch_size):
				samples = np.random.normal(
					loc=mean, scale=std, size=int(Nsamp*1.2))
				int_samples = samples.astype(int)
				int_samples = np.clip(int_samples, 0, size - 1)
				mask[i, :, :, int_samples] = 1
				c_from = size // 2 - Nsamp_center // 2
				mask[i, :, :, c_from:c_from + Nsamp_center] = 1
	elif type == 'uniform1d':
		mask = torch.zeros_like(img)
		if fix:
			samples = np.random.choice(size, int(Nsamp-Nsamp_center), p=p, replace=False)
			mask[..., samples] = 1
			# ACS region
			c_from = size // 2 - Nsamp_center // 2
			mask[..., c_from:c_from + Nsamp_center] = 1
		else:
			for i in range(batch_size):
				samples = np.random.choice(size, int(Nsamp - Nsamp_center), p=p, replace=False)
				mask[i, :, :, samples] = 1
				# ACS region
				c_from = size // 2 - Nsamp_center // 2
				mask[i, :, :, c_from:c_from+Nsamp_center] = 1
	elif type == 'poisson':
		mask = poisson((size, size), accel=acc_factor).astype(np.complex64)
		mask = torch.from_numpy(mask)
	else:
		NotImplementedError(f'Mask type {type} is currently not supported.')

	return mask


def prepare_masks(n_masks = 1000, size=320, center_fraction=0.04):
	Nsamp_center = int(size * center_fraction)
	c_from = size // 2 - Nsamp_center // 2
	all_all_masks = np.zeros((2000, 12, size), dtype=np.bool_)
	for i in range(len(all_all_masks)):
		mask = np.zeros(320)
		mask[c_from:c_from + Nsamp_center] = 1
		acceleration_factors = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
		Nsamp_prev = Nsamp_center
		p = np.ones(size)
		p[c_from:c_from + Nsamp_center] = 0
		p /= p.sum()
		all_masks = np.zeros((len(acceleration_factors), size))
		for c, acc_factor in enumerate(reversed(acceleration_factors)):
			Nsamp = size // acc_factor
			Newsamp = Nsamp - Nsamp_prev
			samples = np.random.choice(size, Newsamp, p=p, replace=False)
			p[samples] = 0
			mask[samples] = 1
			p /= p.sum()
			all_masks[c] = mask
			Nsamp_prev = Nsamp
		all_all_masks[i] = all_masks
	return all_all_masks






def kspace_to_nchw(tensor):
	"""
	Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
	(N, C, H, W) 4D format for processing by 2D CNNs.

	Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

	C is the coils interleaved with real and imaginary values as separate channels.
	C is therefore always 2 * Coil.

	Singlecoil data is assumed to be in the 5D format with Coil = 1

	Args:
		tensor (torch.Tensor): Input data in 5D kspace tensor format.
	Returns:
		tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
	"""
	assert isinstance(tensor, torch.Tensor)
	assert tensor.dim() == 5
	s = tensor.shape
	assert s[-1] == 2
	tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(
		shape=(s[0], 2 * s[1], s[2], s[3]))
	return tensor


def nchw_to_kspace(tensor):
	"""
	Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

	This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
	If the coil dimension is not divisible by 2, the function assumes that the input data is 'real' data,
	and thus pads the imaginary dimension as 0.
	"""
	assert isinstance(tensor, torch.Tensor)
	assert tensor.dim() == 4
	s = tensor.shape
	if tensor.shape[1] == 1:
		imag_tensor = torch.zeros(s, device=tensor.device)
		tensor = torch.cat((tensor, imag_tensor), dim=1)
		s = tensor.shape
	tensor = tensor.view(
		size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
	return tensor


def root_sum_of_squares(data, dim=0):
	"""
	Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
	Args:
		data (torch.Tensor): The input tensor
		dim (int): The dimensions along which to apply the RSS transform
	Returns:
		torch.Tensor: The RSS value
	"""
	return torch.sqrt((data ** 2).sum(dim))


def save_data(fname, arr):
	""" Save data as .npy and .png """
	np.save(fname + '.npy', arr)
	plt.imsave(fname + '.png', arr, cmap='gray')


def mean_std(vals: list):
	return mean(vals), stdev(vals)


def cal_metric(comp, label):
	LoG = functools.partial(gaussian_laplace, sigma=1.5)
	psnr_val = peak_signal_noise_ratio(comp, label)
	ssim_val = structural_similarity(comp, label)
	hfen_val = mse(LoG(comp), LoG(label))
	gmsd_val = gmsd(label, comp)
	return psnr_val, ssim_val, hfen_val, gmsd_val


def _Dz(x): # Batch direction
	y = torch.zeros_like(x)
	y[:-1] = x[1:]
	y[-1] = x[0]
	return y - x


def _DzT(x): # Batch direction
	y = torch.zeros_like(x)
	y[:-1] = x[1:]
	y[-1] = x[0]

	tempt = -(y-x)
	difft = tempt[:-1]
	y[1:] = difft
	y[0] = x[-1] - x[0]

	return y



def real_to_nchw_comp(x):
	"""
	[1, 2, 320, 320] real --> [1, 1, 320, 320] comp
	"""
	if len(x.shape) == 4:
		x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
	elif len(x.shape) == 3:
		x = x[0:1, :, :] + x[1:2, :, :] * 1j
	return x


def comp_to_nchw_real(x):
	"""
	[1, 1, 320, 320] complex --> [1, 2, 320, 320] real
	"""
	x = torch.view_as_real(x)
	x = torch.permute(x, (0, 4, 2, 3, 1))
	x = x.squeeze(dim=-1)
	return x


def PSNR(reconstruction, ground_truth, data_range=None):
	gt = np.asarray(ground_truth)
	mse = np.mean((np.asarray(reconstruction) - gt)**2)
	if mse == 0.:
		return float('inf')
	if data_range is None:
		data_range = np.max(gt) - np.min(gt)
	return 20*np.log10(data_range) - 10*np.log10(mse)


def SSIM(reconstruction, ground_truth, data_range=None):
	gt = np.asarray(ground_truth)
	if data_range is None:
		data_range = np.max(gt) - np.min(gt)
	return structural_similarity(reconstruction, gt, data_range=data_range)



def update_ema(ema_weights, model_weights, alpha):
	"""
	Update the EMA weights on the CPU.
	:param ema_weights: Exponential moving average weights (stored on CPU)
	:param model_weights: Current model weights (may be on GPU)
	:param alpha: Decay factor for EMA
	"""
	for ema_w, model_w in zip(ema_weights, model_weights):
		# Ensure model weights are on CPU for the update
		model_w_cpu = model_w.data.to('cpu')
		ema_w.data.mul_(alpha).add_(model_w_cpu, alpha=1-alpha)
		
		
def apply_ema_weights(model, ema_weights, device):
	"""
	Apply EMA weights to the model for inference.
	:param model: PyTorch model
	:param ema_weights: Dictionary of EMA weights (stored on CPU)
	:param device: Device to perform inference (usually 'cuda')
	"""
	original_weights = {}
	for name, module in model.named_modules():
		if name in ema_weights:
			original_weights[name] = {pname: p.data.clone() for pname, p in module.named_parameters()}
			for pname, p in module.named_parameters():
				# Copy EMA weight to the GPU for inference
				ema_weight = ema_weights[name][pname].to(device)
				p.data.copy_(ema_weight)
	return original_weights

def restore_original_weights(model, original_weights):
	"""
	Restore original weights to the model.
	:param model: PyTorch model
	:param original_weights: Dictionary of original weights
	"""
	for name, module in model.named_modules():
		if name in original_weights:
			for pname, p in module.named_parameters():
				# Original weights are already on the GPU
				p.data.copy_(original_weights[name][pname])


if __name__ == '__main__':
	# We want to pre-create masks for the testing data
	# load yamlfile
	import json
	with open('data/test_patients.json', 'r') as f:
		test_patients = json.load(f)

	masks = prepare_masks(len(test_patients))
	print('ok')
	# save masks as dictionary
	# make masks to dictionary:
	mask_dict = {}
	for i, patient in enumerate(test_patients):
		mask_dict[patient] = masks[i]
	import pickle
	with open('data/test_masks.pkl', 'wb') as f:
		pickle.dump(mask_dict, f)
