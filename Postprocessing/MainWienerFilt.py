
"""MainWienerFilt.py

Translated from MainWienerFilt.m + Func_Wiener_Filter.m (ref: Chen et al, https://www.nature.com/articles/nature12009) into Python.

Uses: numpy, scipy, hyperspy, matplotlib

main block loads a .mat, .rec or .tif file (if .mat, expects key named 'Reconstruction' with the recon data), creates a mask, runs the filter, normalises and saves a TIFF stack and a Hyperspy '.hspy' file.
"""

from __future__ import annotations

import numpy as np

import scipy.io
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

import hyperspy.api as hs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # not used but needed for if 3D projection plots

import os

def func_wiener_filter(RRR: np.ndarray, SSS: np.ndarray, lam: float) -> np.ndarray:
	"""Apply Wiener filter to 3D reconstruction (port of Func_Wiener_Filter.m).

	Parameters:
	    RRR : ndarray
	        3D reconstruction array (Tsize x Tsize x Tsize).
	    SSS : ndarray
	        Binary mask (same shape as RRR); non-zero indicates valid region.
	    lam : float
	        Lambda parameter used in the original code.

	Returns:
	    PPPM : ndarray
	        Filtered 3D volume (same shape as RRR).
	"""
	# Ensure float
	RRR = np.asarray(RRR, dtype=float)

	# Accept non-cubic inputs: compute per-axis sizes and centers
	nx, ny, nz = RRR.shape
	# compute floating-point centers and integer center indices for each axis
	cx = (nx - 1) / 2.0
	cy = (ny - 1) / 2.0
	cz = (nz - 1) / 2.0
	center = (cx, cy, cz)
	center_idx = (int(round(cx)), int(round(cy)), int(round(cz)))

	# Threshold small values as in MATLAB
	thresh = 0.001 * RRR.max()
	RRR[RRR < thresh] = 0

	# Diagnostic slice projection location: choose center slice along Z
	PZ = int(round(cz))
	Th = 3

	# Power spectrum intensity
	III = (np.abs(fftshift(fftn(RRR)))) ** 2

	# Build mask BS with a radius Rin removed (not used later in MATLAB result)
	Rin = 40
	# Build coordinate grids for arbitrary shapes
	coords_x = np.arange(nx) - cx
	coords_y = np.arange(ny) - cy
	coords_z = np.arange(nz) - cz
	X, Y, Z = np.meshgrid(coords_x, coords_y, coords_z, indexing="ij")
	Rgrid = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
	BS = np.ones_like(Rgrid)
	BS[Rgrid <= Rin] = 0

	DDD = III

	# Compute max radial index based on the current volume shape
	max_rad = int(np.floor(np.sqrt(cx ** 2 + cy ** 2 + cz ** 2))) + 2

	Inoise = np.zeros(max_rad, dtype=float)
	Cnoise = np.zeros(max_rad, dtype=float)

	# Accumulate radial averages
	# round distances to nearest integer to index bins (MATLAB uses round(D)+1)
	# Use vectorised approach for speed
	Dflat = Rgrid.ravel()
	Dvals = np.round(Dflat).astype(int)
	IIIflat = DDD.ravel()
	# Clip indices to valid range
	Dvals_clipped = np.clip(Dvals, 0, max_rad - 1)
	for idx, val in enumerate(IIIflat):
		if val != 0:
			bin_idx = Dvals_clipped[idx]
			Inoise[bin_idx] += val
			Cnoise[bin_idx] += 1

	# Compute averages where counts > 0
	InoiseAVE = np.zeros_like(Inoise)
	nonzero = Cnoise != 0
	InoiseAVE[nonzero] = Inoise[nonzero] / Cnoise[nonzero]

	# Smooth with window size 3 (simple moving average) similar to MATLAB smooth(...,3)
	kernel = np.ones(3) / 3.0
	InoiseAVE = np.convolve(InoiseAVE, kernel, mode="same")

	# Create NNN volume by interpolating InoiseAVE at the radial distances
	# Use np.interp, which will extrapolate the last value for out-of-range
	radial_positions = np.arange(len(InoiseAVE))
	# Flatten and interpolate
	NNN_flat = np.interp(Dflat, radial_positions, InoiseAVE, right=InoiseAVE[-1])
	NNN = NNN_flat.reshape(Rgrid.shape)

	# FFT of original
	KKK = fftshift(fftn(RRR))
	AAA = (np.abs(fftshift(fftn(RRR)))) ** 2

	PPP = AAA - lam * NNN
	PPP[PPP < 0] = 0
	# Avoid division by zero by adding tiny epsilon where needed
	denom = PPP + lam * NNN
	with np.errstate(divide="ignore", invalid="ignore"):
		FL = np.zeros_like(PPP)
		mask = denom != 0
		FL[mask] = PPP[mask] / denom[mask]

	KKKM = FL * KKK
	# Zero DC component
	# Zero the DC / central voxel using the computed 3D index
	KKKM[center_idx[0], center_idx[1], center_idx[2]] = 0

	PPPM = np.real(ifftn(ifftshift(KKKM)))
	PPPM[SSS == 0] = 0

	return PPPM

def _normalize_to_uint16(volume: np.ndarray) -> np.ndarray:
	"""Normalise volume to range [0, 65535] and cast to uint16.

	Parameters:
	    volume : ndarray
	        Input volume array (any dtype).

	Returns:
	    ndarray
	        Normalised volume as uint16 (range [0, 65535]).
	"""
	vol = volume.copy()
	vol = vol - vol.min()
	maxv = vol.max()
	if maxv == 0:
		return np.zeros_like(vol, dtype=np.uint16)
	vol = vol / maxv
	return (vol * 65535.0).astype(np.uint16)


def main():
	from pathlib import Path
	# Load input according to file extension. Supports:
	# .mat  : scipy.io.loadmat (expects 'Reconstruction' or picks largest ndarray)
	# .tif / .tiff : hyperspy load
	# .mrc / .rec  : mrcfile (if available)
	DATA_DIR = Path("/YOUR/WORKING/PATH/HERE")
	mat_filename = "INPUT_FILENAME.tif" # change this and above line to path and name of input file
	mat_path = DATA_DIR / mat_filename
	
	path = Path(mat_path)
	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {mat_path}")

	ext = path.suffix.lower()
	if ext == ".mat":
		mat = scipy.io.loadmat(mat_path)
		if "Reconstruction" in mat:
			AAA = mat["Reconstruction"]
		else:
			arrays = {k: v for k, v in mat.items() if isinstance(v, np.ndarray)}
			if not arrays:
				raise KeyError("No arrays found in .mat file")
			AAA = max(arrays.values(), key=lambda x: x.size)
		AAA = AAA.astype(float)
		AAA = AAA + AAA.min()

	elif ext in (".tif", ".tiff"):
		# Use hyperspy to load TIFF stacks
		hs_obj = hs.load(mat_path)
		# hyperspy signal typically stores data in .data
		try:
			AAA = hs_obj.data.copy()
		except Exception:
			# convert to numpy
			AAA = np.asarray(hs_obj)
		AAA = AAA.astype(float)
		AAA = AAA + AAA.min()

	elif ext in (".mrc", ".rec"):
		# Try mrcfile if available
		try:
			import mrcfile
			with mrcfile.open(mat_path, permissive=True) as m:
				AAA = m.data.copy()
		except Exception as e:
			raise RuntimeError(f"Could not read MRC/REC file: {e}")
		AAA = AAA.astype(float)
		AAA = AAA + AAA.min()

	else:
		raise ValueError(f"Unsupported file extension: {ext}")

	# Create binary mask SSS similar to MATLAB
	SSS = (AAA != 0).astype(np.uint8)

	# Call translated Wiener filter
	lam = 3.0
	PPPM = func_wiener_filter(AAA, SSS, lam)

	# Normalise and save as multi-page TIFF
	image_stack = _normalize_to_uint16(PPPM)
	dir_name = os.path.dirname(mat_path) or os.getcwd()
	base = os.path.splitext(os.path.basename(mat_path))[0]
	
	out_tiff = os.path.join(dir_name, f"{base}_wiener3.tif")
	#hs_sig_tiff = hs.signals.Signal2D(image_stack)
	#hs_sig_tiff.save(out_tiff, overwrite=True)
	out_hspy = os.path.join(dir_name, f"{base}_wiener3.hspy")
	hs_signal_out = hs.signals.Signal2D(PPPM)
	hs_signal_out.save(out_hspy, overwrite=True)

	#print("Saved Wiener TIFF:", out_tiff)
	print("Saved Wiener Hyperspy file:", out_hspy)

if __name__ == "__main__":
	main()



