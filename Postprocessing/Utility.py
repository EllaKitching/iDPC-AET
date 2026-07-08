""" file containing utility scripts e.g. file conversion"""

import os
import glob
import numpy as np
import tifffile as tiff
from scipy.io import loadmat
import h5py

def convert_to_uint16(vol):
    """Convert a NumPy array to uint16 for TIFF, handling complex and structured arrays."""
    # complex arrays from MATLAB v7.3
    if hasattr(vol.dtype, 'names') and vol.dtype.names == ('real', 'imag'):
        vol = np.sqrt(vol['real']**2 + vol['imag']**2)

    # standard complex arrays
    if np.iscomplexobj(vol):
        vol = np.abs(vol)

    # scale floats to uint16
    if np.issubdtype(vol.dtype, np.floating):
        vol = (vol / vol.max() * 65535).astype(np.uint16)
    else:
        vol = vol.astype(np.uint16)

    return vol

def save_3d_arrays_from_mat(path, outDir):
    """Load a MAT file (classic or v7.3) and save all 3D+ arrays as TIFF stacks."""
    try:
        # Attempt v7.3 HDF5
        with h5py.File(path, 'r') as f:
            datasets = []

            # Collect all 3D+ datasets
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 3:
                    datasets.append((name, obj.shape))
            f.visititems(collect_datasets)

            if not datasets:
                print(f"  No 3D+ datasets in {os.path.basename(path)}")
                return

            for name, shape in datasets:
                vol = f[name][:]
                vol = convert_to_uint16(vol)

                clean_name = name.replace('/', '_')
                out_path = os.path.join(
                    outDir,
                    f"{os.path.splitext(os.path.basename(path))[0]}_{clean_name}.tif"
                )
                tiff.imwrite(out_path, vol, compression=None)
                print(f"  Saved {out_path}")

    except (OSError, RuntimeError):
        # MAT<v7.3
        mat = loadmat(path)
        # Remove metadata
        mat = {k:v for k,v in mat.items() if not k.startswith('__')}

        for k, v in mat.items():
            if isinstance(v, np.ndarray) and len(v.shape) >= 3:
                vol = convert_to_uint16(v)
                out_path = os.path.join(
                    outDir,
                    f"{os.path.splitext(os.path.basename(path))[0]}_{k}.tif"
                )
                tiff.imwrite(out_path, vol, compression=None)
                print(f"  Saved {out_path}")
