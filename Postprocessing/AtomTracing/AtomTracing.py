# Drafting, translating and testing MATLAB code in Python
# Based on UCLA Miao Group files on Atom tracing, which in turn was based on R.Xu, UCLA, 2014.
# Version adapted from is  Y. Yang, C.-C. Chen, M. C. Scott, C. Ophus, R. Xu, A. Pryor Jr., L. Wu, F. Sun, W. Theis, J. Zhou, M. Eisenbach, P. R. C. Kent, R. F. Sabirianov, H. Zeng, P. Ercius and J. Miao, “Deciphering chemical order/disorder and material properties at the single-atom level”, Nature 542, 75-79 (2017). https://dx.doi.org/10.1038/nature21042

import scipy
import numpy as np
import importlib
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import binary_dilation, label, sum_labels, maximum_filter

if importlib.util.find_spec("cupy") and importlib.util.find_spec("cupyx.scipy.ndimage"):
    cp = importlib.import_module("cupy")
    cpndi = importlib.import_module("cupyx.scipy.ndimage")
    HAS_CUPY = True
else:
    cp = None
    cpndi = None
    HAS_CUPY = False

if importlib.util.find_spec("joblib"):
    joblib = importlib.import_module("joblib")
    Parallel = joblib.Parallel
    delayed = joblib.delayed
    HAS_JOBLIB = True
else:
    Parallel = None
    delayed = None
    HAS_JOBLIB = False

import time

class Stats:
    "empty initalisation of stats class - using append to add on each value"
    def __init__(self):
        self.resnorm = []
        self.residual = []
        self.fit = []
        self.peak_bg = []
        self.peak_height = []
        self.peak_FWHM = []
        self.localmax_pos = []

def find_possible_atoms(DataMatrix, Dmin, Dmin_presort, MaxNumberAtoms, BoxSize1, BoxSize2, Th, fit_method='curve_fit', use_gpu=False, stats=True, presort=True, parallel_statsF=False, initial_atom_pos=None):
    """
    Unified atom tracing pipeline, with different implementations.
    For most efficient, GPU, parrallel stats, and lsq recommended.
    Modes:
    - stats: toggle statsI/statsF computation
    - presort: deterministic peak ordering vs adaptive maxima loop
    - parallel_statsF: optional per-atom statsF evaluation
    - use_gpu: GPU-assisted ROI + subtraction pipeline
    - fit_method: 'curve_fit' or lsq for respective scipy function
    - initial_atom_pos: option to load in initial positions. None if from scratch.
    """

    [box1coordinates, BoxCenter1, BoxRadius1, sphere1] = create_box(BoxSize1)
    [box2coordinates, BoxCenter2, BoxRadius2, sphere2] = create_box(BoxSize2)

    if use_gpu and not HAS_CUPY:
        raise ImportError("use_gpu=True requires CuPy and cupyx.scipy.ndimage to be installed.")

    atom_pos = np.zeros((3, MaxNumberAtoms))
    
    i_atom = 0
    
    if initial_atom_pos is not None:
        n_initial = initial_atom_pos.shape[1]
        atom_pos[:, :n_initial] = initial_atom_pos
        i_atom = n_initial
        print(f"Initialised with {n_initial} existing atom positions.")
    
    close_pos = []
    nclose = 0

    Xsize, Ysize, Zsize = DataMatrix.shape
    CurrData = DataMatrix.copy()

    # GPU init
    if use_gpu:
        box1coordinates_gpu = box_to_gpu(box1coordinates)
        box2coordinates_gpu = box_to_gpu(box2coordinates)
        M = find_maxima_3D_GPU(CurrData, sphere1)
    else:
        M = find_maxima_3D(CurrData, sphere1)

    noatomM = np.zeros_like(DataMatrix)

    statsI = Stats() if stats else None
    statsF = Stats() if (stats and parallel_statsF) else None

    # presort vs adpative diverge
    if presort:
        ind_find = np.nonzero(M)
        if np.count_nonzero(M) == 0:
            raise Exception("Error: No peaks within data.")

        xx, yy, zz = ind_find
        peak_sort, ind_sorted, xx_sort, yy_sort, zz_sort = get_peak_values(CurrData, xx, yy, zz, ind_find)
        
        # Filter presorted list to peaks above threshold only
        above_th = peak_sort >= Th
        peak_sort  = peak_sort[above_th]
        ind_sorted = ind_sorted[above_th]
        xx_sort    = xx_sort[above_th]
        yy_sort    = yy_sort[above_th]
        zz_sort    = zz_sort[above_th]

        if len(peak_sort) == 0:
            raise Exception("Error: No peaks above threshold Th within data.")
        else:
            print(f"Peaks above threshold: {len(peak_sort)} / {np.count_nonzero(M)} total maxima")

        peak_iter = range(len(peak_sort))

    else:
        peak_iter = range(int(1e8)) # essentitally inf
    # MAIN LOOP FOR ATOM TRACING
    for i in peak_iter:

        i_atom += 1
        if i_atom > MaxNumberAtoms:
            break

        # peak selection changings with presort vs adaptive
        if presort:
            x, y, z = xx_sort[i], yy_sort[i], zz_sort[i]
        else:
            ind_find = np.nonzero(M)
            if np.count_nonzero(M) == 0:
                break
            xx, yy, zz = ind_find

            peak_sort, ind_sorted, xx_sort, yy_sort, zz_sort = get_peak_values(CurrData, xx, yy, zz, ind_find)

            x, y, z = xx_sort[0], yy_sort[0], zz_sort[0]
        
        #print(f"Atom number:{i_atom:04d}, candidate located at ({x}, {y}, {z})")
        # Prescreen: skip if already within Dmin of an accepted atom
        if i_atom > 1:
            #print(f"DEBUG PRINT: Comparing against {i_atom-1} accepted atoms: {atom_pos[:, :i_atom-1]}") # toggle this on if needed to troubleshoot
            d_pre, _, nearest_pos = get_minimum_distance(float(x), float(y), float(z), atom_pos[:, :i_atom-1])
            if d_pre < Dmin_presort:
                print(f"Atom number:{i_atom:04d} \t Prescreened: candidate ({x}, {y}, {z}) "
                      f"too close to accepted atom at ({nearest_pos[0]:.2f}, {nearest_pos[1]:.2f}, {nearest_pos[2]:.2f}) "
                      f"(distance={d_pre:.3f} voxels < Dmin_presort={Dmin_presort} voxels)")
                i_atom -= 1
                continue

        # ROI
        DataBox1 = get_data_box(CurrData, x, y, z, BoxRadius1)

        ymax_init = DataBox1[BoxCenter1-1, BoxCenter1-1, BoxCenter1-1]

        fit_param_init = np.array( [0, ymax_init, 0, 0, 0, 0.2, 0.2, 0.8, 0.0, 0.0, 0.0], dtype=float)

        fixed = np.zeros(11)

        lb = np.array([0, 0, -BoxRadius1, -BoxRadius1, -BoxRadius1,
                       0, 0, 0, -np.pi, 0, -np.pi], dtype=float)

        ub = np.array([np.inf, np.inf, BoxRadius1, BoxRadius1, BoxRadius1,
                       np.inf, np.inf, np.inf, np.pi, np.pi, np.pi], dtype=float)

        # fitting
        try:
            fit_resultI, resnorm, residual = _fit_gaussian(
                fit_param_init,
                box1coordinates,
                DataBox1,
                fixed,
                lb,
                ub,
                fit_method=fit_method,
                use_gpu=use_gpu,
                box_coordinates_gpu=box1coordinates_gpu if use_gpu else None
            )
        except RuntimeError:
            i_atom -= 1
            continue

        new_x = x + fit_resultI[2]
        new_y = y + fit_resultI[3]
        new_z = z + fit_resultI[4]

        rx, ry, rz = round(new_x), round(new_y), round(new_z)

        dx, dy, dz = new_x - rx, new_y - ry, new_z - rz

        # constraints applied 
        if i_atom > 1:
            d0, _, _ = get_minimum_distance(new_x, new_y, new_z, atom_pos[:, :i_atom-1])
        else:
            d0 = np.inf

        valid_atom = 1

        if d0 < Dmin:
            valid_atom = 0

        if (new_x < 1 or new_y < 1 or new_z < 1 or
            new_x > Xsize or new_y > Ysize or new_z > Zsize):
            valid_atom = 0

        if valid_atom == 0:
            close_pos.append([new_x, new_y, new_z])
            M[x, y, z] = 0
            i_atom -= 1
            if not presort:
                M[CurrData < Th] = 0
            continue

        #  subtraction pipeline 
        fit_resultI[2:5] = [dx, dy, dz]

        if use_gpu:
            FitBox2 = calc_gauss3D_PD_GPU(fit_resultI, box2coordinates_gpu)
        else:
            FitBox2 = calc_gauss3D_PD(fit_resultI, box2coordinates)

        FitBox2 -= fit_resultI[0]

        DataBox2 = get_data_box(CurrData, rx, ry, rz, BoxRadius2)
        DiffBox2 = np.maximum(DataBox2 - FitBox2, 0)

        CurrData[
            int(rx-BoxRadius2-1):int(rx+BoxRadius2),
            int(ry-BoxRadius2-1):int(ry+BoxRadius2),
            int(rz-BoxRadius2-1):int(rz+BoxRadius2)
        ] = DiffBox2

        # statsI 
        if stats:
            statsI.resnorm.append(resnorm)
            statsI.residual.append(residual.ravel())
            statsI.fit.append(fit_resultI)
            statsI.peak_bg.append(fit_resultI[0])
            statsI.peak_height.append(fit_resultI[1])
            statsI.peak_FWHM.append(2.3548 / (np.sqrt(2) * fit_resultI[5:8]))
            statsI.localmax_pos.append([x, y, z])

        atom_pos[:, i_atom-1] = [new_x, new_y, new_z]

        #  statsF optional 
        if stats and parallel_statsF:
            try:
                roi = get_data_box(CurrData, rx, ry, rz, BoxRadius1)

                fitF, resF, resF_res = _fit_gaussian(
                    fit_param_init,
                    box1coordinates,
                    roi,
                    fixed,
                    lb,
                    ub,
                    fit_method=fit_method,
                    use_gpu=use_gpu,
                    box_coordinates_gpu=box1coordinates_gpu if use_gpu else None
                )

                statsF.resnorm.append(resF)
                statsF.residual.append(resF_res.ravel())
                statsF.fit.append(fitF)
                statsF.peak_bg.append(fitF[0])
                statsF.peak_height.append(fitF[1])
                statsF.peak_FWHM.append(2.3548 / (np.sqrt(2) * fitF[5:8]))
                statsF.localmax_pos.append([rx, ry, rz])

            except RuntimeError:
                pass

    if stats:
        if parallel_statsF:
            return atom_pos, close_pos, statsI, statsF
        return atom_pos, close_pos, statsI

    return atom_pos, close_pos

def create_box(BoxSize): # outputs needed: boxCoordinates, BoxCenter, BoxRadius, sphere
    #ref M. Bartels, UCLA, 2014, Y.Yang, UCLA, 2015.
    #small function to create box coordinates and correpsonding spherical mask - use odd BoxSize
    assert BoxSize % 2 == 1, "BoxSize must be odd"
    BoxCenter = (BoxSize+1)//2  
    BoxRadius = (BoxSize-1)//2 

    #box coordinate systems
    boxX,boxY,boxZ = np.meshgrid(
    np.arange(-BoxRadius, BoxRadius + 1),
    np.arange(-BoxRadius, BoxRadius + 1),
    np.arange(-BoxRadius, BoxRadius + 1),
    indexing = 'ij') 
    
    #calculate spherical mask
    sphere = np.sqrt(boxX**2+boxY**2+boxZ**2)<=BoxSize/2

    # Create boxCoordinates dictionary
    boxCoordinates = {
        'x': boxX,
        'y': boxY,
        'z': boxZ,
    }

    return boxCoordinates, BoxCenter, BoxRadius, sphere

def find_maxima_3D(data, sphere):
    #M. Bartels, UCLA, 2014, Y.Yang, UCLA, 2015.
    #find all maxima of a 3D matrix, with a local neighbourhood defined by sphere mask from create_box
    #reworked to change method of filter.

    #for each voxel, assign the maximum in the neighbourhood
    from scipy.ndimage import maximum_filter
    mat3d_neighbours = maximum_filter(data, size=sphere.shape[0]) 
    #a maximum is where the matrix is larger than all neighbors
    M = data == mat3d_neighbours 
    return M

def find_maxima_3D_GPU(data, sphere):
    """
    find all maxima of a 3D matrix, with a local neighbourhood defined by sphere mask from create_box
    reworked to change method of filter.
    """

    #for each voxel, assign the maximum in the neighbourhood
    if not HAS_CUPY:
        raise ImportError("find_maxima_3D_GPU requires CuPy to be installed.")

    data_gpu = cp.asarray(data)
    mat3d_neighbours = cpndi.maximum_filter(data_gpu, size=sphere.shape[0])
    M = data_gpu == mat3d_neighbours
    M = cp.asnumpy(M)  # bring back for the rest of the loop
    return M

def find_maxima_3D_old(data, sphere):
    """
    find all maxima of a 3D matrix, with a local neighbourhood defined by sphere mask from create_box
    Old verison - good as a fallback. Slower. Edit back into pipeline if facing issues
    """

    col = sphere.shape[0] # specifies mask - do you need to write as sphere? size (mask, 1) specifies size of mask in column dimension
    c = (col+1)//2 # do we need to edit +1 here as matlab counts from 1 instead of 0.
    mask=sphere>0
    mask[c,c,c] = False #excludes the pixel itself from the neighbourhood

    #for each voxel, assign the maximum in the neighbourhood
    #from scipy.ndimage import binary_dilation
    
    mat3d_neighbours = binary_dilation(data, structure=mask) # binary dilation is for binary arrays, grey_dilation works for greyscale images. 
    #a maximum is where the matrix is larger than all neighbors
    M = data > mat3d_neighbours 
    return M

def get_peak_values(CurrData, xx, yy, zz, ind_find):
    peaks = np.zeros(ind_find[0].shape[0])
    for kkk in range(ind_find[0].shape[0]):
        x = int(xx[kkk])
        y = int(yy[kkk])
        z = int(zz[kkk])
        peaks[kkk] = np.sum(CurrData[x-1:x+2, y-1:y+2, z-1:z+2])
    
    ind_sorted = np.argsort(peaks)[::-1]  # Sort indices in descending order
    peak_sort = peaks[ind_sorted]
    xx_sort = xx[ind_sorted] # get coord positions
    yy_sort = yy[ind_sorted]
    zz_sort = zz[ind_sorted]
    return peak_sort, ind_sorted,  xx_sort, yy_sort, zz_sort

def get_data_box(data, x, y, z, BoxRadius):
    """
    Extracts a data box of a set radius from a data matrix.
    Parameters:
        data (numpy.ndarray): The data matrix.
        x, y, z (int): Coordinates of the desired center of the box.
        BoxRadius (int): Radius of the box.

    Returns:
        numpy.ndarray: The data box.
    """
    DataBox = data[x - BoxRadius : x + BoxRadius + 1,
                      y - BoxRadius : y + BoxRadius + 1,
                      z - BoxRadius : z + BoxRadius + 1]
    return DataBox

def _fit_gaussian(fit_param_init, box_coordinates, DataBox, fixed, lb, ub,
                  fit_method='curve_fit', use_gpu=False, box_coordinates_gpu=None):
    """
    'curve_fit' : scipy.optimize.curve_fit via fit_gauss3D_PD (default,
                  matches original MATLAB lsqcurvefit behaviour most closely)
    'lsq'       : scipy.optimize.least_squares via fit_gauss3D_PD_lsq
                  (lower overhead, same algorithm)
    """
    if fit_method == 'lsq':
        return fit_gauss3D_PD_lsq(
            fit_param_init,
            box_coordinates,
            DataBox,
            fixed,
            lb,
            ub,
            use_gpu=use_gpu,
            box_coordinates_gpu=box_coordinates_gpu,
        )
    else:
        return fit_gauss3D_PD(
            fit_param_init,
            box_coordinates,
            DataBox,
            fixed,
            lb,
            ub,
            use_gpu=use_gpu,
            box_coordinates_gpu=box_coordinates_gpu,
        )

def fit_gauss3D_PD(fit_param_init, xdata, ydata, fixed = None, lowerbound = None, upperbound = None,
                   use_gpu=False, box_coordinates_gpu=None):
    """
    Fit 3D gaussian and allow fixed parameters as well as bounds
    Y. Yang, UCLA, 2015, Modified from the original version by M. Bartels, UCLA, 2014 to ensure the Gaussian is always positively defined.
    
    Parameters:
        fit_param_init (numpy.ndarray): The inital parameters.
        x, y, z (int): Coordinates of the desired center of the box.
        BoxRadius (int): Radius of the box.

    Returns:
        numpy.ndarray: The data box.
    """
    
    # Set standard values for optional parameters
    if fixed is None:
        fixed = np.zeros_like(fit_param_init)
    if lowerbound is None:
        lowerbound = np.full_like(fit_param_init, -np.inf)
    if upperbound is None:
        upperbound = np.full_like(fit_param_init, np.inf)

    # Convert to double 
    fit_param_init = fit_param_init.astype(float)
    ydata = ydata.astype(float)

    if use_gpu and not HAS_CUPY:
        raise ImportError("use_gpu=True requires CuPy and cupyx.scipy.ndimage to be installed.")

    # Store all initial parameters
    fit_param_init_all = fit_param_init.copy()

    # Only fit the variable parameters
    fit_param_init = fit_param_init_all[fixed == 0]
    lowerbound = lowerbound[fixed == 0]
    upperbound = upperbound[fixed == 0]

    opt = {'xtol': 1e-12}
    opt['disp'] = 0  # Display is turned off
    ydata = ydata.flatten() # TESTING
    ydata_gpu = cp.asarray(ydata) if use_gpu else None
    
    def Fhelp(xdata, *params): # swapped x, xdata for xdata. *params to take all params
        # Helper function to deal with fixed parameters
        nonlocal fit_param_init_all #ensures the global value doesnt change
        # Merge with fixed parameters
        x_current = params #cant copy tuple
        x = fit_param_init_all.copy()
        x[fixed == 0] = x_current

        # Calculate the actual function
        if use_gpu:
            y_gpu = calc_gauss3D_PDFlat_GPU(x, box_coordinates_gpu if box_coordinates_gpu is not None else xdata)
            return cp.asnumpy(y_gpu)

        y = calc_gauss3D_PDFlat(x, xdata)
        return y

    # Do the fit - necessary changes needed here as lsqcurvefit does not exist in python. 
    x, pcov, infodict, mesg , ier = curve_fit(Fhelp, xdata, ydata, p0=fit_param_init, bounds=(lowerbound, upperbound), method='trf', ftol=1e-12, xtol=1e-12, full_output=True, max_nfev=5000)
    infolist = list(infodict.values())
    
    residual = infolist[1] #fvec is the second value in dicitonary and = residual from matlab
    r = np.linalg.norm(residual) # np.linalg.norm calculates 2-norm - needs to be squared
    resnorm = np.square(r)

    # Add fixed parameters to the final result again
    fit_param_init_all[fixed == 0] = x
    x = fit_param_init_all

    return x, resnorm, residual

def fit_gauss3D_PD_lsq(fit_param_init, xdata, ydata, fixed=None, lowerbound=None, upperbound=None,
                       use_gpu=False, box_coordinates_gpu=None):
    if fixed is None:
        fixed = np.zeros_like(fit_param_init)
    if lowerbound is None:
        lowerbound = np.full_like(fit_param_init, -np.inf)
    if upperbound is None:
        upperbound = np.full_like(fit_param_init, np.inf)

    fit_param_init     = fit_param_init.astype(float)
    ydata              = ydata.astype(float).flatten()
    fit_param_init_all = fit_param_init.copy()

    if use_gpu and not HAS_CUPY:
        raise ImportError("use_gpu=True requires CuPy and cupyx.scipy.ndimage to be installed.")

    ydata_gpu = cp.asarray(ydata) if use_gpu else None

    p0 = fit_param_init_all[fixed == 0]
    lb = lowerbound[fixed == 0]
    ub = upperbound[fixed == 0]

    def residuals_fn(params):
        x = fit_param_init_all.copy()
        x[fixed == 0] = params
        if use_gpu:
            model_gpu = calc_gauss3D_PDFlat_GPU(x, box_coordinates_gpu if box_coordinates_gpu is not None else xdata)
            return cp.asnumpy(model_gpu - ydata_gpu)

        return calc_gauss3D_PDFlat(x, xdata) - ydata

    result   = least_squares(residuals_fn, p0, bounds=(lb, ub),
                             method='trf', ftol=1e-12, xtol=1e-12, max_nfev=5000)

    residual = result.fun
    resnorm  = np.sum(residual**2)

    fit_param_init_all[fixed == 0] = result.x
    return fit_param_init_all, resnorm, residual


def get_minimum_distance(x, y, z, atom_pos): # atom pos must have coords on columns - if its on rows, adjust
    # M. Bartels, UCLA, 2014
    dx = x - atom_pos[0,:]
    dy = y - atom_pos[1,:]
    dz = z - atom_pos[2,:]
    # minimal distance
    DD = np.sqrt(dx**2 + dy**2 + dz**2)
    DD[DD==0] = 100
    minD = np.min(DD)
    atom_number = np.where(DD == minD)  #argwhere, where or non zero give similar but slightly different results 
    # no = row, atom_number = col of each value's output. each is an array. may need to convert atom_number to list output
    pos_eigh = atom_pos[:,atom_number].flatten() # flatten ensures 1D array as columns 

    return minD, atom_number, pos_eigh


def MatrixQuaternionRot(vector, theta):
    theta = theta * np.pi / 180
    vector = vector / np.sqrt(np.dot(vector, vector))
    w = np.cos(theta / 2)
    x = -np.sin(theta / 2) * vector[0]
    y = -np.sin(theta / 2) * vector[1]
    z = -np.sin(theta / 2) * vector[2]

    RotM = np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
        [2 * x * y - 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z + 2 * w * x],
        [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x**2 - 2 * y**2]
    ])

    return RotM

def calc_gauss3D_PDFlat(x, box_coordinates):
    L, M, N = box_coordinates['x'].shape # check box coords format - as a dict are x y and z - checked with create box, all good.
    Num = L * M * N
    v = np.array([
        np.reshape(box_coordinates['x'] - x[2], Num),
        np.reshape(box_coordinates['y'] - x[3], Num),
        np.reshape(box_coordinates['z'] - x[4], Num)
    ])

    vector1 = np.array([0, 0, 1])
    rotmat1 = MatrixQuaternionRot(vector1, x[8])

    vector2 = np.array([0, 1, 0])
    rotmat2 = MatrixQuaternionRot(vector2, x[9])

    vector3 = np.array([0, 0, 1])
    rotmat3 = MatrixQuaternionRot(vector3, x[10])

    rotMAT = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.array([
        [x[5], 0, 0],
        [0, x[6], 0],
        [0, 0, x[7]]
    ])

    A = np.dot(np.dot(rotMAT.T, D), rotMAT)
    #y = x[1] * np.reshape(np.exp(-np.einsum('ijk,ki->ji', v, np.einsum('ijk,ki->ji', A, v))), (L, M, N)) + x[0] #einsum = einstein summation, preferable for larger but currently failing out
    y = x[1] * np.reshape(np.exp(-np.sum(v * np.dot(A, v), axis=0)), (L, M, N)) + x[0] # removed transpose as numpy should handle? and as error is raised from unequivalent dimensions
    y=y.flatten()

    return y

def calc_gauss3D_PD(x, box_coordinates):
    L, M, N = box_coordinates['x'].shape # check box coords format - as a dict are x y and z - checked with create box, all good.
    Num = L * M * N
    v = np.array([
        np.reshape(box_coordinates['x'] - x[2], Num),
        np.reshape(box_coordinates['y'] - x[3], Num),
        np.reshape(box_coordinates['z'] - x[4], Num)
    ])

    vector1 = np.array([0, 0, 1])
    rotmat1 = MatrixQuaternionRot(vector1, x[8])

    vector2 = np.array([0, 1, 0])
    rotmat2 = MatrixQuaternionRot(vector2, x[9])

    vector3 = np.array([0, 0, 1])
    rotmat3 = MatrixQuaternionRot(vector3, x[10])

    rotMAT = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.array([
        [x[5], 0, 0],
        [0, x[6], 0],
        [0, 0, x[7]]
    ])

    A = np.dot(np.dot(rotMAT.T, D), rotMAT)
    y= x[1] * np.reshape(np.exp(-np.sum(v * np.dot(A, v), axis=0)), (L, M, N)) + x[0] 
    return y

def box_to_gpu(box_coordinates):
    if not HAS_CUPY:
        raise ImportError("box_to_gpu requires CuPy to be installed.")

    return {
        'x': cp.asarray(box_coordinates['x']),
        'y': cp.asarray(box_coordinates['y']),
        'z': cp.asarray(box_coordinates['z']),}

def calc_gauss3D_PDFlat_GPU(x, box_coordinates_gpu):
    """
    GPU variant of calc_gauss3D_PDFlat
    """    
    L, M, N = box_coordinates_gpu['x'].shape
    Num = L * M * N

    v_gpu = cp.array([
        cp.reshape(box_coordinates_gpu['x'] - x[2], Num),
        cp.reshape(box_coordinates_gpu['y'] - x[3], Num),
        cp.reshape(box_coordinates_gpu['z'] - x[4], Num)
    ])

    rotmat1 = MatrixQuaternionRot(np.array([0, 0, 1]), x[8])
    rotmat2 = MatrixQuaternionRot(np.array([0, 1, 0]), x[9])
    rotmat3 = MatrixQuaternionRot(np.array([0, 0, 1]), x[10])
    rotMAT  = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.array([[x[5], 0, 0],
                  [0, x[6], 0],
                  [0, 0, x[7]]])
    
    A_np = np.dot(np.dot(rotMAT.T, D), rotMAT)

    A_gpu = cp.asarray(A_np)

    y = x[1] * cp.reshape(
            cp.exp(-cp.sum(v_gpu * cp.dot(A_gpu, v_gpu), axis=0)),
            (L, M, N)
        ) + x[0]

    return y.flatten()


def calc_gauss3D_PD_GPU(x, box_coordinates_gpu):
    """GPU variant of calc_gauss3D_PD """
    L, M, N = box_coordinates_gpu['x'].shape
    Num = L * M * N

    v_gpu = cp.array([
        cp.reshape(box_coordinates_gpu['x'] - x[2], Num),
        cp.reshape(box_coordinates_gpu['y'] - x[3], Num),
        cp.reshape(box_coordinates_gpu['z'] - x[4], Num)
    ])

    rotmat1 = MatrixQuaternionRot(np.array([0, 0, 1]), x[8])
    rotmat2 = MatrixQuaternionRot(np.array([0, 1, 0]), x[9])
    rotmat3 = MatrixQuaternionRot(np.array([0, 0, 1]), x[10])
    rotMAT  = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.array([[x[5], 0, 0],
                  [0, x[6], 0],
                  [0, 0, x[7]]])
    A_gpu = cp.asarray(np.dot(np.dot(rotMAT.T, D), rotMAT))

    y = x[1] * cp.reshape(
            cp.exp(-cp.sum(v_gpu * cp.dot(A_gpu, v_gpu), axis=0)),
            (L, M, N)
        ) + x[0]

    return cp.asnumpy(y)

def stats_cal_PD(data,common_atoms2,common_atoms1,BoxSize1, use_gpu=False):
    AA = (common_atoms1+common_atoms2)/2
    AB = np.vstack(AA) # using vstack and not np.array gives 3,1, array gives 3,
    box1coordinates, BoxCenter1, BoxRadius1, sphere1 = create_box(BoxSize1)
    box1coordinates_gpu = box_to_gpu(box1coordinates) if use_gpu else None

    for num in range(len(AB[1])):
        data_box1,datasphere1 = get_atom_box(data, AB[:, num], BoxSize1, use_gpu=use_gpu)

        ymax_init = data_box1[BoxCenter1-1, BoxCenter1-1, BoxCenter1-1]# -1 for python indexing
        ymax_init = max(0, ymax_init)
        fit_param_init = np.array([0, ymax_init, 0, 0, 0, 0.2, 0.2, 0.8, 0.0, 0.0, 0.0])
        fixed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        lb = np.array([0, 0, -3, -3, -3, 0, 0, 0, -np.pi, 0, -np.pi])
        ub = np.array([np.inf, np.inf, 3, 3, 3, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi])

        fit_result, resnorm, residual = fit_gauss3D_PD(
            fit_param_init,
            box1coordinates,
            data_box1,
            fixed,
            lb,
            ub,
            use_gpu=use_gpu,
            box_coordinates_gpu=box1coordinates_gpu,
        )
        
    return fit_result

def _compute_statsF_single(DataMatrix, pos, BoxSize1, use_gpu=False):
    """Worker function for parallel statsF computation."""
    try:
        fit_result = stats_cal_PD(DataMatrix, pos, pos, BoxSize1, use_gpu=use_gpu)
        return {
            'fit':         fit_result,
            'peak_bg':     fit_result[0],
            'peak_height': fit_result[1],
            'peak_FWHM':   None,   # filled in caller using statsI FWHM
            'localmax_pos': None,  # filled in caller
            'error':       False
        }
    except RuntimeError:
        return {
            'fit': 0, 'peak_bg': 0, 'peak_height': 0,
            'peak_FWHM': 0, 'localmax_pos': 0, 'error': True
        }

def build_statsF_parallel(DataMatrix, atom_pos, statsI, BoxSize1, n_atoms, use_gpu=False):
    """
    Recompute statsF for all accepted atoms in parallel against the original
    DataMatrix. Called after the main tracing loop completes.
    statsI is passed to recover FWHM and localmax_pos for each atom.
    """
    positions = [atom_pos[:, i] for i in range(n_atoms)]

    if HAS_JOBLIB:
        results = Parallel(n_jobs=-1)(
            delayed(_compute_statsF_single)(DataMatrix, pos, BoxSize1, use_gpu)
            for pos in positions
        )
    else:
        results = [_compute_statsF_single(DataMatrix, pos, BoxSize1, use_gpu) for pos in positions]

    statsF = Stats()
    for i, r in enumerate(results):
        statsF.fit.append(r['fit'])
        statsF.peak_bg.append(r['peak_bg'])
        statsF.peak_height.append(r['peak_height'])
        # FWHM and localmax_pos come from statsI — statsF refit doesn't change them
        statsF.peak_FWHM.append(statsI.peak_FWHM[i])
        statsF.localmax_pos.append(statsI.localmax_pos[i])

    return statsF

def get_atom_box(DataMatrix,AtomPos,BoxSize, use_gpu=False):
    import math
    if use_gpu and not HAS_CUPY:
        raise ImportError("use_gpu=True requires CuPy and cupyx.scipy.ndimage to be installed.")

    BoxRadius = (BoxSize - 1)/2
    
    #box coordinate systems
    boxX,boxY,boxZ = np.meshgrid(
    np.arange(-BoxRadius, BoxRadius + 1),
    np.arange(-BoxRadius, BoxRadius + 1),
    np.arange(-BoxRadius, BoxRadius + 1),
    indexing = 'ij') 
    
    x=AtomPos[0]
    y=AtomPos[1]
    z=AtomPos[2]
    
    #integer indices
    fx=math.floor(x)
    fy=math.floor(y)
    fz=math.floor(z)
    
    #off-center shift of peak
    shiftx = (x-fx)
    shifty = (y-fy)
    shiftz = (z-fz)
    
    if shiftx>0.5:
        fx=fx+1
        shiftx=shiftx-1

    if shifty>0.5:
        fy=fy+1
        shifty=shifty-1

    if shiftz>0.5:
        fz=fz+1
        shiftz=shiftz-1
    
    #calculate spherical mask
    sphere = np.sqrt(boxX**2+boxY**2+boxZ**2)<=BoxSize/2
    
    #start stop index, -1 extra on start for conversion
    DataBox = DataMatrix[int(fx-BoxRadius-2):int(fx+BoxRadius+1), int(fy-BoxRadius-2):int(fy+BoxRadius+1) , int(fz-BoxRadius-2):int(fz+BoxRadius+1)]
    if use_gpu:
        DataBox = FourierShift3D_GPU(DataBox,-shiftx,-shifty,-shiftz)
    else:
        DataBox = FourierShift3D(DataBox,-shiftx,-shifty,-shiftz)
    DataBox = DataBox [1:-1,1:-1,1:-1] # should exclude first and last element
    DataSphere = DataBox*sphere
    
    return DataBox, DataSphere

def FourierShift3D(img,dx,dy,dz):
    nx, ny, nz = img.shape
    X, Y, Z = np.meshgrid(np.arange(-(nx-1)//2, (nx+1)//2),
                          np.arange(-(ny-1)//2, (ny+1)//2),
                          np.arange(-(nz-1)//2, (nz+1)//2), indexing='ij')
    
    # perform inverse 3D Fourier transform on an array
    F = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(img)))
    Pfactor = np.exp(2*np.pi*1j*(dx*X/nx + dy*Y/ny + dz*Z/nz))
    # perform 3D Fourier transform on an array
    img2 = np.real(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(F*Pfactor))))
    
    return img2

def FourierShift3D_GPU(img,dx,dy,dz):
    if not HAS_CUPY:
        raise ImportError("FourierShift3D_GPU requires CuPy to be installed.")

    nx, ny, nz = img.shape
    img_gpu = cp.asarray(img)
    X, Y, Z = cp.meshgrid(np.arange(-(nx-1)//2, (nx+1)//2),
                          np.arange(-(ny-1)//2, (ny+1)//2),
                          np.arange(-(nz-1)//2, (nz+1)//2), indexing='ij')
    F = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(img_gpu)))
    Pfactor = cp.exp(2 * cp.pi * 1j * (dx*X/nx + dy*Y/ny + dz*Z/nz))
    img2 = cp.real(cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(F * Pfactor))))
    img2 = cp.asnumpy(img2)
    return img2

def extract_clean_3d_object(volume, min_intensity, looseness=4):
    """Creates a support mask from basic intesntiy thresholding. """
    
    raw_mask = volume >= min_intensity
    if not np.any(raw_mask):
        return np.zeros_like(raw_mask, dtype=bool)
    labeled_mask, num_features = label(raw_mask)
    
    if num_features > 0:
        component_sizes = sum_labels(raw_mask, labeled_mask, range(1, num_features + 1))
        largest_label = np.argmax(component_sizes) + 1
        object_support_mask = labeled_mask == largest_label
    else:
        object_support_mask = raw_mask

    if looseness > 0:
        final_object_item = binary_dilation(object_support_mask, iterations=looseness)
    else:
        final_object_item = object_support_mask

    return final_object_item


if __name__ == "__main__":
    print("AtomTracing.py is intended to be imported as a module.")


