# Python Script used to implement tk_r_em denoiser
import hyperspy.api as hs
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tk_r_em import load_network

# check if GPU is available
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

def intgrad2d(gradient: np.ndarray, sampling: tuple[float, float] = None):
    """
    Perform Fourier-space integration of gradient. Taken from the beta version of abTEM

    Parameters:
    gradient : two np.ndarrays
        The x- and y-components of the gradient.
    sampling : two float
        Lateral sampling of the gradients. Default is 1.0.

    Returns:
    np.ndarray
        Integrated center of mass measurement
    """
    gx, gy = gradient
    (nx, ny) = gx.shape
    ikx = np.fft.fftfreq(nx, d=sampling[0])
    iky = np.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = np.meshgrid(ikx, iky, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k == 0] = 1e-12
    That = (np.fft.fft2(gx) * grid_ikx + np.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = np.real(np.fft.ifft2(That))
    T -= T.min()
    return T

def DPC_to_iDPC(DPC1, DPC2, DPC3, DPC4):
    """
    Perform transformation from DPC to iDPC. Adapated for Hyperspy.
    Parameters:
        DPC1 , DPC2, DPC3, DPC4:
            Input of DPC segments is dependent on your detector rotation.
            For Cardiff TF Spectra:
            DFS(0,4)=DPC1
            DFS(1,5)=DPC2
            DFS(3,7)=DPC3
            DFS(5,8)=DPC4

    Returns:
        np.ndarray
            Integrated DPC measurement
    """
    signal_01 = np.mean(np.array([ DPC1, DPC2 ]), axis=0 )
    signal_23 = np.mean(np.array([ DPC3, DPC4 ]), axis=0 )
    signal_03 = np.mean(np.array([ DPC1, DPC4 ]), axis=0 )
    signal_21 = np.mean(np.array([ DPC3, DPC2 ]), axis=0 )
    differential_signal_y = signal_01 - signal_23
    differential_signal_x = signal_03 - signal_21
    grad= differential_signal_y,differential_signal_x

    idpc = intgrad2d(grad,[1,1])
    return idpc

def main(): 
    savepath = r'/PATH/TO/SAVE'
    datapath = r'/PATH/TO/DATA'
    
    EMDdata = hs.load(datapath + r'/*.emd')
    
    haadf_list = []
    
    for i in range(len(EMDdata)):
        for k in range(len(EMDdata[i])):
            if EMDdata[i][k].metadata['General'].title == 'HAADF':
                haadfdata = EMDdata[i][k]
                haadf_list.append(haadfdata)
        #print(i) #if needed for trouble shooting
    
    idpc_list = []
    
    for i in range(len(EMDdata)):
        for k in range(len(EMDdata[i])):
            if EMDdata[i][k].metadata['General'].title == 'iDPC':
                idpcdata = EMDdata[i][k]
                idpc_list.append(idpcdata)
        #print(i) #if needed for trouble shooting
    
    # select one of the available networks from [sfr_hrsem, sfr_lrsem, sfr_hrstem, sfr_lrstem, sfr_hrtem, sfr_lrtem]
    net_name = 'sfr_hrstem'
    
    # load its corresponding model
    r_em_nn = load_network(net_name)
    #r_em_nn.summary()
    
    haadf_filt = []
    # run inference
    for haadf in haadf_list:
        haadf_denoise = r_em_nn.predict_patch_based(haadf.data, patch_size=256, stride=128, batch_size=16)
        haadf_denoiseHS = hs.signals.Signal2D(haadf_denoise, metadata = haadf.metadata.as_dictionary())
        haadf_filt.append(haadf_denoiseHS)
    
    haadf_saveStack = hs.stack(haadf_filt)
    haadf_saveStack.save(savepath + r'/HAADFdenoiseCeO2.hspy')
    
    idpc_filt = []
    # run inference
    for idpc in idpc_list:
        idpc_denoise = r_em_nn.predict_patch_based(idpc.data, patch_size=256, stride=128, batch_size=16)
        idpc_denoiseHS = hs.signals.Signal2D(idpc_denoise, metadata = idpc.metadata.as_dictionary())
        idpc_filt.append(idpc_denoiseHS)
    
    idpc_saveStack = hs.stack(idpc_filt)
    idpc_saveStack.save(savepath + r'idpcdenoiseCeO2.hspy')
    



