# Integrated Differential Phase Constrast Atomic Electron Tomography (iDPC-AET)
A code repository to accompany the paper "Imaging Of Subsurface Vacancies In Ceria Using Atomic Electron Tomography".

This repository contains all the code utilised to generate iDPC-AET reconstructions. Scripts are provided, alongside links to relevant published AET methods that have been utilised and/or adapted for this pipeline. 

The code in this repository is organised into 3 categories.

## 2) Simulation of Tilt Series:
Simulations of iDPC images were performed in [abTEM](https://github.com/abTEM/abTEM).   
More information on iDPC simulations are available in the [idpc-simulations repository](https://github.com/EllaKitching/idpc-simulations/tree/main)   

## 2) Tilt Series Preprocessing:
**Denoising** - CNN model tk_r_em, credit: Ivan Lobato et al., [GitHub](https://github.com/Ivanlh20/tk_r_em), [paper](https://www.nature.com/articles/s41524-023-01188-0)   
**Background Removal**  - Averaging and Inpainting methods are both included.   
**Alignment** - Manual alignment and COM-CL Methods used, credit for COM-CL: M. C. Scott et al., [paper](https://www.nature.com/articles/nature10934), [source codes](https://www.physics.ucla.edu/research/imaging/ProjectionAlignment/index.html)

## 3) Reconstruction Algorithms:
**SIRT** - Please note this was only used for reconstructing simulations.   
**Expectation Maxisimation** - completed in [Inspect3d](https://www.thermofisher.com/uk/en/home/electron-microscopy/products/software-em-3d-vis/inspect-3d-software.html)   
**RESIRE** - credit: Minh Pham et al., [paper](https://www.nature.com/articles/s41598-023-31124-7), [zenodo](https://zenodo.org/records/7273314)  

## 4) Reconstruction Volume Postprocessing:
**3D Wiener Filtering** - credit: Chien-Chun Chen et al., [paper](https://doi.org/10.1038/nature12009), [source codes](https://www.physics.ucla.edu/research/imaging/dislocations/)  
**Atom Tracing** - credit: Rui Xu et al., [paper](https://www.nature.com/articles/nmat4426), [source codes](https://www.physics.ucla.edu/research/imaging/3Datoms/index.html)  
**Polyhedral Template Matching**  - credit: Zezhou Li et al., [paper](https://www.nature.com/articles/s41467-023-38536-z), [GitHub](https://github.com/live-long-and-prosper/PdPtCoreShellNanoparticles/tree/main_new)
