# Dataset for the paper: Digital twins enable full-reference quality assessment of photoacoustic image reconstructions

## Authors
Janek Gröhl<sup>1,2,3</sup>, 
Leonid Kunyansky<sup>4</sup>, 
Jenni Poimala<sup>5</sup>, 
Thomas R. Else<sup>1,2,6</sup>, 
Francesca Di Cecio<sup>1,2</sup>, 
Sarah E. Bohndiek<sup>1,2</sup>, 
Ben T. Cox<sup>7</sup>, and 
Andreas Hauptmann<sup>8,9</sup>

## Affiliations
1. Department of Physics, University of Cambridge, U.K.  
2. Cancer Research UK Cambridge Institute, University of Cambridge, U.K.  
3. ENI-G, a Joint Initiative of the University Medical Center Göttingen and the Max Planck Institute for 
Multidisciplinary Sciences, Göttingen, Germany
4. Department of Mathematics, University of Arizona, USA  
5. Department of Technical Physics, University of Eastern Finland, Finland  
6. Department of Bioengineering, Imperial College London, U.K.  
7. Department of Medical Physics and Biomedical Engineering, University College London, U.K.  
8. Research Unit of Mathematical Sciences, University of Oulu, Finland  
9. Department of Computer Science, University College London, U.K.  

## Summary
**Digital twins for photoacoustic imaging:** This repository accompanies the paper, which introduces a *digital twin* 
framework to enable full-reference image quality assessment (IQA) in photoacoustic tomography. In conventional 
experiments, quantitative comparison of reconstruction algorithms is challenging because no-reference IQA measures 
are often inadequate, and full-reference measures require a known ideal reference image (which is unavailable for 
real tissue or physical phantoms). The paper’s approach overcomes this by using numerical *tissue-mimicking phantoms* 
and a model of the imaging system as a **digital twin** of the experiment. By calibrating the simulations to match 
experimental data, the authors create a reference object that is close to a “ground truth” and corresponding simulated 
sensor data that mimic the real experiment, thus reducing the *simulation-experiment gap*.

Using this digital twin, the paper quantitatively compares multiple state-of-the-art image reconstruction algorithms 
for photoacoustic imaging. Among these is a fast Fourier transform-based reconstruction algorithm for circular detection 
geometries, which is **tested on experimental data for the first time**. The results demonstrate that the digital phantom 
twin approach enables rigorous, full-reference IQA of reconstructions: for example, the Fourier algorithm achieved image 
quality comparable to iterative time reversal but at significantly lower computational cost. This highlights the utility 
of the digital twin framework for assessing reconstruction accuracy and the fidelity of the forward model, facilitating 
fair comparisons of different algorithms.

## Data Structure
```
project_root/

├── calibration/
│   ├── p0/                    # Initial pressure distributions (300x300)
│   ├── raw/
│   │   ├── exp/               # Experimental measurements (256, 2030)
│   │   ├── sim/               # Calibrated simulations (256, 2030)
│   │   └── sim_raw/           # Uncalibrated simulations (256, 2030)
│   └── recons/
│       ├── bp/                # Delay-and-sum reconstruction (300x300)
│       ├── bph/               # Filtered backprojection reconstruction (300x300)
│       ├── fft/               # Fourier transform-based reconstruction (300x300)
│       ├── ittr/              # Iterative time reversal reconstruction (300x300)
│       ├── mb/                # Model-based reconstruction (300x300)
│       └── tr/                # Time reversal reconstruction (300x300)
├── full_view/
│   ├── raw/                   # Simulated time series data, equivalent to "sim_raw" (340, 2030)
│   ├── fft/                   # Fourier transform-based reconstruction (300x300)
│   ├── ittr/                  # Iterative time reversal reconstruction (300x300)
│   ├── mb/                    # Model-based reconstruction (300x300)
│   └── detector_positions.txt # List of all detector coordinates for full-view array (340 elements)
└── testing/
    └── ... (same structure as calibration)
```

### Data Format
- All data are stored as `.npy` NumPy arrays unless otherwise stated.
- Time series data size: `256 x 2030` (detector_elements, time_steps)
- Image sizes (initial pressure and reconstructions): `300 x 300` pixels with resolution `0.10667 mm/pixel` (32 x 32 mm).

## Usage Recommendations

1. **Calibration phase:**
   - Use `calibration/` to tune reconstruction methods or forward models.
   - Simulated and experimental data are paired; use *p₀* as reference.

2. **Testing phase:**
   - Apply trained/calibrated methods to `testing/`.
   - Compare reconstructions using PSNR, SSIM, or other metrics against *p₀*.

You can use reconstructions in `recons/` to reproduce main paper results.

The code repository in https://github.com/BohndiekLab/CircularFFTReconPA contains all codes necessary to access the 
data and to reproduce the findings of the paper from the data.

## Citation
If you use this code or data in your research, please cite the corresponding paper.

Janek Gröhl *et al.*, **“Digital twins enable full-reference quality assessment of photoacoustic image reconstructions”**

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the data, provided appropriate credit is given.


## Funding Acknowledgements

- **Deutsche Forschungsgemeinschaft (DFG)** - projects GR5824/1 and GR5824/2 (supporting J.G.).  
- **U.S. National Science Foundation (NSF)** - award DMS-2405348 (supporting L.K.).  
- **Cancer Research UK (CRUK)** - A29580 (supporting T.R.E.).  
- **Research Council of Finland** - Flagship programme projects 359186 and 358944, Centre of Excellence projects 353093 and 353086, and Academy Research Fellow project 338408 (supporting A.H. and J.P.).  
- **European Research Council (ERC)** - European Union Horizon 2020 programme, grant 101001417 (*QUANTOM* project, supporting J.P.).  
- **Finnish Ministry of Education and Culture** - support for a doctoral programme pilot *“Mathematics of Sensing, Imaging and Modelling”* (supporting J.P.).  
- **Engineering and Physical Sciences Research Council (EPSRC), UK** - grants EP/W029324/1 and EP/T014369/1 (supporting B.T.C.), as well as EP/R014604/1, which provided support for the Isaac Newton Institute’s **“Rich and Nonlinear Tomography”** programme.
- **Physics of Life Network Phase 3 (PoLNET3)** - summer 2023 student bursary *“Calibrating numerical photoacoustic forward models with experimental measurements”* (supporting F.D.C.).
