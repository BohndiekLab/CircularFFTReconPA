# Digital twins enable full-reference quality assessment of photoacoustic image reconstructions

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
3. ENI-G, a Joint Initiative of the University Medical Center Göttingen and the Max Planck Institute for Multidisciplinary Sciences, Göttingen, Germany
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
experimental data, the authors create a reference object that is close to a “ground truth” and corresponding 
simulated sensor data that mimic the real experiment, thus reducing the *simulation-experiment gap*.

Using this digital twin, the paper quantitatively compares multiple state-of-the-art image reconstruction algorithms 
for photoacoustic imaging. Among these is a fast Fourier transform-based reconstruction algorithm for circular 
detection geometries, which is **tested on experimental data for the first time**. The results demonstrate that 
the digital phantom twin approach enables rigorous, full-reference IQA of reconstructions: for example, the Fourier 
algorithm achieved image quality comparable to iterative time reversal but at significantly lower computational cost. 
This highlights the utility of the digital twin framework for assessing reconstruction accuracy and the fidelity of 
the forward model, facilitating fair comparisons of different algorithms.

## Code Overview
The code contains Python scripts and modules implementing the image reconstruction algorithms and supporting 
utilities for data processing and analysis. The repository includes multiple reconstruction methods:
**Delay and Sum**, **Filtered back-projection**, **model-based reconstruction (MB)**, and a **fast Fourier 
transform-based (FFT) reconstruction** algorithm for circular sensor geometries. Each method has a dedicated 
script or function in `code/recon_algorithms` for performing the reconstruction on the input data.

- **Delay and Sum**: `code/recon_algorithms/backprojection.py`
- **Filtered back-projection**: `code/recon_algorithms/backprojection_hilbert.py`
- **Model-based reconstruction (MB)**: `code/recon_algorithms/modelbased.py`
- **Fast Fourier transform-based (FFT) reconstruction**: `code/recon_algorithms/fft/fast_inverse_CRUK.py`

> [!CAUTION]
> Please note that the **time reversal (TR)** and **iterative time reversal (ITTR)** algorithms are not included in this repository, but the reconstruction results are included in the data on Zenodo to allow reproducing the results.

Configuration of paths and parameters is centralised in `utils/constants.py`. **Before running any reconstruction, open this file and adjust the file paths** to point to your local `data/` directory. This ensures the code knows where to find the input files and where to save outputs.

Typical usage workflow:
1. **Run reconstruction scripts:** Execute the scripts for each reconstruction algorithm to generate reconstructed images by running the files listed above. For example, running the FFT reconstruction script will read the time-series data from `data/.../raw/` and produce reconstructed images in `data/.../recons/fft/`. Similarly, run the BP, TR, ITTR, MB, and BPH reconstruction scripts to populate their respective subfolders. Each script uses the experimental (`exp`) or simulated (`sim`/`sim_raw`) data as input and saves the reconstructed image files (e.g., as NumPy arrays or images) in the corresponding algorithm’s folder under `recons`.
2. **Compute IQA metrics:** After all reconstructions are obtained, use the provided `compute_measures.py` script to calculate full-reference quality metrics. This step will compare each reconstructed image against the ground truth image from `p0/` (the “reference” provided by the digital twin) and compute metrics such as PSNR, SSIM, and other figures of merit. The metrics can be computed for both the simulated data (where ground truth is known) and, by extension, help evaluate how well the calibration holds for experimental reconstructions. The results of this step might be saved as a table or CSV, or printed to the console, depending on the implementation. By changing the `data_source` parameter from (`exp`) to (`sim`/`sim_raw`), it can be controlled whether the measures should be computed on the experimental or simulated data sets.
3. **Visualisation:** Finally, run the visualisation or plotting scripts to generate figures that compare the different reconstruction methods. These scripts can create side-by-side image comparisons, difference images, or plots of the IQA metrics for each algorithm. This helps reproduce the figures and quantitative comparisons presented in the paper. For instance, you might generate a figure showing all reconstructions for a given phantom, or a bar chart of the SSIM/PSNR values of all methods on the testing set. There are dedicated scripts for each figure of the manuscript:

- `compare_reconstructions.py` - **Figure 4** in the manuscript
- `full_view_vs_limited_view.py` - **Figure 5 Step A**: First run this for model-based, iterative time reversal and FFT-based reconstruction.
- `full_view_vs_limited_view_overview.py` - **Figure 5 Step B**: Afterwards, this script reproduces Figure 5 in the manuscript.
- `show_forward_model_accuracy.py` - **Figure 3** in the manuscript

## Citation
If you use this code or data in your research, please cite the corresponding paper:

Janek Gröhl *et al.*, **“Digital twins enable full-reference quality assessment of photoacoustic image reconstructions,”** arXiv preprint arXiv:2505.24514 (2025).

## Data
All data are publicly available on Zenodo: https://doi.org/10.5281/zenodo.15388429

## License
This project is released under the **MIT License** (see the `LICENSE` file for details).

## Acknowledgements

**Funding Acknowledgements:** Development of this code and the underlying research were supported by multiple grants:
- **Deutsche Forschungsgemeinschaft (DFG)** – projects GR 5824/1 and GR 5824/2 (supporting J.G.).  
- **U.S. National Science Foundation (NSF)** – award DMS-2405348 (supporting L.K.).  
- **Cancer Research UK (CRUK)** – A29580 (supporting T.R.E.).  
- **Research Council of Finland** – Flagship programme projects 359186 and 358944, Centre of Excellence projects 353093 and 353086, and Academy Research Fellow project 338408 (supporting A.H. and J.P.).  
- **European Research Council (ERC)** – European Union Horizon 2020 programme, grant 101001417 (*QUANTOM* project, supporting J.P.).  
- **Finnish Ministry of Education and Culture** – support for a doctoral programme pilot *“Mathematics of Sensing, Imaging and Modelling”* (supporting J.P.).  
- **Engineering and Physical Sciences Research Council (EPSRC), UK** – grants EP/W029324/1 and EP/T014369/1 (supporting B.T.C.), as well as EP/R014604/1, which provided support for the Isaac Newton Institute’s **“Rich and Nonlinear Tomography”** programme.
- **Physics of Life Network Phase 3 (PoLNET3)** – summer 2023 student bursary *“Calibrating numerical photoacoustic forward models with experimental measurements”* (supporting F.D.C.).
