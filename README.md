# Curate - Bespoke Medical Image Curation

## Setup

Basic overview of the installation steps.

We have the choice of using Anaconda in native Win 11 or in Ubuntu 20.04 if using the Windows Subsystem for Linux

(Optional) Install WSL2.
[WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Install Anaconda (Download for Linux on WSL2 or Windows for Win11)
[Anaconda](https://www.anaconda.com/)

Create a new conda environment
```
conda create --name curator python=3.9
```

Activate the new environment
```
conda activate curator
```

This has been tested on Python 3.9.7

Make sure the following package versions are installed (via pip):
```
```

Install the following additional packages via apt on Ubuntu 20.04
```
```

## Additional CUDA setup notes

```
```

## Usage

```
python curator.py [-h] input_dir output_dir --csv_file --log_file
```

## Arguments

* `input_dir`: folder containing dicom images to be curated (will search all subdirectories of this folder).
* `output_dir': folder to place all the sorted dicom images by accession and series
* `csv_file`: csv file containing a summary of all studies and series. Each series is a row.
* 'log_file': Log file where results will be stored