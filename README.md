# Curator - Bespoke Medical Image Curation

## Setup

Basic overview of the installation steps.

We have the choice of using Anaconda in native Win 11 or in Ubuntu 20.04 if using the Windows Subsystem for Linux

(Optional) Install [WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Install [Anaconda](https://www.anaconda.com/) (Download for Linux on WSL2 or Windows for Win11)

Create a new conda environment
```
conda create --name curator python=3.8.12
```

Activate the new environment
```
conda activate curator
```

This has been tested on Python 3.8.12

Make sure the following package versions are installed (via pip):
```
dicom2nifti>=2.3.0
pydicom>=2.2.2
tqdm>=4.62.3
pandas>=1.3.4
tensorflow>=2.9.1
```

## Additional CUDA setup notes

```
```

## Usage

```
python curator.py [-h] mode

python curator.py parse --input_dir /dcmdirectory/ --output_dir /outdcmdirectory/ --csv_file csv_file.csv

python curator.py nifti --csv_file csv_file.csv --output_dir /outdcmdirectory/

python curator.py classify --input_dir /niidirectory/

python segmentation.py --input_dir /niidirectory/

```

## Arguments

* `mode`: curator mode: ['parse','nifti','train','inference']
* `--input_dir`: folder containing dicom images to be curated (will search all subdirectories of this folder).
* `--output_dir`: folder to place all the sorted dicom images by accession and series
* `--csv_file`: csv file containing a summary of all studies and series. Each series is a row.
* `--log_file`: Log file where results will be stored
* `--use_patientname_as_foldername`: (Parse) optional feature to use dicom field PatientName as foldername (default is AccessionNumber).
* `--use_cmr_info_as_filename`: (Parse) optional feature to place dicom fields SlicePosition_TriggerTime in dicom filename

# Additional usage details 

* mode = "parse"
* Read a folder containing unlabeled dicom images. sort and copy the files, and generate a csv file summarizing all the valid series for labeling.
* Mandatory arguments: --input_dir, --output_dir, --csv_file

* mode = "nifti"
* Read a csv file containing columns "dcmdir" and "label". Images in dcmdir will be converted to nifti, one nifti per row.
* Hint: Run parse to generate a csv file with dcmdir, add labels (contrast, 
* Mandatory arguments: --csv_file, --output_dir

* mode = "train"
* Train a NN to label imaging data
* Mandatory arguments: --input_dir, --csv_file

* mode = "classify"
* Given a trained NN, make new inferences on unseen data
* Mandartory arguments: --input_dir