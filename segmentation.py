# system
import argparse
import os
import sys
import logging
import glob
import csv

# required modules
import matplotlib.pyplot as plt
import torch 
import nibabel as nb
import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import pydicom

#debugging
from PIL import Image

def initLogger(name,logfile):
    # Create debugging information (logger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(logfile)
    streamHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger

def segment_nii(cine, model):
    # adjust dimensions
    cine_arr = cine.get_fdata()

    roi_size = (256, 256)
    sw_batch_size = 1

    # get slices
    slices = []
    for phase in range(cine_arr.shape[3]):
        slice = torch.from_numpy(cine_arr[:,:,0,phase]).float().unsqueeze(0).unsqueeze(0)
        output = torch.argmax(sliding_window_inference(slice, roi_size, sw_batch_size, model), dim=1).detach()
        slices.append(output.numpy())

    slices = np.transpose(np.asarray(slices)).swapaxes(0,1)
    print(slices.shape)

    # make a copy of original nifti with the segmented ventricle
    copynifti = nb.Nifti1Image(slices, cine.affine, cine.header)
    # assert copynifti.header == cine.header
    # print(copynifti.shape)
    # assert copynifti.shape == cine.shape
    return copynifti

def segment_dicom(cine, model):

    # convert dicom to torch tensor
    slice = np.rot90(cine.pixel_array.astype('float64'), 3)
    slice = torch.from_numpy(slice.copy()).float().unsqueeze(0).unsqueeze(0)

    roi_size = (256, 256)
    sw_batch_size = 1
    result = sliding_window_inference(slice, roi_size, sw_batch_size, model)
    output = torch.argmax(result, dim=1).detach()
    return np.transpose(output[0,:,:].numpy()) * 100

if __name__ == "__main__":

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segmentation - Create lables of segmented ventricles in cardiac MRI")
    parser.add_argument("--input_dir", type=str, help="Input directory with nifti images to be curated")
    parser.add_argument("--log_file",type=str,help="txt file with log info",default="segmentation_log.txt")
    parser.add_argument("--use_dicom",action='store_true',help="use dicom instead of nifti in classifier mode")
    parser.set_defaults(use_dicom=False)
    args = parser.parse_args()
    
    # Initialize logger
    logger=initLogger("segmentation",args.log_file)

    # check for empty input directory
    if args.input_dir is None:
            logger.error("    Check that --input_dir is defined")
            sys.exit()

    # create a segmentation folder
    foldername = '{}_segmentations'.format(os.path.basename(args.input_dir))
    output_path = os.path.join(os.path.dirname(args.input_dir), foldername)
    if not(os.path.exists(output_path)):
        os.mkdir(os.path.join(output_path))

    # find all cine images
    if (args.use_dicom == True):
        cines = []
        foldernames = []
        skipped = 0

        # check all series that are four dimensional and have more than 20 phases
        for folder in os.listdir(args.input_dir):
            folderpath = os.path.join(args.input_dir, folder)
            if glob.glob(os.path.join(folderpath,'*.dcm')):
                onlyfiles = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.dcm')]
                fourdimension = all([pydicom.read_file(f).SeriesInstanceUID == pydicom.read_file(onlyfiles[0]).SeriesInstanceUID for f in onlyfiles])
                
                if fourdimension and len(onlyfiles) > 20:
                    cines.append(folderpath)
                    foldernames.append(folder)
                else:
                    skipped += 1
        logger.info('       Skipped {0} images, Segmenting {1} cines'.format(skipped, len(cines)))
    else:
        images = glob.glob(os.path.join(args.input_dir,'*.nii')) + glob.glob(os.path.join(args.input_dir,'*.nii.gz'))
        cines = []
        skipped = 0
        for nifti_path in images:
            nifti = nb.load(nifti_path)
            if len(nifti.shape) != 4 or nifti.shape[3] < 20:
                skipped += 1
            else:
                cines.append((nifti, nifti_path))
        logger.info('       Skipped {0} images, Segmenting {1} cines'.format(skipped, len(cines)))

    # load monai UNet model
    device = torch.device("cpu")
    model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    )
    model.load_state_dict(torch.load('best_metric_model.pth', map_location=torch.device('cpu')))

    # generate predictions for each cine
    if (args.use_dicom == True):
        for (cinefolder, nameonly) in zip(cines, foldernames):
            # create a directory for each series
            savepath = os.path.join(output_path, nameonly)
            if not(os.path.exists(savepath)):
                os.mkdir(savepath)

            # write a csv file
            with open(os.path.join(savepath, 'segmentations.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['file path', 'trigger time', 'slice location', 'RV blood pool', 'LV myocardium', 'LV blood pool', 'voxel size'])

                # segment each dicom slice
                for dcm in os.listdir(cinefolder):
                    dicom = pydicom.dcmread(os.path.join(cinefolder, dcm))
                    segmented = segment_dicom(dicom, model)

                    assert segmented.shape == dicom.pixel_array.shape
                    dicom.PixelData = segmented.astype(np.int16).tobytes()
                    dicom.SeriesNumber = dicom.SeriesNumber + 3000
                    dicom.save_as(os.path.join(savepath, dcm))

                    # write entries into csv file
                    row = [os.path.join(cinefolder, dcm), dicom.TriggerTime, dicom.SliceLocation]
                    voxelsize = dicom.PixelSpacing[0] * dicom.PixelSpacing[1] * max(dicom.SliceThickness, dicom.SpacingBetweenSlices)

                    # get counts of each pixel
                    counts = np.unique(segmented.astype(np.int16), return_counts=True)[1]
                    row.extend([counts[3] * voxelsize, counts[2] * voxelsize, counts[1] * voxelsize, voxelsize])
                    writer.writerow(row)

    else:
        for cine, file_name in cines:
            segmented = segment_nii(cine, model)
            nb.save(segmented, os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0] + '_segmented.nii'))