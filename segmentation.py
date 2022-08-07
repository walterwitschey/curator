# system
import argparse
import os
import sys
import logging
import glob

# required modules
import matplotlib.pyplot as plt
import torch 
import nibabel as nb
import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

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

def segment(cine, model):
    # adjust dimensions
    cine_arr = cine.get_fdata()

    print(cine_arr.shape)

    roi_size = (256, 256)
    sw_batch_size = 1

    # get slices
    slices = []
    for phase in range(cine_arr.shape[3]):
        slice = torch.from_numpy(cine_arr[:,:,0,phase]).float().unsqueeze(0).unsqueeze(0)
        output = torch.argmax(sliding_window_inference(slice, roi_size, sw_batch_size, model), dim=1).detach()
        slices.append(output.numpy())
    
    slices = np.transpose(np.asarray(slices)).swapaxes(0,1)

    # make a copy of original nifti with the segmented ventricle
    copynifti = nb.Nifti1Image(slices, cine.affine, cine.header)
    # assert copynifti.header == cine.header
    print(copynifti.shape)
    # assert copynifti.shape == cine.shape
    return copynifti

if __name__ == "__main__":

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segmentation - Create lables of segmented ventricles in cardiac MRI")
    parser.add_argument("--input_dir", type=str, help="Input directory with nifti images to be curated")
    parser.add_argument("--log_file",type=str,help="txt file with log info",default="segmentation_log.txt")

    args = parser.parse_args()
    
    # Initialize logger
    logger=initLogger("segmentation",args.log_file)

    # check for empty input directory
    if args.input_dir is None:
            logger.error("    Check that --input_dir is defined")
            sys.exit()

    # create a segmentation folder
    output_path = os.path.join(args.input_dir,'segmentations')
    if not(os.path.exists(output_path)):
        os.mkdir(os.path.join(output_path))

    # find all cine images
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
    for cine, file_name in cines:
        segmented = segment(cine, model)
        nb.save(segmented, os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0] + '_segmented.nii'))