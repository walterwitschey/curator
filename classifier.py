from asyncio.windows_events import NULL
from PIL import Image as im
import os
import csv
import logging
import glob
import re
from tqdm import tqdm
import shutil
logger=logging.getLogger("curator")

# load dependencies
import pandas as pd
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import pydicom

# packages for neural network
import tensorflow as tf

contrast_names = ['CINE', 'T1', 'T1_star', 'T2', 'T2_star', 'T1RHO', 'TRUFI', 'LGE', 'PERF_AIF', 'PSIR', 'STIR', 'MAG',
 'SCOUT', 'TSE', 'DE', 'PSIR_MAG', 'HASTE', 'TWIST', 'FL3D', 'PSIR_PHASE', 'PC_MAG', 'PC_PHASE', 'GRE', 'T1_ERROR', 'T1_MAP', 'T1_IR', 'T1RHO_MAP', 'T1RHO_ERROR']
orientation_names = ['2CH', '3CH', '4CH', 'SAX', 'LAX', 'APEX', 'MID', 'BASE', 'LVOT', 'AOFLOW', 'AV', 'LOC',
 'CANDYCANE', 'PAFLOW', 'VLA', 'WHOLE HEART', 'AO_AX', 'AXIAL', 'MPA', 'RPA', 'LPA', 'TRICUSPID', 'SAGITTAL',
  'CORONAL', 'AXIAL_MIP', 'SAGITTAL_MIP', 'CORONAL_MIP', 'RVOT', 'SVC', 'IVC', 'DAO', 'LPV', 'RPV', 'MITRAL', 'RV2CH', 'LV3CH', 'SAG_RVOT']

def int_code(x):
    return int(re.match('[0-9]+', os.path.basename(x)).group(0))

def single_subject(input_dir):
    folders = os.listdir(input_dir)
    checkpath = os.path.join(input_dir, folders[0])
    single = False

    for i in os.listdir(checkpath):
        if i.endswith('.dcm'):
            single = True
    return single

def classifyNifti(input_dir, output_dir, use_dicom=False):
    logger.info("classifier.classifyNifti")
    
    # load nifti or dicom images
    if (use_dicom==True):
        logger.info('       using dicoms')
        # first, check for file hierarchy
        onesubject = single_subject(input_dir)
        if onesubject:
            logger.info('       working with only one subject')
            classifySubject(input_dir, output_dir, use_dicom)

        else:
            logger.info('       working with multiple subjects')

            # check if output path already exists
            outpath = os.path.join(output_dir, 'dcm')
            if not(os.path.isdir(outpath)):
                os.mkdir(outpath)

            # run classifier for each subject
            subjects = os.listdir(input_dir)
            for subject in subjects:

                singleinput = os.path.join(input_dir, subject)
                print(singleinput)
                classifySubject(singleinput, outpath, use_dicom)
    else:
        # use niftis
        classifySubject(input_dir, output_dir, use_dicom)


def classifySubject(input_dir, output_dir, use_dicom):
    
    if (use_dicom == True):
        img_array, phase_counts, fetched_directory, errors, seriesnames = load_dicoms(input_dir)
        newfolder = 'Classified_{}'.format(os.path.basename(input_dir))
        csvpath = os.path.join(output_dir, newfolder)
    else:
        img_array, phase_counts, fetched_directory, errors = load_niftis(input_dir)

    # notify the user if some images were omitted from image array
    if errors > 0:
        logger.info('       failed loading {} images.'.format(errors))

    # generate predictions
    logger.info('       generating predictions for %d images', img_array.shape[0])
    orientation, contrast, cine_probability = generate_predictions(img_array)

    # === debugging purposes only ===
    #save_images(img_array, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images'), orientation, contrast)

    # iterate through directory go generate separate csv file for each study
    if not(use_dicom):
        parsedirectory, chunk = ([] for i in range(2))
        initial = fetched_directory[0][1]
        for num in range(len(fetched_directory)):
            if initial == fetched_directory[num][1]:
                chunk.append(fetched_directory[num] + (orientation_names[orientation[num]], contrast_names[contrast[num]], cine_probability[num]))
            else:
                parsedirectory.append(chunk)
                chunk = []
                chunk.append(fetched_directory[num] + (orientation_names[orientation[num]], contrast_names[contrast[num]], cine_probability[num]))
                initial = fetched_directory[num][1]
        parsedirectory.append(chunk)
    
    # sort dicoms into each category: supports dicom only
    if (use_dicom == True):
        generate_sorted_folder(input_dir, output_dir, orientation, contrast, seriesnames)

    # write predictions to csv
    if (use_dicom == True):

        with open(os.path.join(csvpath,'predictions.csv'),"w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File Name', 'Orientation', 'Contrast', 'Cine_Probability'])

            for i, name in enumerate(seriesnames):
                writer.writerow([name, orientation_names[orientation[i]], contrast_names[contrast[i]], cine_probability[i]])

        # a separate collection only for sax cine images
        with open(os.path.join(csvpath,'sax_cine_list.csv'),"w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File Name', 'Orientation', 'Contrast', 'Cine_Probability'])

            # add images only if probability is high, has 30 phases, and predicted as SAX CINE
            for i, name in enumerate(seriesnames):
                if cine_probability[i] >= 0.9 and phase_counts[i] >= 20 and orientation_names[orientation[i]] == 'SAX' and contrast_names[contrast[i]] == 'Cine':
                    writer.writerow([name, orientation_names[orientation[i]], contrast_names[contrast[i]], cine_probability[i]])

    else:
        for study in parsedirectory:
            with open(os.path.join(study[0][1],'predictions.csv'),"w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File Name', 'Orientation', 'Contrast', 'Cine_Probability'])

                for image in study:
                    filename = image[0][len(image[1]) + 1:]
                    writer.writerow([filename,image[2],image[3], image[4]])

            # a separate collection only for sax cine images
            with open(os.path.join(study[0][1],'sax_cine_list.csv'),"w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File Name', 'Orientation', 'Contrast', 'Cine_Probability'])

                # add images only if probability is high, has 30 phases, and predicted as SAX CINE
                for i, image in enumerate(study):
                    if cine_probability[i] >= 0.9 and phase_counts[i] >= 20 and image[2] == 'SAX' and image[3] == 'Cine':
                        filename = image[0][len(image[1]) + 1:]
                        writer.writerow([filename, image[2], image[3], image[4]])


def load_niftis(input_dir):
    search = sorted(glob.glob(os.path.join(input_dir,'**/*.nii'),recursive=True))
    results = sorted(search, key = int_code)

    # read image files recursively from image directory
    directory = []
    for file in results:
        directory.append((file, os.path.dirname(file)))

    # check for empty folder edge case
    if len(directory) == 0:
        logger.error("classifier was unable to fetch any images from input directory")
        return

    img_array = []
    phase_counts = []
    fetched_directory = []
    errors = 0
    for imgfile, folderpath in tqdm(directory):
        try:
            img = nb.load(imgfile).get_fdata()
        except Exception as e:
            # print(e)
            logging.warning('      failed to convert nifti to numpy array : %s',imgfile)
            errors += 1
        else:
            if len(img.shape) == 4:
                phase_counts.append(img.shape[3])
            else:
                phase_counts.append(-1)
            img_array.append(preprocess_image(img, imgfile))
            fetched_directory.append((imgfile, folderpath))
    img_array = np.array(img_array)
    fetched_directory.sort(key=lambda y: y[1])
    return img_array, phase_counts, fetched_directory, errors

def load_dicoms(input_dir):
    series = []
    seriesnames = []

    for folder in os.listdir(input_dir):
        folderpath = os.path.join(input_dir, folder)
        if glob.glob(os.path.join(folderpath,'*.dcm')):
            series.append(folderpath)

    img_array = []
    phase_counts = []
    fetched_directory = []
    errors = 0

    # load all images in the same series
    for seriespath in tqdm(series):
        try:
            onlyfiles = [os.path.join(seriespath, f) for f in os.listdir(seriespath) if os.path.isfile(os.path.join(seriespath, f))]
            dicoms = [pydicom.read_file(dcm).pixel_array.astype('float64') for dcm in onlyfiles]
            img = np.stack(dicoms, axis=2)
        except:
            logging.warning('      failed to convert dcm to numpy array : %s',seriespath)
            errors += 1
        else:
            # check if image is 3d or 4d by checking if all images have same SeriesInstanceUID
            fourdimension = all([pydicom.read_file(f).SeriesInstanceUID == pydicom.read_file(onlyfiles[0]).SeriesInstanceUID for f in onlyfiles])

            if fourdimension:
                phase_counts.append(len(onlyfiles))
            else:
                phase_counts.append(-1)
            img_array.append(preprocess_image(img, seriespath))
            seriesnames.append(seriespath[len(input_dir)+1:])
            fetched_directory.append(seriespath)
    img_array = np.array(img_array)

    return img_array, phase_counts, fetched_directory, errors, seriesnames

def preprocess_image(img, path):

    # check the dimension of input image
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    elif len(img.shape) == 4:
        img = img.mean(axis=3)
        img = img[:,:,0]
    else:
        logging.warning('       image is out of dimension: %s',path)
        return
    
    img *= (255.0/(np.amax(img)))
    png = im.fromarray(img).resize((150, 150))
    img = 255.0 - np.array(png)
    return img

def generate_predictions(img_array):
    # preprocess image array
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=3)

    # load models
    orientation_model = tf.keras.models.load_model("orientation_model")
    contrast_model = tf.keras.models.load_model("contrast_model")
    predict_orientation = tf.keras.Sequential([orientation_model, tf.keras.layers.Softmax()])
    predict_contrast = tf.keras.Sequential([contrast_model, tf.keras.layers.Softmax()])

    # generate predictions
    orientations = np.argmax(predict_orientation.predict(img_array), axis=1)
    contrasts = np.argmax(predict_contrast.predict(img_array), axis=1)

    # cache and return probabilities for cine
    cine_probability = [pred[0] for pred in contrast_model.predict(img_array)]
    
    # print results (debugging only)
    #for i in range(orientations.shape[0]):
        #print('orientation: %s', orientation_names[orientations[i]])
        #print('contrast: %s', contrast_names[contrasts[i]])

    # TODO: Add criteria
    # If cine, then keep label SAX
    # If only one frame, then can apply apex, mid, or base
    # If localizer (loc), then Scout and TRUFI can be grouped
    return (orientations, contrasts, cine_probability)

def generate_sorted_folder(input_dir, output_dir, orientation, contrast, seriesnames):
    # generate a sorted folder in the parent directory
    newfolder = 'Classified_{}'.format(os.path.basename(input_dir))
    print(output_dir)
    newpath = os.path.join(output_dir, newfolder)

    if not(os.path.isdir(newpath)):
        os.mkdir(newpath)

    # iterate through each series
    for i, name in enumerate(seriesnames):
        classname = '{}_{}'.format(orientation_names[orientation[i]], contrast_names[contrast[i]])
        
        # if this class name folder does not exist, make a new one
        classfolder = os.path.join(newpath, classname)
        if not(os.path.isdir(classfolder)):
            os.mkdir(classfolder)

        # copy paste all series
        src = os.path.join(input_dir, name)
        dst = os.path.join(classfolder, name)
        shutil.copytree(src, dst)
    

# save mean slice images to local path (debugging purposes only)
def save_images(image_array, file_path, orientation, contrast):
    if not(os.path.isdir(file_path)):
        os.mkdir(file_path)

    for slice in range(image_array.shape[0]):
        plt.figure()
        plt.imshow(image_array[slice], cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.title(orientation_names[orientation[slice]] + ' ' + contrast_names[contrast[slice]])
        plt.savefig(os.path.join(file_path, (str(slice) + '.png')))
