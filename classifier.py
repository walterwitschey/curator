from PIL import Image as im
import os
import csv
import logging
import glob
from tqdm import tqdm
logger=logging.getLogger("curator")

# load dependencies
import pandas as pd
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# packages for neural network
import tensorflow as tf

contrast_names = ["Cine","T1","T1*","T2","T2*","T1rho","Trufi","LGE","PERF_AIF","PSIR","STIR","MAG","Scout","TSE", "DE"]
orientation_names = ["2ch", "3ch","4ch","SAX","LAX","apex","mid","base","LVOT","Aoflow","AV","loc","candycane","paflow","VLA", "whole heart", "AO_AX"]

def classifyNifti(input_dir):
    logger.info("classifier.classifyNifti")

    # read image files recursively from image directory
    directory = []
    for file in glob.glob(os.path.join(input_dir,'**/*.nii'),recursive=True):
        directory.append((file, os.path.dirname(file)))

    # check for empty folder edge case
    if len(directory) == 0:
        logger.error("classifier was unable to fetch any images from input directory")
        return

    img_array = []
    for imgfile, folderpath in tqdm(directory):
        try:
            img = nb.load(imgfile).get_fdata()
        except:
            logging.warning('      could not find nifti file with path : %s',imgfile)
        else:
            img_array.append(preprocess_image(img, imgfile))
    img_array = np.array(img_array)

    # generate predictions
    logger.info('      generating predictions for %d images', img_array.shape[0])
    orientation, contrast = generate_predictions(img_array)

    # === debugging purposes only ===
    #save_images(img_array, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images'), orientation, contrast)

    # iterate through directory go generate separate csv file for each study
    parsedirectory, chunk = ([] for i in range(2))
    initial = directory[0][1]
    for num in range(len(directory)):
        if initial == directory[num][1]:
            chunk.append(directory[num] + (orientation_names[orientation[num]], contrast_names[contrast[num]]))
        else:
            parsedirectory.append(chunk)
            chunk = []
            chunk.append(directory[num] + (orientation_names[orientation[num]], contrast_names[contrast[num]]))
            initial = directory[num][1]
    parsedirectory.append(chunk)

    for study in parsedirectory:
        with open(os.path.join(study[0][1],'predictions.csv'),"w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File Name', 'Orientation', 'Contrast'])

            for image in study:
                filename = image[0][len(image[1]) + 1:]
                writer.writerow([filename,image[2],image[3]])

def preprocess_image(img, path):

    # check the dimension of input image
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    elif len(img.shape) == 4:
        img = img.mean(axis=3)
        img = img[:,:,0]
    else:
        logging.warning('      image is out of dimension: %s',path)
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
    
    # print results (debugging only)
    #for i in range(orientations.shape[0]):
        #print('orientation: %s', orientation_names[orientations[i]])
        #print('contrast: %s', contrast_names[contrasts[i]])

    return (orientations, contrasts)

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
