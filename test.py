from pickletools import float8
import numpy as np
import nibabel as nb
from regex import I, R
from sklearn.feature_extraction import img_to_graph

from PIL import Image as im
import matplotlib.pyplot as plt

def alternatepreprocessing(img, path):
    img = np.asanyarray(img.dataobj)

    # check the dimension of input image
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    elif len(img.shape) == 4:
        img = img.mean(axis=3)
        img = img[:,:,0]
    else:
        print('       image is out of dimension: %s',path)
        return
    
    img = np.array(im.fromarray(img).resize((150, 150)))
    img *= 255.0 / np.amax(img)
    return 255.0 - img

path = '/Users/jinseokim/Downloads/tof-nifti/7094540/9033_SecondaryCapturey.nii'
path2 = '/Users/jinseokim/Downloads/Curator/nii_data/DEETJEN_JEAN_M/2_trufi_loc_multi_iPAT@c_1.3.12.2.1107.5.2.30.57146.20150327102716791321761.0.0.0.nii'
img = nb.load(path2)
print(img.shape)
print(img.get_data_dtype())
output = alternatepreprocessing(img, path)



#img *= (255.0/(np.amax(replicate)))
#png = im.fromarray(img).resize((150, 150))
#img = 255.0 - np.array(png)