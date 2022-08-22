import numpy as np
import nibabel as nb
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import glob
import os
import pydicom

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

roi_size = (256, 256)
sw_batch_size = 1
#import shutil

dicom = pydicom.dcmread(r'C:\Users\jinse\Downloads\test\10017_sa_multi-slice_base_to_apex\1_1.3.6.1.4.1.53684.1.1.4.1671342546.10164.1629290067.503013.dcm')
source = r'C:\Users\jinse\Downloads\test\10017_sa_multi-slice_base_to_apex\1_1.3.6.1.4.1.53684.1.1.4.1671342546.10164.1629290067.503013.dcm'
destination = r'C:\Users\jinse\Downloads\test\1_1.3.6.1.4.1.53684.1.1.4.1671342546.10164.1629290067.503013.dcm'

print(type(dicom.PixelData))
temp = dicom.pixel_array

plt.figure()
plt.imshow(dicom.pixel_array, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

img = np.rot90(dicom.pixel_array.astype('float64'), 3)
slice = torch.from_numpy(img.copy()).float().unsqueeze(0).unsqueeze(0)
output = torch.argmax(sliding_window_inference(slice, roi_size, sw_batch_size, model), dim=1).detach()
print(output.shape)
output = np.transpose(output[0,:,:].numpy()) * 100
output = output.astype(np.int64)
plt.figure()
plt.imshow(output, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()


#dest = shutil.copyfile(source, destination)

arr = dicom.pixel_array
#print(dicom.PixelData)
print(arr.shape)
assert arr.shape == output.shape
#arr = output
arr = np.ones(arr.shape)

dicom.PixelData = output.astype(np.int16).tobytes()
dicom.SeriesNumber = dicom.SeriesNumber + 3000
dicom.save_as('temp.dcm')

print(output.shape)
print(type(output))
plt.figure()
plt.imshow(output, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()