# system
import os
import subprocess
import logging
import glob
logger=logging.getLogger("curator")

# dicom2nifti
import dicom2nifti

# pandas
import pandas as pd

# pydicom
import pydicom

class niftiEngine():

    def __init__(self,name):
        logger.info("niftiEngine.init()")
        self.name = name
    
    def writeCSVToNifti(self,csv_file,output_dir):
        logger.info("niftiEngine.convertCSVToNifti")
        df=pd.read_csv(csv_file)
        
        # error checking
        if not set(["contrast","orientation","dcmDirectory"]).issubset(df.columns):
            logger.error("niftiEngine could not find columns contrast, orientation, dcmdir in csv file")
            return
            
        for index, row in df.iterrows():
            dcmDirectory = row["dcmDirectory"]
            contrast = row["contrast"]
            orientation = row["orientation"]
            self.process_dcm(dcmDirectory,output_dir,contrast)
            
    def process_dcm(self, input_dir, output_dir,
                exclude_tags=[], inject_tags={}, write_slice_jsons=False):
        os.makedirs(output_dir, exist_ok=True)
        # output filenames
        slice_metadata_file = "slice_metadata.json"
        scan_metadata_file = "scan_metadata.json"
        log_file = "dcm2niix_log.txt"
        cmd_file = "dcm2niix_cmd.txt"
        # convert DCM to NII
        #if os.path.exists(output_dir):
        #   os.remove(output_dir)
        logging.info('      dcm2niix()')
        #try:
        os.path.join(output_dir,contrast)
        with open(os.path.join(output_structure, log_file), 'w') as f_log:
            result = subprocess.run(
                ['dcm2niix', '-f', '%s_%d', '-m', 'n', '-o', output_dir, input_dir],
                stdout=f_log, stderr=subprocess.STDOUT, universal_newlines=True)
        #except:
        #    logging.exception("Exception occurred")
        with open(os.path.join(output_dir, cmd_file), 'w') as f_cmd:
            f_cmd.write(' '.join(result.args))
            logging.info("      "+' '.join(result.args))
        #if result.returncode != 0:
        #    print(f'{output_dir}: dcm2niix failed with code {result.returncode}')
        # extract metadata
        dcm_paths = glob.glob(os.path.join(input_dir, '*.dcm'))
        slice_tag_dicts = []
        if not write_slice_jsons:
            dcm_paths = dcm_paths[0:1] # only process first slice
        for dcm_path in dcm_paths:
            logging.info('      dcmread()')
            try:
                dcm_obj = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            except:
                logging.exception("Exception occurred")
            # logging.info('      process_tag_dict()')
            # try:
                # tag_dict = process_tag_dict(dcm_obj.to_json_dict(), unpack_singleton_values=True)
            # except:
                # logging.exception("Exception occurred")
            # tag_dict["DicomDirectory"] = input_dir
            # tag_dict["DicomSliceFile"] = dcm_path
            # tag_dict["NiftiVolumeFile"] = output_dir if os.path.exists(output_dir) else None
            # for tag in exclude_tags:
                # if tag in tag_dict:
                    # del tag_dict[tag]
            # tag_dict.update(inject_tags)
            # slice_tag_dicts.append(tag_dict)
        # TODO - remove tags that are different across slices?
        # logging.info('      dcm metadata dump()')
        # try:
            # with open(os.path.join(output_dir, scan_metadata_file), 'w') as f:
                # json.dump(slice_tag_dicts[0], f, indent=4)
        # except:
            # logging.exception("Exception occurred: json.dump(%s, %s, %s)", slice_tag_dicts[0], f, indent=4)
        # if write_slice_jsons:
            # with open(os.path.join(output_dir, slice_metadata_file), 'w') as f:
                # json.dump(slice_tag_dicts, f, indent=4)
            
            