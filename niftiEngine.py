# system
import logging
logger=logging.getLogger("curator")

# dicom2nifti
import dicom2nifti

# pandas
import pandas as pd

class niftiEngine():

    def __init__(self,name):
        logger.info("niftiEngine.init()")
        self.name = name
    
    def writeCSVToNifti(self,csv_file):
        logger.info("niftiEngine.convertCSVToNifti")
        df=pd.read_csv(csv_file)
        
        # error checking
        if not set(["contrast","orientation","dcmdir"]).issubset(df.columns):
            logger.error("niftiEngine could not find columns contrast, orientation, dcmdir in csv file")
            return