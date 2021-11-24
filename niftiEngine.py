
import dicom2nifti

class niftiEngine():

    def __init__(self,name):
        logger.info("imageReaderWriter.init()")
        self.name = name
        self.output_dir = None
        self.studyList = []