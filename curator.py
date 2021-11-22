
# system
import argparse
import os
import sys
import logging

# custom modules
#import study
import imageReaderWriter

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


if __name__ == "__main__":

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Curator - Create databases of imaging data for manual curation")
    parser.add_argument("input_dir", type=str, help="Input directory with dicom images to be curated")
    parser.add_argument("output_dir",type=str,help="Output directory with curated files")
    parser.add_argument("--csv_file",type=str,help="csv file with data to curate",default="curate.csv")
    parser.add_argument("--log_file",type=str,help="txt file with log info",default="curator_log.txt")
    args = parser.parse_args()
    
    # Initialize logger
    initLogger("curator",args.log_file)
    
    # Initialize and read directory to generate study list
    reader = imageReaderWriter.imageReaderWriter("Image Reader")
    studyList=reader.queryDirectory(args.input_dir,2)
    
    # Copy valid series to the new directory and sort
    output_dcmdir=os.path.join(args.output_dir,"dcm")
    studyList=reader.copyValidSeriesAndUpdateStudyList(output_dcmdir)

    # writeStudyListToCSV
    reader.writeStudyListToCSV(args.csv_file)