
# system
import argparse
import os
import sys
import logging

# custom modules
import imageReaderWriter
import niftiEngine
import classifier

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

if __name__ == "__main__":

    default_include_tags_txt = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "include_tags_default.txt")

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Curator - Create databases of imaging data for manual curation")
    parser.add_argument('curator_mode', choices=['parse', 'nifti', 'train','classify'])
    parser.add_argument("--input_dir", type=str, help="Input directory with dicom images to be curated")
    parser.add_argument("--output_dir",type=str,help="Output directory with curated files")
    parser.add_argument("--csv_file",type=str,help="csv file with data to curate")
    parser.add_argument("--accession_as_folder",type=str,help="use accession number as folder name",default=True)
    parser.add_argument("--log_file",type=str,help="txt file with log info",default="curator_log.txt")
    parser.add_argument("--include_tags_txt", type=str, default=default_include_tags_txt,
        help="List of tags to retain in metadata JSON")
    args = parser.parse_args()
    
    # Initialize logger
    logger=initLogger("curator",args.log_file)
    
    # Parse Directory Mode
    # create a csv file of image series to be labeled
    # todo: merge several csv sheets of labeled data
    if(args.curator_mode=="parse"):
        # In parse directory mode, we prepare a csv file of image series for labeling
        logger.info("Parsing Mode")
        
        if args.input_dir is None or args.output_dir is None or args.csv_file is None:
            logger.error("    Check that --input_dir, --output_dir, --csv_file are defined")
            sys.exit()
        
        # Initialize and read directory to generate study list
        reader = imageReaderWriter.imageReaderWriter("Image Reader")
        studyList=reader.queryDirectory(args.input_dir,2)
        
        # Copy valid series to the new directory and sort
        output_dcmdir=os.path.join(args.output_dir,"dcm")
        studyList=reader.copyValidSeriesAndUpdateStudyList(output_dcmdir,args.accession_as_folder)

        # writeStudyListToCSV
        reader.writeStudyListToCSV(args.csv_file)
        sys.exit()
        
    # Generate Nifti Mode
    # create nifti files for training the NN how to label cardiac images
    if(args.curator_mode=="nifti"):
        # In parse directory mode, we prepare a csv file of image series for labeling
        logger.info("Nifti Mode")
        
        if args.csv_file is None or args.output_dir is None:
            logger.error("    Check that --csv_file, --output_dir are defined")
            sys.exit()
        
        # Initialize niftiEngine
        ne=niftiEngine.niftiEngine("NiftiEngine")

        # Given a csv file with 
        print(type(args.accession_as_folder))
        ne.writeCSVToNifti(args.csv_file,args.output_dir,args.include_tags_txt,args.accession_as_folder)
        sys.exit()

    # Classify mode
    # use pre-trained model to generate predictions about input image dataset
    if(args.curator_mode=="classify"):
        # A valid csv file has to be given as an input
        logger.info("Classify Mode")
        
        if args.input_dir is None:
            logger.error("    Check that --input_dir is defined")
            sys.exit()
        
        classifier.classifyNifti(args.input_dir)
        sys.exit()
    
