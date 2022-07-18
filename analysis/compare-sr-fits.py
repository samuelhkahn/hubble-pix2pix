import argparse
import numpy as np
from astropy.io import fits
import quicklook as jql
import jades_photutils_interface as jpui
import detect_and_mask as dam
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceCatalog
from photutils import SegmentationImage
import time

#scaling
def ds9_scaling(x, a=1000,offset = 0):
    return (np.log10(a*x + 1)/np.log10(a + 1)) - offset

#unscaling
def ds9_unscaling(x, a=1000,offset = 0 ): 
    return (((a + 1)**(x+offset) - 1) / a)

#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Detection flags and options from user.")

    parser.add_argument('-hst',
                default='hst1.fits',
                metavar='hst_image',
                type=str,
                help='Specify the input HST image.')

    parser.add_argument('-sr',
                default='sr1.fits',
                metavar='sr_image',
                type=str,
                help='Specify the input SR image.')

    parser.add_argument('-hsc',
                default='hsc1.fits',
                metavar='hsc_image',
                type=str,
                help='Specify the input HSC image.')

    parser.add_argument('-t','--threshold',
                default=5.0,
                type=float,
                help='Threshold value for object detection.')

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)


    return parser

#function to compute convolved data image
def CreateConvolvedData(data,verbose=False):
    #perform object detection
    time_start = time.time()

    #smooth the image with a gaussian with FWHM = 3 pixels
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)

    #replace Nan with 0
    data_sub_zeros = data.copy()
    data_sub_zeros[np.isnan(data)==True] = 0

    #smooth the image slightly
    convolved_data = convolve(data_sub_zeros, kernel, normalize_kernel=True)

    time_end = time.time()

    if(verbose):
        print(f"Time to convolve data in SR images = {time_end-time_start}.")

    #return the convolved data image
    return convolved_data

#function to compute source catalog from image
def CreateSourceCatalog(data,threshold=5.,verbose=False):

    #perform object detection
    time_start = time.time()

    #get convolved data
    convolved_data = CreateConvolvedData(data,verbose=verbose)

    #set the minimum # of pixels for a source
    npixels = 5

    #detect sources using threshold x rms as the detection threshold
    segm = detect_sources(convolved_data, threshold, npixels=npixels)

    #deblend sources
    segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels, nlevels=32, contrast=0.001)
    time_end = time.time()

    #create the source catalog
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)

    if(verbose):
        print(f"Time to detect, deblend, and catalog sources in SR images = {time_end-time_start}.")



    #return cat and the images
    return cat, data, segm_deblend, convolved_data

#main function
def main():

    #correct missing offset?
    flag_corr = True

    #create the command line argument parser
    parser = create_parser()

    #store the command line arguments
    args   = parser.parse_args()

    #print arguments
    if(args.verbose):
        print(f"HST image = {args.hst}")
        print(f"HSC image = {args.hsc}")
        print(f"SR  image = {args.sr}")
        print(f"Threshold for object detection = {args.threshold} x RMS of HST image.")

    #open the images
    hdu_hst = fits.open(args.hst)
    hdu_hsc = fits.open(args.hsc)
    hdu_sr  = fits.open(args.sr)

    hdu_hst.info()

    #get the images
    data_hst = hdu_hst[0].data
    data_hsc = hdu_hsc[0].data
    data_sr  = hdu_sr[0].data


    #print statistics of each object
    if(args.verbose):
        print(f"HST:")
        jql.stats(data_hst)
        print(f"HSC:")
        jql.stats(data_hsc)
        print(f"SR:")
        jql.stats(data_sr)

    #make a detection threshold
    data_hst_rms = np.nanstd(data_hst)

    #catalog the sources in the hst image
    cat_hst, data_sub_hst, segm_hst, cdata_hst = CreateSourceCatalog(data_hst,threshold=args.threshold*data_hst_rms,verbose=args.verbose)

    #use the results of the HST image to measure the SR object population
    cdata_sr = CreateConvolvedData(data_sr,verbose=args.verbose)

    #save the detection image data
    #use the segmentation image from HST
    jpui.WriteSourceCatalogImages(args.hst+'.output.fits',data_hst, segm_hst, cdata_hst)
    jpui.WriteSourceCatalogImages(args.sr +'.output.fits',data_sr,  segm_hst, cdata_sr)

#run the script
if __name__=="__main__":
    main()