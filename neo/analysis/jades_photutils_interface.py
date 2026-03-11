import numpy as np
from astropy.io import fits
from photutils import segmentation
from photutils import SegmentationImage
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from photutils.segmentation import SourceCatalog
from photutils.segmentation import detect_sources
from detect_and_mask import iterative_detect_and_mask, detect_and_mask


import sep

class EditableSourceCatalog(SourceCatalog):
    """Edtiable version of the Photutils Source Catalog"""
    def print_object(self,index):
        if(self.isscalar):
            print("Attempting to print index = {index} for a scalar SourceCatalog...")
        else:
            print(f"Testing source catalog: {self.xcentroid[index],self.ycentroid[index]}")
            print(f"object index {index}")
            print(f"object label {self.label[index]}")
            print(f"xcentroid {self.xcentroid[index]}")
            print(f"ycentroid {self.ycentroid[index]}")
            print(f"sky_centroid {self.sky_centroid[index]}")
            print(f"sky_centroid_icrs {self.sky_centroid_icrs[index]}")
            print(f"kron_flux {self.kron_flux[index]}")
            print(f"kron_fluxerr {self.kron_fluxerr[index]}")
            print(f"kron_radius {self.kron_radius[index]}")
            print(f"kron_aperture {self.kron_aperture[index]}")
            print(f"isscalar {self.isscalar}")
            print(f"cxx {self.cxx[index]}")
            print(f"cxy {self.cxy[index]}")
            print(f"cyy {self.cyy[index]}")

    def copy_object(self, donor_catalog,index):
        self.xcentroid[index] = donor_catalog[index].xcentroid
        self.ycentroid[index] = donor_catalog[index].ycentroid
        self.sky_centroid[index] = donor_catalog[index].sky_centroid
        self.sky_centroid_icrs[index] = donor_catalog[index].sky_centroid_icrs
        self.sky_bbox_ur[index] = donor_catalog[index].sky_bbox_ur
        self.sky_bbox_ul[index] = donor_catalog[index].sky_bbox_ul
        self.sky_bbox_lr[index] = donor_catalog[index].sky_bbox_lr
        self.sky_bbox_ll[index] = donor_catalog[index].sky_bbox_ll
        self.semiminor_sigma[index] = donor_catalog[index].semiminor_sigma
        self.semimajor_sigma[index] = donor_catalog[index].semimajor_sigma
        self.segment_ma[index] = donor_catalog[index].segment_ma
        self.segment_fluxerr[index] = donor_catalog[index].segment_fluxerr
        self.segment_flux[index] = donor_catalog[index].segment_flux
        self.segment[index] = donor_catalog[index].segment
        #self.properties[index] = donor_catalog[index].properties
        self.orientation[index] = donor_catalog[index].orientation
        #self.nlabels[index] = donor_catalog[index].nlabels
        self.moments_central[index] = donor_catalog[index].moments_central
        self.moments[index] = donor_catalog[index].moments
        self.minval_yindex[index] = donor_catalog[index].minval_yindex
        self.minval_xindex[index] = donor_catalog[index].minval_xindex
        self.minval_index[index] = donor_catalog[index].minval_index
        self.min_value[index] = donor_catalog[index].min_value
        self.maxval_yindex[index] = donor_catalog[index].maxval_yindex
        self.maxval_xindex[index] = donor_catalog[index].maxval_xindex
        self.maxval_index[index] = donor_catalog[index].maxval_index
        self.max_value[index] = donor_catalog[index].max_value
        self.local_background_aperture[index] = donor_catalog[index].local_background_aperture
        self.local_background[index] = donor_catalog[index].local_background
        self.kron_radius[index] = donor_catalog[index].kron_radius
        self.kron_fluxerr[index] = donor_catalog[index].kron_fluxerr
        self.kron_flux[index] = donor_catalog[index].kron_flux
#        self.isscalar[index] = donor_catalog[index].isscalar
        
        self.inertia_tensor[index] = donor_catalog[index].inertia_tensor
        self.gini[index] = donor_catalog[index].gini
        self.fwhm[index] = donor_catalog[index].fwhm
        self.error_ma[index] = donor_catalog[index].error_ma
        self.error[index] = donor_catalog[index].error
        self.equivalent_radius[index] = donor_catalog[index].equivalent_radius
        self.elongation[index] = donor_catalog[index].elongation
        self.eccentricity[index] = donor_catalog[index].eccentricity
        self.data_ma[index] = donor_catalog[index].data_ma
        self.data[index] = donor_catalog[index].data
        self.cyy[index] = donor_catalog[index].cyy
        self.cxy[index] = donor_catalog[index].cxy
        self.cxx[index] = donor_catalog[index].cxx
        self.cutout_maxval_index[index] = donor_catalog[index].cutout_maxval_index
        self.cutout_minval_index[index] = donor_catalog[index].cutout_minval_index
        self.cutout_centroid[index] = donor_catalog[index].cutout_centroid
        self.covariance_eigvals[index] = donor_catalog[index].covariance_eigvals
        self.covariance[index] = donor_catalog[index].covariance
        self.covar_sigy2[index] = donor_catalog[index].covar_sigy2
        self.covar_sigxy[index] = donor_catalog[index].covar_sigxy
        self.covar_sigx2[index] = donor_catalog[index].covar_sigx2
        self.convdata_ma[index] = donor_catalog[index].convdata_ma
        self.convdata[index] = donor_catalog[index].convdata
        self.centroid[index] = donor_catalog[index].centroid
        self.bbox_ymax[index] = donor_catalog[index].bbox_ymax
        self.bbox_ymin[index] = donor_catalog[index].bbox_ymin
        self.bbox_xmax[index] = donor_catalog[index].bbox_xmax
        self.bbox_xmin[index] = donor_catalog[index].bbox_xmin
        self.bbox[index] = donor_catalog[index].bbox
        self.background_sum[index] = donor_catalog[index].background_sum
        self.background_ma[index] = donor_catalog[index].background_ma
        self.background_centroid[index] = donor_catalog[index].background_centroid
        self.background[index] = donor_catalog[index].background
        self.area[index] = donor_catalog[index].area

    #if('extra_properties' in donor[idx].properties):
    #    recv[idx].extra_properties = donor[idx].extra_properties

def WriteSourceCatalogImages(fname,data_sub,segm_deblend,convolved_data,header=None):

    """Write out the FITS file containing the 
    images used to construct a SourceCatalog
    via photutils"""

    #create an hdu for each image
    hdu_primary = fits.PrimaryHDU(header=header)
    hdu_data_sub = fits.ImageHDU(data=data_sub,header=header)
    hdu_segm_deblend = fits.ImageHDU(data=segm_deblend,header=header)
    hdu_convolved_data = fits.ImageHDU(data=convolved_data,header=header)

    #create the hdu list
    hdu_list = fits.HDUList(hdus=[hdu_primary,hdu_data_sub,hdu_segm_deblend,hdu_convolved_data]) 

    #write the hdu list to a FITS file
    hdu_list.writeto(fname,overwrite=True)

def ReadSourceCatalogImages(fname):

    """Read in the FITS file containing the 
    images used to construct a SourceCatalog
    via photutils"""

    hdu_list = fits.open(fname)
    data_sub = hdu_list[1].data
    segm_deblend = hdu_list[2].data
    convolved_data = hdu_list[3].data
    return data_sub,segm_deblend,convolved_data

def ObjectFittingCutoutRegion(tbl,idx,shape,f_cutout=3,f_grow=2,verbose=False):

    if(verbose):
        print(f"Object {idx} xcentroid: {tbl[idx]['xcentroid']}")
        print(f"Object {idx} ycentroid: {tbl[idx]['ycentroid']}")
        print(f"Object {idx} semimajor_sigma {tbl[idx]['semimajor_sigma'].value}")

    x_cutout_center = int(tbl[idx]['xcentroid'])
    y_cutout_center = int(tbl[idx]['ycentroid'])

    #initial size of cutout in multiples of semimajor_sigma
    dx_cutout = f_cutout*int(tbl[idx]['semimajor_sigma'].value)

    #get fiducial ymin and ymax
    ymin = y_cutout_center - dx_cutout
    ymax = y_cutout_center + dx_cutout
    xmin = x_cutout_center - dx_cutout
    xmax = x_cutout_center + dx_cutout

    if(verbose):
        print(f"Initial ymin,ymax = {ymin},{ymax}")
        print(f"Initial xmin,xmax = {xmin},{xmax}")

    ymin_check = int(tbl[idx]['bbox_ymin'] - f_grow*tbl[idx]['semimajor_sigma'].value)
    xmin_check = int(tbl[idx]['bbox_xmin'] - f_grow*tbl[idx]['semimajor_sigma'].value)
    ymax_check = int(tbl[idx]['bbox_ymax'] + f_grow*tbl[idx]['semimajor_sigma'].value)
    xmax_check = int(tbl[idx]['bbox_xmax'] + f_grow*tbl[idx]['semimajor_sigma'].value)

    if(verbose):
        print(f"Prev ymin,xmin = {ymin,xmin}")
        print(f"Nominal ymin,xmin check = {ymin_check,xmin_check}")
        print(f"Nominal ymin,xmin check = {ymax_check,xmax_check}")

    if(ymin>ymin_check):
        ymin = ymin_check
    if(xmin>xmin_check):
        xmin = xmin_check
    if(ymax<ymax_check):
        ymax = ymax_check
    if(xmax<xmax_check):
        xmax = xmax_check

    #make the image square
    dxx = xmax-xmin
    dyy = ymax-ymin
    if(dxx>dyy):
        ymax += 0.5*(dxx-dyy)
        ymin -= 0.5*(dxx-dyy)
    elif(dyy>dxx):
        xmax += 0.5*(dyy-dxx)
        xmin -= 0.5*(dyy-dxx)

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)
       
    #limit to edges of image
    if(ymin<0):
        ymin = 0
    if(xmin<0):
        xmin = 0
    if(ymax>shape[1]):
        ymax = shape[1]
    if(xmax>shape[0]):
        xmax = shape[0]

    if(verbose):
        print(f"Final xmin,xmax = {xmin},{xmax}")
        print(f"Final ymin,ymax = {ymin},{ymax}")
    return xmin,xmax,ymin,ymax


def CombineSegmaps(segm_original,segm_model,verbose=False):
    """Creates a combined segmentation map
    between a model corrected segmentation map
    for a single object and an original segmentation map"""

    if(verbose):
        print(f"Segmap original min/max : {segm_original.data.min()}/{segm_original.data.max()}")
        print(f"Segmap model min/max : {segm_model.data.min()}/{segm_model.data.max()}")

    #create data array for the combined
    #segmentation map
    segmap_res_comb_data = np.zeros_like(segm_original.data)
#    segmap_res_comb_data = segm_model.copy()

    #get the indices of non-zero elements of input and corrected
    #segmentation
    idx_overlap = np.where( (segm_model.data!=0)&(segm_original.data!=0) )

    #get indices of the segmap that are zero in the model but contain
    #objects in the original
    idx_segres_noint  = np.where( (segm_model.data==0)&(segm_original.data!=0) )

    #get indices of the segmap that have an object in the model but are
    #zero in the original
    idx_segres_zeros   = np.where( (segm_model.data!=0)&(segm_original.data==0) )


    #loop through objects in the original segmap that are zeros
    #in the model segmap, and add them to the combined segmap data
    for label in np.unique(segm_original.data[idx_overlap]):
        idx_label = np.where(segm_original.data==label)
        segmap_res_comb_data[idx_label] = label

    #add the maximum index value to the zero regions in the original
#    segmap_res_comb_data[idx_segres_zeros] += segm_original.data.max()
    segmap_res_comb_data[idx_segres_zeros] = 1 + segm_original.data.max()

    #
    idx_comb_noint = np.where((segmap_res_comb_data==0)&(segm_original.data!=0))
    
    if(verbose):
        print("Length of zeros in combined that are non zero in original = ",len(idx_comb_noint),len(idx_comb_noint[0]))
    segmap_res_comb_data[idx_comb_noint] = segm_original.data[idx_comb_noint]

    #create a segmentation map from the data
    segmap_res_comb = SegmentationImage(segmap_res_comb_data)

    #return the new combined segmentation map
    return segmap_res_comb


def GetGeometry(cat_tab,idx_obj,xmin=0,ymin=0):
    """Get the ellipsoidal geometry object from
    and object in a tabular view of a catalog"""

    #cat_tab is the tabular view of the source catalog
    #created with "cat.to_table()"

    #idx_obj is index of the object of interest

    #xmin is the minimum xpixel of the cutout
    #containing the object

    #ymin is the minimum of the ypixe of the cutout
    #containing the object
    x0 = cat_tab[idx_obj]['xcentroid']-xmin
    y0 = cat_tab[idx_obj]['ycentroid']-ymin
    sma = cat_tab[idx_obj]['semimajor_sigma'].value
    eps = cat_tab[idx_obj]['eccentricity'].value
    pa = cat_tab[idx_obj]['orientation'].value
    
    #compute the ellipse geometry from the source properties
    geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa * np.pi / 180.)

    #return the geometry of the object
    return geometry



def SubtractEllipsoidalModel(data_cutout, cat_tab, idx_obj, xmin=0, ymin=0, flag_bkg=True, sma_factor=6, verbose=False, flag_monotonic=True):

    """Fit and subtract an ellipsoidal model of an object from an image"""

    #data_cutout is the cutout of the original flux
    #image containing the object we want to remove

    #cat_tab is the tabular view of the source catalog
    #created with "cat.to_table()"

    #idx_obj is index of the object of interest

    #xmin is the minimum xpixel of the cutout
    #containing the object

    #ymin is the minimum of the ypixe of the cutout
    #containing the object

    #copy the cutout centered on the object
    data_co_mask = data_cutout.copy()

    #mark all nan as zeros
    data_co_mask[np.isnan(data_cutout)==True] = 0

    #get the ellipsoidal geometry of the object to remove
    geometry = GetGeometry(cat_tab, idx_obj, xmin=xmin, ymin=ymin)

    if(verbose):
        print(f"GEO SMA = {geometry.sma}")

    #get the ellipse model
    ellipse = Ellipse(data_co_mask, geometry)

    #fit the image isophotes
    isolist = ellipse.fit_image(maxsma = geometry.sma*sma_factor)

    if(flag_monotonic):
        if(verbose):
            print("Limiting isolist to monotonic behavior...")
        imin = np.argmin(isolist.intens)
        fmin = isolist.intens[imin]
        if(verbose):
            print(f"imin {imin} fmin {fmin}")
        nintens = len(isolist.intens)
        for i in range(imin,nintens,1):
            print(i,isolist[i].intens,fmin)
            isolist[i].intens = fmin
#            if(isolist.intens[i]>fmin):
#                isolist.__setitem__(i,fmin)
#                isolist.intens[i] = fmin


    #create model image
    model_image = build_ellipse_model(data_co_mask.shape, isolist)

    #create a residual image
    residual = data_cutout - model_image

    #return the result
    return residual, model_image, isolist, ellipse


def SubtractAndCorrect(data_cutout, cat_tab, idx_obj, xmin=0, ymin=0, threshold=3., sma_factor=6, verbose=False):

    """Given an input cutout of data, a photutils source catalog in tabular form,
    an object index, the offsets for the data cutout from the image used to generate
    the catalog, and a detection threshold, this will:

    1) create an ellipsoidal model for an object and remove it from an image
    2) perform background subtraction and detection using SEP of the residual image sources
    3) detect the model object from the model image
    4) correct the segmap from detection on the residual image with the model object
    5) correct the catalog computed from the residual with the properties 
       of the object in the cutout image"""

    #get residual, model image, isolist, and ellipse
    residual, model_image, isolist, ellipse = SubtractEllipsoidalModel(data_cutout, cat_tab, idx_obj, xmin=xmin, ymin=ymin, sma_factor=sma_factor, verbose=verbose)



    #perform SEP object detection
    #and create a Photuils segmap
    #from the result
    #get a mask of nan's
    residual_mask = np.zeros_like(residual,dtype=bool)
    residual_mask[np.isnan(residual)==True] = True

    #measure a background
    residual_bkg = sep.Background(residual, mask=residual_mask)

    #subtract the background
    #need to check whether this residual is best for object detection, or if a constant bkg would be better
    residual_sub = residual - residual_bkg

    #perform object detection
    #on background-subtracted residual image
    #using SEP
    residual_objects, segmap_residual_sep = sep.extract(residual_sub, threshold, err=residual_bkg.globalrms, mask=residual_mask, segmentation_map=True)


    #Create a Photutils SegmentationImage from the SEP segmap
    segmap_residual = segmentation.SegmentationImage(segmap_residual_sep)


    #for brightest objects this does not work well...
    #segmap_residual = detect_sources(residual_sub, threshold, npixels=5)
    #segmap_residual_deblend = segmentation.deblend_sources(residual_sub, segmap_residual, npixels=5, nlevels=32, contrast=0.001)


    #Create a Photutils SegmentationImage from the model image
    npixels = 5
    segmap_model = detect_sources(model_image, 10, npixels=npixels)

    #corrected segmap with the model object inserted
    segmap_corrected = CombineSegmaps(segmap_residual,segmap_model,verbose=verbose)

    #source catalog created from original cutout image
    #and corrected segmap
    cat_data_cutout = SourceCatalog(data_cutout, segmap_corrected)

    #corrected source catalog created from the
    #background-subtracted residual image between data and model
    #and the corrected segmap
    cat_corrected = EditableSourceCatalog(residual_sub, segmap_corrected)

    #copy from catalog created from the cutout
    #for the selected object with index idx_max
    idx_max = np.argmax(segmap_corrected.areas)
    cat_corrected.copy_object(cat_data_cutout,idx_max)

    #return catalog of the region and corrected segmentation map
    #along with the residual and model images
    return cat_corrected, segmap_corrected, residual, model_image, segmap_residual, isolist#, residual_bkg.back()