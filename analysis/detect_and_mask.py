import sep
import numpy as np
def detect_and_mask(data, scale=3.0, threshold=1.5, mask=None):
    
    #reset set_pixstack
    sep.set_extract_pixstack(1000000)
    
    if(mask is None):
        mask = np.zeros_like(data,dtype=bool)
    
    #measure a background
    bkg = sep.Background(data, mask=mask)
    
    #subtract the background
    data_sub = data - bkg

    #perform object detection
    objects = sep.extract(data_sub, threshold, err=bkg.globalrms, mask=mask)

    #create a boolean elliptical mask
    sep.mask_ellipse(mask,objects['x'],objects['y'],objects['a'],objects['b'],objects['theta'],r=scale)
    
    #return the mask
    return mask, objects

def iterative_detect_and_mask(data_in, scale=3.0, niter=3, threshold=1.5, flag_swap=False):

    if(flag_swap):
        data = data_in.byteswap().newbyteorder()
    else:
        data = data_in
        
    mask, objects = detect_and_mask(data, scale=scale, threshold=threshold)
    
    for i in range(niter-1):
        mask, new_objects = detect_and_mask(data,scale=scale, threshold=threshold, mask=mask)
        objects = np.append(objects,new_objects)
    return mask, objects
