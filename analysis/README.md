# sr-comparison-analysis
Comparison analysis for Superresolution HSC->HST project

### Pre-requisites

1) sep
2) astropy
3) photutils
4) the handy python scripts called by the compare-sr-fits.py script, all included in the repo
5) numpy, matplotlib, etc.

### How to use this repo:

1) Run `perform_comparison.sh`, which will create `*.output.fits` in the `results/` directory.  By default, the script uses the `*1.fits1` image, but you can edit the script to run on other images.

2) The output images contain 3 layers:
    * The data image (HST or SR)
    * The segmentation map determined by running object detection on the HST images (same for both)
    * A convolved version of the data used by photutils to make measurements

3) The notebook `Examine-Detections.ipynb` shows how to make tables of the source catalogs measured from these image.

    * The source catalogs constructed nominally should be identical, object by object, if the images are identical.
    * The HST and SR images are not identical, so we've restricted to detecting sources in the HST images and measuring the properties on the HST and SR images for comparison.
    * By using the HST segmentation maps, we ensure the same number of objects and the same areas of the image are assigned to the same objects in the HST and SR images. This should make comparing object-by-object easy (no missing objects, no need to match separately objects between HST and SR catalogs).
    * The columns I'd recommend comparing in the photutils catalogs are listed in the notebook. When converting between SourceCatalog and astropy tables, these columns are retained.

4) Future ideas
    * We could adapt the notebook scripts to create loss functions based on object-by-object comparisons between the images.