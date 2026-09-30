======= Shape-from-Shading Products for the 2022 Artemis III Candidate Landing Regions =======
======= A3CLR22-#12 'de Gerlache-Kocher Massif' =======

S. Bertone (creator), R. A. Beyer (reviewer), 2025-12-17

Reference: Bertone S. et al. (2026), Enhanced Topography Models for selected Lunar South Pole regions with Shape-from-Shading, The Planetary Science Journal, doi:XYZ
Data repository: Bertone S. (2026), Zenodo, doi:XYZ

The A3CLR22-#12 'de Gerlache-Kocher Massif' ROI is defined with this WKT polygon in long/lat (or DEM CRS):

POLYGON ((-123400.00000000 -46100.00000000, -103400.00000000 -46100.00000000, -103400.00000000 -66100.00000000, -123400.00000000 -66100.00000000, -123400.00000000 -46100.00000000))

and has the following bounding box:

Upper Left  ( -123400.000,  -46100.000) (110d29' 5.19"W, 85d39'28.46"S)
Lower Left  ( -123400.000,  -66100.000) (118d10'33.65"W, 85d23' 9.50"S)
Upper Right ( -103400.000,  -46100.000) (114d 1'45.45"W, 86d16' 4.28"S)
Lower Right ( -103400.000,  -66100.000) (122d35'21.69"W, 85d57'16.41"S)

The first set of coordinates is in South Polar Stereographic, and the second set is in long/lat.

All map-projected data in these directories are in a Polar Stereographic projection,
defined with the following PROJ.4 String:
+proj=stere +lon_0=0 +lat_0=-90 +R=1737400 +units=m

Standard Goddard Lunar Data (GLD) products follow the ROIID_GLDPROD_RES[_...] naming scheme, where:
  ROIID = A312 (region identifier)
  RES   = 5 (resolution in m/pix)
and additional suffixes encode illumination or baseline where relevant.

This directory contains (where present):

* A312_GLDELEV_005.tif  (GLD01: Elevation)
	Final elevation product (SfS), 32-bit GeoTIFF with heights in meters 
	relative to a lunar radius of 1,737,400 m at 5 m/pixel.

* A312_GLDMASK_005.tif  (GLD02: Data Mask / point count)
	Data mask / count map. Each pixel value indicates the number of illuminated LROC image 
	pixels (DN > 0.001) that contributed to the SfS solution for that pixel.

* A312_GLDISGM_005.tif  (GLD03: Elevation uncertainty)
	Elevation uncertainty map. Pixel values give an estimated height uncertainty (m) 
	derived from perturbation tests using simulated images generated from the SfS DEM.

* A312_GLDDIFF_005.tif  (GLD04: Elevation difference to reference, if present)
	Elevation difference to the reference LOLA DEM, in meters (SfS-derived GLD01 minus LOLA).

* SLOPE_MAPS/A312_GLDTSLP_005_*.tif  (GLD05: Slope at different baselines)
	Slope maps derived from GLD01 at one or more baseline scales.

* ROUGHNESS_MAPS/A312_GLDTRGH_005_*.tif  (GLD06: Roughness at different baselines)
	VRM roughness maps derived from GLD01 at one or more baseline scales.

* A312_GLDHILL_005_315_45.tif  (GLD07: Hillshade)
	Hillshade derived from GLD01 using illumination azimuth 315° and elevation 45°.

* A312_GLDOMOS_005.tif  (GLD08: Orthomosaic)
	Maximally-lit orthomosaic based on LROC NAC images. For each pixel, the brightest 
	valid input pixel is selected. This yields a mosaic with apparent illumination 
	from multiple azimuths while emphasizing all illuminated terrain. 32-bit GeoTIFF.

* A312_GLDBRES_005.tif  (GLD09: Best resolution of input images)
	Best resolution map. Pixel value indicates the best input image resolution (m) 
	for the data used in the SfS solution at that location.

* A312_GLDSBCT_005.tif  (GLD10: Solar bins count per pixel, if present)
	Solar bins count per pixel. Number of distinct solar illumination bins contributing.

* LROC-NAC-maps/A312_GLDIMGP_005_IMGID.tif  (GLD11: Input mapprojected images)
	Input LROC NAC images, orthorectified and cropped to the SfS DEM. Each image is stored 
	as a 32-bit GeoTIFF with name ROIID_GLDIMGP_RES_IMGID.tif, where IMGID encodes the NAC ID.

* LROC-NAC-adjust/A312_GLDACAM_005_IMGID.json  (GLD12: Input adjusted cameras)
	Per-image adjusted camera solutions exported as JSON. Filenames follow the pattern 
	ROIID_GLDACAM_RES_IMGID.json, where IMGID matches the corresponding GLDIMGP entry.

Methods
-------
The LOLA map is from the https://pgda.gsfc.nasa.gov/products/78
(where this location is labeled "Site42") based on Barker et al.
[1].  The hillshade maps were also created via GDAL [2].

The Integrated Software for Imagers and Spectrometers
(ISIS, v 7.2.0, [3]) was used for initial processing of the Lunar Reconnaissance
Orbiter Camera (LROC) images.  All of the other data products in
this package were created with the NASA Ames Stereo Pipeline
(ASP, v 3.4.0, [4]), specifically the techniques in
Alexandrov & Beyer [5] for creation of the shape-from-shading terrain
model.  Chapter 12 of the ASP handbook
(https://stereopipeline.readthedocs.io/en/latest/sfs_usage.html)
provides an excellent practical walkthrough on how to apply the
techniques.

The 1469 images that intersected the area (by > 1%) were evaluated based on
their meta-data. Images with a resolution exceeding 1.5 m/pix or an off-nadir
angle > 5 deg were removed. At this stage, the images were given an initial
screening for illumination and jitter, and some were removed. The remaining
images were map-projected at 1 m/pix and bundle adjusted. After assessing their
relative alignment, only images mutually agreeing to <1.8 meters (85%) were retained.
Images with 100 matches or less with any other were also removed. We built stereo DEMs
from a few narrow-convergence angle pairs, then aligned their mosaic
(we average elevations at overlapping areas) to the LOLA terrain to register the
whole cloud of adjusted cameras. At this point, an optimal selection of 239 images
were identified to maximise surface coverage and the variety of illumination
angles and azimuths available at each pixel. After a final verification, these
images were deemed of high enough quality to be included in the final product.


The initial SfS model (ar-sfs-dem-noblend.tif) was created with:

sfs \
    -i ldem_0_5mpp.tif \
    -o sfs_sel0_0.1_1e-3/run \
    --threads 12 \
    --image-list final_selection_images.list \
    --camera-list final_selection_cameras.list \
    --image-exposures-prefix sfs_sel0_0.1_1e-3/run \
    --min-blend-size 20 \
    --allow-borderline-data \
    --bundle-adjust-prefix ba_align/run \
    --crop-input-images \
    --reflectance-type 1 \
    --max-iterations 5 \
    --shadow-threshold 0.002 \
    --save-dem-with-nodata \
    --allow-borderline-data \
    --save-sparingly \
    --blending-dist 100 \
    --smoothness-weight 0.1 \
    --robust-threshold 0.005 \
    --initial-dem-constraint-weight 0.001

Specific settings were used to refine selected sub-tiles, namely removing
--allow-borderline-data or adding --float-albedo.


Product Description
-------------------
These data products fit the definition of "foundational" data
products [6]. The source LROC image data from which the topography
and mosaic products are created are "controlled" by means of bundle
adjustment. These data are "absolutely controlled" because they
are rigorously tied to the underlying LOLA geodetic coordinate
reference frame.

The LOLA and SfS-DEM products are orthonormic heights relative to
the mean lunar radius of 1,737,400 m.  It is important to note that
the SfS-DEM data do not have fundamentally different absolute RMS
errors from the errors reported by [1] for the LOLA data.  The
SfS-DEM data points are simply different estimates of elevation
than the LOLA data points, but absolute RMS errors are the same as
the parent LOLA data set.

The LROC NAC maps are orthorectified and have had their pixels projected
onto the SfS terrain model. Pixel values in these data products were
sourced from images that were radiometrically calibrated via ISIS [3]
and are thus in units of I/F.

Comparison of the height-error and weight maps in this package to other
sfs solutions will show sfs data in what appears to be unilluminated
areas.  This is an effect of choosing the --shadow-threshold to be
only 0.002.  The effect of choosing a low value results in more "dark"
pixels being treated as "real" pixels that sfs attempts to solve for,
resulting in height-error and weight values in dark zones. The reason
this value was chosen is because selecting a higher value resulted in
the dark floors of smaller craters not being treated correctly. This
altered value produces some noise in large unilluminated areas (which is
obviously erroneous), but provides better terrain for these small craters
in this area.


Credit
------
These shape-from-shading terrain models and associated products
were created by

Stefano Bertone, Univ. of Maryland College Park / NASA Goddard Space Flight Center

and

Ross A. Beyer
at NASA Ames Research Center

If you have questions about these data products, please contact
Ross.A.Beyer@nasa.gov.

The creation of these products was funded by NASA's Moon to Mars (M2M)
project.

Resources supporting this work were provided by the NASA High-End
Computing (HEC) Program through the NASA Center for Climate Simulation (NCCS)
at Goddard Space Flight Center.


References
----------
1. Barker, M.K., et al. (2021),
   Improved LOLA Elevation Maps for South Pole Landing Sites: Error
   Estimates and Their Impact on Illumination Conditions,
   Planetary & Space Science, Volume 203, 1 September 2021, 105119,
   https://doi.org/10.1016/j.pss.2020.105119

2. GDAL/OGR contributors (2020),
   GDAL/OGR Geospatial Data Abstraction Software Library,
   Open Source Geospatial Foundation,http://www.gdal.org

3. Laura, et al. (2023),
   ISIS 8.0.0 Public Release.
   https://isis.astrogeology.usgs.gov

4. O. Alexandrov, et al. (2023),
   NeoGeographyToolkit/StereoPipeline 3.4.0-alpha-2023-09-21
   Zenodo, https://doi.org/10.5281/zenodo.8366083

5. Alexandrov, O. & R. A. Beyer (2018),
   Multiview shape-from-shading for planetary images.
   Earth and Space Science, 5, 652-666,
   https://doi.org/10.1029/2018EA000390

6. Laura, J. R. & R. A. Beyer (2021),
   Knowledge Inventory of Foundational Data Products in Planetary Science.
   Planetary Science Journal, Volume 2, 18
   https://doi.org/10.3847/PSJ/abcb94