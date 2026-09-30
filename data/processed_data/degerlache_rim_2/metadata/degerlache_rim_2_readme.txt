======= Shape-from-Shading Products for the 2022 Artemis III Candidate Landing Regions =======
======= A3CLR22-#4 =======

S. Bertone (creator), R. A. Beyer (reviewer), 2025-12-17

Reference: Bertone S. et al. (2026), Enhanced Topography Models for selected Lunar South Pole regions with Shape-from-Shading, The Planetary Science Journal, doi:XYZ
Data repository: Bertone S. (2026), Zenodo, doi:XYZ

The A3CLR22-#4 ROI is defined with this WKT polygon in long/lat (or DEM CRS):

POLYGON ((-58500.00000000 33000.00000000, -38500.00000000 33000.00000000, -38500.00000000 13000.00000000, -58500.00000000 13000.00000000, -58500.00000000 33000.00000000))

and has the following bounding box:

Upper Left  (  -58500.000,   33000.000) ( 60d34'21.16"W, 87d47' 7.04"S)
Lower Left  (  -58500.000,   13000.000) ( 77d28'16.29"W, 88d 1'26.14"S)
Upper Right (  -38500.000,   33000.000) ( 49d23'55.34"W, 88d19'40.41"S)
Lower Right (  -38500.000,   13000.000) ( 71d20'31.60"W, 88d39'35.95"S)

The first set of coordinates is in South Polar Stereographic, and the second set is in long/lat.

All map-projected data in these directories are in a Polar Stereographic projection,
defined with the following PROJ.4 String:
+proj=stere +lon_0=0 +lat_0=-90 +R=1737400 +units=m

Standard Goddard Lunar Data (GLD) products follow the ROIID_GLDPROD_RES[_...] naming scheme, where:
  ROIID = A304 (region identifier)
  RES   = 5 (resolution in m/pix)
and additional suffixes encode illumination or baseline where relevant.

This directory contains (where present):

* A304_GLDELEV_005.tif  (GLD01: Elevation)
	Final elevation product (SfS), 32-bit GeoTIFF with heights in meters 
	relative to a lunar radius of 1,737,400 m at 5 m/pixel.

* A304_GLDMASK_005.tif  (GLD02: Data Mask / point count)
	Data mask / count map. Each pixel value indicates the number of illuminated LROC image 
	pixels (DN > 0.001) that contributed to the SfS solution for that pixel.

* A304_GLDISGM_005.tif  (GLD03: Elevation uncertainty)
	Elevation uncertainty map. Pixel values give an estimated height uncertainty (m) 
	derived from perturbation tests using simulated images generated from the SfS DEM.

* A304_GLDDIFF_005.tif  (GLD04: Elevation difference to reference, if present)
	Elevation difference to the reference LOLA DEM, in meters (SfS-derived GLD01 minus LOLA).

* SLOPE_MAPS/A304_GLDTSLP_005_*.tif  (GLD05: Slope at different baselines)
	Slope maps derived from GLD01 at one or more baseline scales.

* ROUGHNESS_MAPS/A304_GLDTRGH_005_*.tif  (GLD06: Roughness at different baselines)
	VRM roughness maps derived from GLD01 at one or more baseline scales.

* A304_GLDHILL_005_315_45.tif  (GLD07: Hillshade)
	Hillshade derived from GLD01 using illumination azimuth 315° and elevation 45°.

* A304_GLDOMOS_005.tif  (GLD08: Orthomosaic)
	Maximally-lit orthomosaic based on LROC NAC images. For each pixel, the brightest 
	valid input pixel is selected. This yields a mosaic with apparent illumination 
	from multiple azimuths while emphasizing all illuminated terrain. 32-bit GeoTIFF.

* A304_GLDBRES_005.tif  (GLD09: Best resolution of input images)
	Best resolution map. Pixel value indicates the best input image resolution (m) 
	for the data used in the SfS solution at that location.

* A304_GLDSBCT_005.tif  (GLD10: Solar bins count per pixel, if present)
	Solar bins count per pixel. Number of distinct solar illumination bins contributing.

* LROC-NAC-maps/A304_GLDIMGP_005_IMGID.tif  (GLD11: Input mapprojected images)
	Input LROC NAC images, orthorectified and cropped to the SfS DEM. Each image is stored 
	as a 32-bit GeoTIFF with name ROIID_GLDIMGP_RES_IMGID.tif, where IMGID encodes the NAC ID.

* LROC-NAC-adjust/A304_GLDACAM_005_IMGID.json  (GLD12: Input adjusted cameras)
	Per-image adjusted camera solutions exported as JSON. Filenames follow the pattern 
	ROIID_GLDACAM_RES_IMGID.json, where IMGID matches the corresponding GLDIMGP entry.

Notes
-----
All rasters are single-band compressed GeoTIFFs, on a common CRS/extent/grid unless otherwise noted.

