======= Shape-from-Shading Products for the 2022 Artemis III Candidate Landing Regions =======
======= A3CLR22-#6 =======

S. Bertone (creator), R. A. Beyer (reviewer), 2025-12-17

Reference: Bertone S. et al. (2026), Enhanced Topography Models for selected Lunar South Pole regions with Shape-from-Shading, The Planetary Science Journal, doi:XYZ
Data repository: Bertone S. (2026), Zenodo, doi:XYZ

The A3CLR22-#6 ROI is defined with this WKT polygon in long/lat (or DEM CRS):

POLYGON ((-19000.00000000 -4000.00000000, -3000.00000000 -4000.00000000, -3000.00000000 -20000.00000000, -19000.00000000 -20000.00000000, -19000.00000000 -4000.00000000))

and has the following bounding box:

Upper Left  (  -19000.000,   -4000.000) (101d53'19.17"W, 89d21'34.89"S)
Lower Left  (  -19000.000,  -20000.000) (136d28' 7.68"W, 89d 5'25.02"S)
Upper Right (   -3000.000,   -4000.000) (143d 7'48.37"W, 89d50' 6.40"S)
Lower Right (   -3000.000,  -20000.000) (171d28' 9.24"W, 89d19'59.06"S)

The first set of coordinates is in South Polar Stereographic, and the second set is in long/lat.

All map-projected data in these directories are in a Polar Stereographic projection,
defined with the following PROJ.4 String:
+proj=stere +lon_0=0 +lat_0=-90 +R=1737400 +units=m

Standard Goddard Lunar Data (GLD) products follow the ROIID_GLDPROD_RES[_...] naming scheme, where:
  ROIID = A306 (region identifier)
  RES   = 5 (resolution in m/pix)
and additional suffixes encode illumination or baseline where relevant.

This directory contains (where present):

* A306_GLDELEV_005.tif  (GLD01: Elevation)
	Final elevation product (SfS), 32-bit GeoTIFF with heights in meters 
	relative to a lunar radius of 1,737,400 m at 5 m/pixel.

* A306_GLDMASK_005.tif  (GLD02: Data Mask / point count)
	Data mask / count map. Each pixel value indicates the number of illuminated LROC image 
	pixels (DN > 0.001) that contributed to the SfS solution for that pixel.

* A306_GLDISGM_005.tif  (GLD03: Elevation uncertainty)
	Elevation uncertainty map. Pixel values give an estimated height uncertainty (m) 
	derived from perturbation tests using simulated images generated from the SfS DEM.

* A306_GLDDIFF_005.tif  (GLD04: Elevation difference to reference, if present)
	Elevation difference to the reference LOLA DEM, in meters (SfS-derived GLD01 minus LOLA).

* SLOPE_MAPS/A306_GLDTSLP_005_*.tif  (GLD05: Slope at different baselines)
	Slope maps derived from GLD01 at one or more baseline scales.

* ROUGHNESS_MAPS/A306_GLDTRGH_005_*.tif  (GLD06: Roughness at different baselines)
	VRM roughness maps derived from GLD01 at one or more baseline scales.

* A306_GLDHILL_005_315_45.tif  (GLD07: Hillshade)
	Hillshade derived from GLD01 using illumination azimuth 315° and elevation 45°.

* A306_GLDOMOS_005.tif  (GLD08: Orthomosaic)
	Maximally-lit orthomosaic based on LROC NAC images. For each pixel, the brightest 
	valid input pixel is selected. This yields a mosaic with apparent illumination 
	from multiple azimuths while emphasizing all illuminated terrain. 32-bit GeoTIFF.

* A306_GLDBRES_005.tif  (GLD09: Best resolution of input images)
	Best resolution map. Pixel value indicates the best input image resolution (m) 
	for the data used in the SfS solution at that location.

* A306_GLDSBCT_005.tif  (GLD10: Solar bins count per pixel, if present)
	Solar bins count per pixel. Number of distinct solar illumination bins contributing.

* LROC-NAC-maps/A306_GLDIMGP_005_IMGID.tif  (GLD11: Input mapprojected images)
	Input LROC NAC images, orthorectified and cropped to the SfS DEM. Each image is stored 
	as a 32-bit GeoTIFF with name ROIID_GLDIMGP_RES_IMGID.tif, where IMGID encodes the NAC ID.

* LROC-NAC-adjust/A306_GLDACAM_005_IMGID.json  (GLD12: Input adjusted cameras)
	Per-image adjusted camera solutions exported as JSON. Filenames follow the pattern 
	ROIID_GLDACAM_RES_IMGID.json, where IMGID matches the corresponding GLDIMGP entry.

Notes
-----
All rasters are single-band compressed GeoTIFFs, on a common CRS/extent/grid unless otherwise noted.

