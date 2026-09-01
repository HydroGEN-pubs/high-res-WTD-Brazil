import numpy as np
import rioxarray
import xarray as xr

base_dir = ''
out_dir = ''
elevation = rioxarray.open_rasterio(base_dir+'hydrosheds_elev_5641.tif')
elevation = elevation.squeeze('band')


##################################
######  WTD PREDICTION  TIF ######
##################################
wtd_prediction_np = np.load(out_dir+'wtd_mean_estimate_RF_median.npy')

wtd_prediction = xr.DataArray(
    wtd_prediction_np,  
    dims=elevation.dims,
    coords=elevation.coords,
    name='wtd_median'
)

wtd_prediction.rio.write_crs("epsg:5641", inplace=True)
wtd_prediction.rio.to_raster(out_dir+'wtd_mean_estimate_RF_median.tif')

del wtd_prediction_np, wtd_prediction

######################
###### IQR  TIF ######
######################

iqr = np.load(out_dir+'wtd_mean_estimate_RF_Brasil_IQR.npy')

iqr_prediction = xr.DataArray(
    iqr,  
    dims=elevation.dims,
    coords=elevation.coords,
    name='wtd_iqr'
)

iqr_prediction.rio.write_crs("epsg:5641", inplace=True)
iqr_prediction.rio.to_raster(out_dir+'wtd_mean_estimate_RF_Brasil_IQR.tif')

del iqr, iqr_prediction
