import numpy as np
import xarray
import rioxarray
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os


out_dir = ''

# Load model
with open(out_dir+'RF_model_FINAL.pkl', 'rb') as file:
    loaded_RF_model = pickle.load(file)


estimators = loaded_RF_model.estimators_
n_estimators = 300


in_vars =['elevation','slope','x_dist_stream', 'hand',
           'pr','T2m', 'PME',
           'log_hydraulic_cond', 'sand','clay']

'''
Dataloader for WTD mean estimation over Brazil (based on code from Ma et al 2026, doi.org/10.1038/s43247-025-03094-3)
'''

class all_WTD_Mean_Estimation_Data:
    '''
    Return dataset used to contruct ML models for estimating water table depth (WTD)
    over Brazil.

    Input data are:
                    1. Elevation (elevation);
                    2. Topographic slope (slope);
                    3. Horizontal distance to streams (x_dist_stream);
                    4. Height above nearest drainage (hand);
                    5. Annual mean precipitation (pr);
                    6. Annual mean temperature (T2m);
                    7. Precipitation minus evapotranspiration (PME);
                    8. Log of hydraulic conductivity (log_hydraulic_cond);
                    9. Sand content (sand);
                    10. Clay content (clay).

    Output data are: long-term mean water table depth (mwtd)

    :param in_vars:
        Input variables.
    :param indexes:
        Indexes of grid cells where data are obtained. 
    :param base_dir:
        Locations of data used to construct ML models. 
    '''
    def __init__(
        self,
        in_vars=None,
        grid_cell_indexes=None,
        grid_cell_indexes_era5=None,
        grid_cell_indexes_soilgrid=None,
        loc_indexes=None,
        base_dir=''
    ):
        super().__init__()
        self.in_vars = in_vars

        # Get paths for files with input data
        data_paths = {
            'elevation': base_dir+'hydrosheds_elev_5641.tif',
            'slope': base_dir+'hydrosheds_slope_5641.tif',
            'x_dist_stream': base_dir+'hydroriver_Xdistance2streams.tif',
            'hand':base_dir+'hydrosheds_hand_nn_interp.tif',
            'pr': base_dir+'era5_meanP_v2.tif',
            'T2m': base_dir+'era5_avgT_celsius.tif',
            'PME': base_dir+'era5_PME_v2.tif',
            'sand': base_dir+'sand_2m_mean_Brazil.tif',
            'clay': base_dir+'clay_2m_mean_Brazil.tif',
            'log_hydraulic_cond': base_dir+'hydraulic_conductivity.tif',
        }
        
        # Create empty array for input variables
        n_features = len(in_vars)
        x_new = np.empty([n_features,loc_indexes.shape[0]],dtype=np.float32)

        ## START BY READING IN TOPOGRAPHIC VALUES ON LOCATION GRID ####
        # # location index
        in_loc_x = xarray.DataArray(loc_indexes[:, 1],dims='points')
        in_loc_y = xarray.DataArray(loc_indexes[:, 0],dims='points')


        # ELEVATION
        elev_dataset = rioxarray.open_rasterio(data_paths['elevation']).isel(x=in_loc_x,y=in_loc_y)
        elev_dataset = elev_dataset.astype('float32')
        elev_data = elev_dataset.data[0,:]
        elev_data[elev_data==32767] = np.nan #dealing w missing data
        
        x_new[0,:] = elev_data
        
        del elev_dataset

        # SLOPE
        slope_dataset = rioxarray.open_rasterio(data_paths['slope']).isel(x=in_loc_x,y=in_loc_y)
        slope_data = slope_dataset.data[0,:]
        slope_data[slope_data==-9999.0] = np.nan #dealing w missing data
        
        x_new[1,:] = slope_data
        
        del slope_dataset, slope_data

        # HORIZONTAL DISTANCE TO STREAM 
        Xdis2streams_dataset = rioxarray.open_rasterio(data_paths['x_dist_stream']).isel(x=in_loc_x,y=in_loc_y)
        Xdis2streams_data = Xdis2streams_dataset.data[0,:].astype('float32')
        # we are already masking all the nan elev pixels

        x_new[2,:] = Xdis2streams_data
        
        del elev_data, Xdis2streams_dataset, Xdis2streams_data

        #HEIGHT ABOVE NEAREST DRAINAGE
        hand_dataset = rioxarray.open_rasterio(data_paths['hand']).isel(x=in_loc_x,y=in_loc_y)
        hand_data = hand_dataset.data[0,:].astype('float32')
        # Missing data already defined as nan
        
        x_new[3,:] = hand_data
        
        del hand_dataset, hand_data

        # ERA5 climate data — load all three before applying neighbor fill (consistent with data prep)
        in_era5_x = xarray.DataArray(grid_cell_indexes_era5[0], dims='points')
        in_era5_y = xarray.DataArray(grid_cell_indexes_era5[1], dims='points')

        pr_dataset  = rioxarray.open_rasterio(data_paths['pr']).isel(x=in_era5_x, y=in_era5_y)
        pr_data     = pr_dataset.data[0,:].astype('float32')
        del pr_dataset

        t2m_dataset = rioxarray.open_rasterio(data_paths['T2m']).isel(x=in_era5_x, y=in_era5_y)
        t2m_data    = t2m_dataset.data[0,:].astype('float32')
        del t2m_dataset

        pme_dataset = rioxarray.open_rasterio(data_paths['PME']).isel(x=in_era5_x, y=in_era5_y)
        pme_data    = pme_dataset.data[0,:].astype('float32')
        del pme_dataset

        # Neighbor-fill for all three climate variables where PME == 0 — matches dataprep logic:
        # trigger on PME == 0; fill PME, pr, and T2m from neighbors (y+1, y-1, x-1);
        # only apply if at least one valid pr neighbor exists
        zero_mask = (pme_data == 0)
        if zero_mask.any():
            pr_full  = rioxarray.open_rasterio(data_paths['pr'])
            t2m_full = rioxarray.open_rasterio(data_paths['T2m'])
            pme_full = rioxarray.open_rasterio(data_paths['PME'])
            y_arr = np.array(grid_cell_indexes_era5[1])
            x_arr = np.array(grid_cell_indexes_era5[0])
            for idx in np.where(zero_mask)[0]:
                pme_neighbors = [float(pme_full.data[0, y_arr[idx]+1, x_arr[idx]]),
                                 float(pme_full.data[0, y_arr[idx]-1, x_arr[idx]]),
                                 float(pme_full.data[0, y_arr[idx],   x_arr[idx]-1])]
                pr_neighbors  = [float(pr_full.data[0,  y_arr[idx]+1, x_arr[idx]]),
                                 float(pr_full.data[0,  y_arr[idx]-1, x_arr[idx]]),
                                 float(pr_full.data[0,  y_arr[idx],   x_arr[idx]-1])]
                t2m_neighbors = [float(t2m_full.data[0, y_arr[idx]+1, x_arr[idx]]),
                                 float(t2m_full.data[0, y_arr[idx]-1, x_arr[idx]]),
                                 float(t2m_full.data[0, y_arr[idx],   x_arr[idx]-1])]
                pme_valid = [v for v in pme_neighbors if v != 0]
                pr_valid  = [v for v in pr_neighbors  if v != 0]
                t2m_valid = [v for v in t2m_neighbors if not np.isnan(v)]
                if pr_valid:
                    pme_data[idx] = np.float32(np.mean(pme_valid))
                    pr_data[idx]  = np.float32(np.mean(pr_valid))
                    t2m_data[idx] = np.float32(np.mean(t2m_valid))
            del pr_full, t2m_full, pme_full

        # Set remaining zeros to NaN (T2m uses NaN natively in the source raster)
        pr_data[pr_data == 0]   = np.nan
        pme_data[pme_data == 0] = np.nan

        x_new[4,:] = pr_data
        x_new[5,:] = t2m_data
        x_new[6,:] = pme_data

        del pr_data, t2m_data, pme_data
        
        logk_dataset = rioxarray.open_rasterio(data_paths['log_hydraulic_cond']).isel(x=in_loc_x,y=in_loc_y)
        logk_data = logk_dataset.data[0,:].astype('float32')
        logk_data[logk_data==logk_dataset._FillValue] = np.nan ####
        
        x_new[7,:] = logk_data
        
        del logk_dataset, logk_data    
  
        in_soilgrid_x = xarray.DataArray(grid_cell_indexes_soilgrid[0],dims='points')
        in_soilgrid_y = xarray.DataArray(grid_cell_indexes_soilgrid[1],dims='points')
        
        #Add mean sand and clay contents from SoilGrids 2.0
        sand_dataset = rioxarray.open_rasterio(data_paths['sand']).isel(x=in_soilgrid_x,y=in_soilgrid_y)
        sand_data = sand_dataset.data[0,:].astype('float32')
        sand_data[sand_data==sand_dataset._FillValue] = 0 #consistent with what we did in training
        
        x_new[8,:] = sand_data
        
        del sand_dataset, sand_data

        clay_dataset = rioxarray.open_rasterio(data_paths['clay']).isel(x=in_soilgrid_x,y=in_soilgrid_y)
        clay_data = clay_dataset.data[0,:].astype('float32')
        clay_data[clay_data==clay_dataset._FillValue] = 0 #consistent with what we did in training
        
        x_new[9,:] = clay_data
        
        del clay_dataset, clay_data
        
        self.in_data = np.stack(x_new).astype(np.float32).reshape(n_features, -1)
        
    def __getitem__(self, idx):
        return (self.in_data[:,idx])

    def __len__(self):
        return self.in_data.shape[1]

def WTD_mean_estimation(X_section):
    
    all_results = []
    
    for n in range(n_estimators):
        all_results.append(estimators[n].predict(X_section))
        
    all_results = np.array(all_results)

    return (np.median(all_results,axis=0),
            np.quantile(all_results,0.25,axis=0),
            np.quantile(all_results,0.75,axis=0))

def piece(x_indexes_section,y_indexes_section,x_indexes_era5_section,y_indexes_era5_section,x_indexes_soilgrid_section,y_indexes_soilgrid_section,loc_indexes_section):
    num_loops = 40
    
    for i in range(num_loops):
        
        if i == num_loops - 1:
            x_indexes_section_small = x_indexes_section[x_indexes_section.shape[0]//num_loops*i:]
            y_indexes_section_small = y_indexes_section[y_indexes_section.shape[0]//num_loops*i:]
            x_indexes_era5_section_small = x_indexes_era5_section[x_indexes_era5_section.shape[0]//num_loops*i:]
            y_indexes_era5_section_small = y_indexes_era5_section[y_indexes_era5_section.shape[0]//num_loops*i:]
            x_indexes_soilgrid_section_small = x_indexes_soilgrid_section[x_indexes_soilgrid_section.shape[0]//num_loops*i:]
            y_indexes_soilgrid_section_small = y_indexes_soilgrid_section[y_indexes_soilgrid_section.shape[0]//num_loops*i:]
            loc_indexes_section_small = loc_indexes_section[loc_indexes_section.shape[0]//num_loops*i:,:]
        else:
            x_indexes_section_small = x_indexes_section[x_indexes_section.shape[0]//num_loops*i:x_indexes_section.shape[0]//num_loops*(i+1)]
            y_indexes_section_small = y_indexes_section[y_indexes_section.shape[0]//num_loops*i:y_indexes_section.shape[0]//num_loops*(i+1)]
            x_indexes_era5_section_small = x_indexes_era5_section[x_indexes_era5_section.shape[0]//num_loops*i:x_indexes_era5_section.shape[0]//num_loops*(i+1)]
            y_indexes_era5_section_small = y_indexes_era5_section[y_indexes_era5_section.shape[0]//num_loops*i:y_indexes_era5_section.shape[0]//num_loops*(i+1)]
            x_indexes_soilgrid_section_small = x_indexes_soilgrid_section[x_indexes_soilgrid_section.shape[0]//num_loops*i:x_indexes_soilgrid_section.shape[0]//num_loops*(i+1)]
            y_indexes_soilgrid_section_small = y_indexes_soilgrid_section[y_indexes_soilgrid_section.shape[0]//num_loops*i:y_indexes_soilgrid_section.shape[0]//num_loops*(i+1)]
            loc_indexes_section_small = loc_indexes_section[loc_indexes_section.shape[0]//num_loops*i:loc_indexes_section.shape[0]//num_loops*(i+1),:]

        x_indexes_section_small = x_indexes_section_small.astype(np.float32)
        y_indexes_section_small = y_indexes_section_small.astype(np.float32)
        x_indexes_era5_section_small = x_indexes_era5_section_small.astype(np.float32)
        y_indexes_era5_section_small = y_indexes_era5_section_small.astype(np.float32)
        x_indexes_soilgrid_section_small = x_indexes_soilgrid_section_small.astype(np.float32)
        y_indexes_soilgrid_section_small = y_indexes_soilgrid_section_small.astype(np.float32)
        loc_indexes_section_small = loc_indexes_section_small.astype(np.float32)

        #Load dataset
        dataset = all_WTD_Mean_Estimation_Data(in_vars,
                                                         grid_cell_indexes=[x_indexes_section_small.astype(int),y_indexes_section_small.astype(int)],
                                                         grid_cell_indexes_era5=[x_indexes_era5_section_small.astype(int),y_indexes_era5_section_small.astype(int)],
                                                         grid_cell_indexes_soilgrid=[x_indexes_soilgrid_section_small.astype(int),y_indexes_soilgrid_section_small.astype(int)],
                                                         loc_indexes=loc_indexes_section_small.astype(int))
        X_dataset = dataset[:]
        X_section = np.swapaxes(X_dataset, 0, 1) #input shape (n_samples, n_features)

        (median, arr_25, arr_75) = WTD_mean_estimation(X_section)
        
        if i == 0:
            median_all = median
            arr_25_all = arr_25
            arr_75_all = arr_75

        else:
            median_all = np.concatenate((median_all,median))
            arr_25_all = np.concatenate((arr_25_all,arr_25))
            arr_75_all = np.concatenate((arr_75_all,arr_75))
        
    return (median_all, arr_25_all, arr_75_all)

if __name__ == '__main__':

    
    #Load masked x and y indexes
    path=''
    x_indexes = np.load(path+'x_indexes_masked_90m.npy')
    y_indexes = np.load(path+'y_indexes_masked_90m.npy') # these are MASKED for only locations within Brazil from elevation raster (see masked_grid_90m.py)
    
    x_indexes_era5 = np.load(path+'x_indexes_era5_90m.npy')
    y_indexes_era5 = np.load(path+'y_indexes_era5_90m.npy')

    x_indexes_proj = np.load(path+'x_indexes_climate_90m.npy')
    y_indexes_proj = np.load(path+'y_indexes_climate_90m.npy')

    x_indexes_soilgrid = np.load(path+'x_indexes_soilgrids_v2_90m.npy')
    y_indexes_soilgrid = np.load(path+'y_indexes_soilgrids_v2_90m.npy')

    nx = x_indexes.shape[1]
    ny = x_indexes.shape[0]

    # Create mask based on nan values on lookup tables
    mask = np.isnan(x_indexes) + np.isnan(y_indexes) + np.isnan(x_indexes_soilgrid) + np.isnan(y_indexes_soilgrid) + np.isnan(x_indexes_proj) + np.isnan(y_indexes_proj)
    mask[mask>=1] = 1

    loc_indexes = np.argwhere(~mask)

    x_indexes = x_indexes[~mask]
    y_indexes = y_indexes[~mask]

    x_indexes_era5 = x_indexes_era5[~mask]
    y_indexes_era5 = y_indexes_era5[~mask]

    x_indexes_proj = x_indexes_proj[~mask]
    y_indexes_proj = y_indexes_proj[~mask]

    x_indexes_soilgrid = x_indexes_soilgrid[~mask]
    y_indexes_soilgrid = y_indexes_soilgrid[~mask]

    del mask


    total_sections = int(os.getenv('SLURM_CPUS_PER_TASK'))
    
    #Initialization
    all_results_arr     = np.empty(loc_indexes.shape[0], dtype='float32') * np.nan
    all_results_arr_25  = np.empty(loc_indexes.shape[0], dtype='float32') * np.nan
    all_results_arr_75  = np.empty(loc_indexes.shape[0], dtype='float32') * np.nan

    #Multi-thread
    future_map = {}  # mapping from Future objects to the timesteps that they will populate

    with ProcessPoolExecutor(total_sections) as executor:

        for section in range(total_sections):

            if section == total_sections - 1:
                x_indexes_section        = x_indexes[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                y_indexes_section        = y_indexes[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                x_indexes_era5_section   = x_indexes_era5[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                y_indexes_era5_section   = y_indexes_era5[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                x_indexes_soilgrid_section = x_indexes_soilgrid[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                y_indexes_soilgrid_section = y_indexes_soilgrid[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):]
                loc_indexes_section      = loc_indexes[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):,:]
            else:
                x_indexes_section        = x_indexes[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                y_indexes_section        = y_indexes[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                x_indexes_era5_section   = x_indexes_era5[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                y_indexes_era5_section   = y_indexes_era5[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                x_indexes_soilgrid_section = x_indexes_soilgrid[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                y_indexes_soilgrid_section = y_indexes_soilgrid[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]
                loc_indexes_section      = loc_indexes[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)]

            future = executor.submit(piece, x_indexes_section, y_indexes_section,
                                     x_indexes_era5_section, y_indexes_era5_section,
                                     x_indexes_soilgrid_section, y_indexes_soilgrid_section,
                                     loc_indexes_section)
            future_map[future] = section

        del x_indexes, y_indexes, x_indexes_era5, y_indexes_era5, x_indexes_soilgrid, y_indexes_soilgrid
        
        for future in tqdm(as_completed(future_map),total=total_sections):
            # This block is entered once for every completed Future
            section = future_map[future]

            if  section == total_sections - 1:
                all_results_arr[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):] = future.result()[0]
                all_results_arr_25[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):] = future.result()[1]
                all_results_arr_75[int(loc_indexes.shape[0]/total_sections)*(total_sections-1):] = future.result()[2]

            else:
                all_results_arr[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)] = future.result()[0]
                all_results_arr_25[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)] = future.result()[1]
                all_results_arr_75[int(loc_indexes.shape[0]/total_sections)*section:int(loc_indexes.shape[0]/total_sections)*(section+1)] = future.result()[2]
            
    all_data_map = np.empty([ny,nx])*np.nan
    all_data_map[loc_indexes[:,0],loc_indexes[:,1]] = all_results_arr

    #Save data 
    np.save(out_dir+'wtd_mean_estimate_RF_median.npy',all_data_map) # we use this as the final product
    
    del all_results_arr, all_data_map

    all_data_map_25 = np.empty([ny,nx])*np.nan
    all_data_map_25[loc_indexes[:,0],loc_indexes[:,1]] = all_results_arr_25

    all_data_map_75 = np.empty([ny,nx])*np.nan
    all_data_map_75[loc_indexes[:,0],loc_indexes[:,1]] = all_results_arr_75

    all_data_map_iqr = all_data_map_75 - all_data_map_25

    np.save(out_dir+'wtd_mean_estimate_RF_Brasil_IQR.npy',all_data_map_iqr)
    np.save(out_dir+'wtd_mean_estimate_RF_Brasil_25th.npy',all_data_map_25)
    np.save(out_dir+'wtd_mean_estimate_RF_Brasil_75th.npy',all_data_map_75)
    
    del all_results_arr_25, all_results_arr_75, all_data_map_25, all_data_map_75, all_data_map_iqr