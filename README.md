# High Resolution Static Water Table Depth Estimation over Brazil

This repo includes the workflow of [Fleury et al., 2026](https://iopscience.iop.org/article/10.1088/1748-9326/ae9bd7) to create Brazil's high resolution map of (static) water table depth (WTD). The scripts were developed to train a random forest model for estimating the long-term mean WTD across Brazil using available observations, and to generate a long-term mean WTD map at a ~90 m resolution for Brazil.<br>
<br>
The workflow was adapted from the code developed for the contiguous United States by [Ma et al., 2026](https://doi.org/10.1038/s43247-025-03094-3).<br>
<br>
The final dataset is accessible via the [HydroData repository](https://hydroframe.org/hydrodata).<br>
<br>
For questions, please contact Maria Fleury (mfleury@princeton.edu).<br>

## Model Training & Deployment
The following scripts used for model training and deployment can be found in the main folder:
1. `1_RF_model_training.py`: The script used to train the random forest model for estimating long-term mean WTD over Brazil, evaluate it on a held-out test set, and compute permutation feature importance. It reads a prepared csv of well observations and input data. Links to the WTD and input datasets are provided below.
2. `2_WTD_prediction.py`: The script used for generating the high resolution WTD map along with its associated uncertainty maps (25th percentile, 75th percentile, and interquartile range across the trees of the forest). Input datasets are too large to share. One can adapt this script to their own datasets and domains.
3. `2_WTD_prediction.slurm`: The Slurm job script used to run `2_WTD_prediction.py`. The number of parallel sections is read from the `SLURM_CPUS_PER_TASK` environment variable.
4. `3_exporting_tif.py`: The script used for converting the resulting npy files from `2_WTD_prediction.py` to GeoTIFFs (EPSG:5641, SIRGAS 2000 / Brazil Polyconic).
5. `4_shap_analysis.py`: The script used to compute SHAP values for the trained model over the test dataset, to interpret the contribution of each predictor.


## Input Data Sources

The target variable is the mean water table depth (`mwtd`). This is comprised of measurements from the Groundwater Well Database for Brazil ([Uchôa et al., 2025](https://www.nature.com/articles/s41597-025-05843-7)), which recently standardized and quality assured all the WTD data available through the Geological Survey of Brazil (SGB). As well as long-term means calculated using data from SGB's Integrated Groundwater Monitoring Network Project ([RIMAS](https://rimasweb.sgb.gov.br/layout/))

In addition to the well data, WTD is assumed to be 0 at 6,000 random locations along the river network (HydroRIVERS); these points are added to the training set only to improve the detection of saturated conditions near water bodies.

The model uses ten predictors of long-term mean WTD:

| Input variable | Name in code | Description | Data source |
| --- | --- | --- | --- |
| Average precipitation | `pr` | Monthly total precipitation from 1950-2025, averaged per year | ERA5-Land ([Muñoz-Sabater et al., 2021](https://doi.org/10.5194/essd-13-4349-2021)) |
| Average temperature | `T2m` | Monthly average temperature from 1950-2025, averaged per year | ERA5-Land ([Muñoz-Sabater et al., 2021](https://doi.org/10.5194/essd-13-4349-2021)) |
| Average precipitation minus evapotranspiration (PME) | `PME` | Monthly total precipitation minus total evapotranspiration from 1950-2025, averaged per year | ERA5-Land ([Muñoz-Sabater et al., 2021](https://doi.org/10.5194/essd-13-4349-2021)) |
| Elevation | `elevation` | Conditioned digital elevation model (DEM) | HydroSHEDS ([Lehner et al., 2008](https://doi.org/10.1029/2008EO100001)) |
| Slope | `slope` | Slope, derived from elevation | HydroSHEDS ([Lehner et al., 2008](https://doi.org/10.1029/2008EO100001)) |
| Horizontal distance to stream | `x_dist_stream` | Horizontal distance to the nearest river pixel | HydroRIVERS ([Lehner & Grill, 2013](https://doi.org/10.1002/hyp.9740)) |
| Height above nearest drainage (HAND) | `hand` | Elevation difference to the nearest river pixel | HydroSHEDS + HydroRIVERS ([Lehner et al., 2008](https://doi.org/10.1029/2008EO100001); [Lehner & Grill, 2013](https://doi.org/10.1002/hyp.9740)) |
| Log of hydraulic conductivity (log(K)) | `log_hydraulic_cond` | Converted from near surface global permeability values | GLHYMPS 2.0 ([Gleeson et al., 2014](https://doi.org/10.1002/2014GL059856); [Huscroft et al., 2018](https://doi.org/10.1002/2017GL075860)) |
| Sand content | `sand` | Sand content averaged over the 200 cm below the land surface | SoilGrids 2.0 ([Poggio et al., 2021](https://doi.org/10.5194/soil-7-217-2021)) |
| Clay content | `clay` | Clay content averaged over the 200 cm below the land surface | SoilGrids 2.0 ([Poggio et al., 2021](https://doi.org/10.5194/soil-7-217-2021)) |


## About
For more information, please refer to the paper: [https://iopscience.iop.org/article/10.1088/1748-9326/ae9bd7](https://iopscience.iop.org/article/10.1088/1748-9326/ae9bd7)<br>

## Funding

This research has been supported by the U.S. National Science Foundation Convergence Accelerator
Program (grant no. CA-2040542)
