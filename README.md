# Multispectral Landcover Classification
This project is to demonstrate how tensorflow can be used to model and make predictions about multispectral data. Data are gathered from the [LandCoverNet](https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/) dataset. This project is NOT complete. It has been setup hastily to demonstrate usage of machine learning algorithms, python, and data visualization. The results are not *currently* very good...at all

Improvement areas are listed within this document.

## Goal
Model the multispectral data such that given an unknown feature vector, the type of land cover can be determined. The dataset includes: 
- (Semi) Natural Vegetation
- Artificial Bareground
- Cultivated Vegetation
- Natural Bareground
- No Data
- Permanent Snow/Ice
- Water
- Woody Vegetation)

Data from the sensor ([Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)) is available in 13 separate bands in varying spatial resolutions. Currently, only the bands at a 10m resolution are used - Bands 2, 3, 4, and 8. ([Bands Description](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial))


## Dataset Considerations
As with most datasets, there are data to weed out or fish through. 

#### Clouds
The LandCoverNet dataset contains images with clouds. Currently, any pixels with a cloud probability over 50% are ignored from both testing and training. It could be interesting to dig a little deeper into this.

#### Acquisition
Acquisition of the data was performed by using documentation and the jupyter notebook [here](https://github.com/radiantearth/mlhub-tutorials/blob/main/notebooks/radiant-mlhub-landcovernet.ipynb). Data was acquired by requesting 1 tile/chip from the data set with each classification category. There is some overlap.
## Process
### Data Gathering
Data was gathered using the [LandCover.ipynb](LandCover.ipynb) file. Ultimately, 5 tiles were chosen for training and 1 was chosen for testing.

### Preprocessing
Pixels with a cloud probability > 50% are filtered out. For improvements on the data, additional bands could be used if they are converted to the same 10m resolution of the visible-spectrum bands.


### Training
Bands 2,3,4, and 8 were used as they have the same resolution (without preprocessing). A basic RandomForestModel was used from the keras library. All default parameters are currently used. Improvements could be made by hypertuning the model/changing the defaults. This step is currently done in the scratch.ipynb file.

### Evaluation
The model was evaluated on 1 random tile from the dataset.

## Roadmap
This is where this project could potentially go.


[x] Code infrastructure for building and evaluating a model
[] Model visualization - add plots and images
[x] Code up more specific dataset retrieval functions
[x] Handle pixels that have a high probability of cloud cover
[] Hypertuning to find optimal settings for the model
[] Investigate possibility to predict how the terrain changes over time
