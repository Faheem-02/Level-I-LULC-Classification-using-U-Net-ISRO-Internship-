# Dataset Information

## 🔒 Original Dataset

**Note: The original dataset used in this project cannot be shared publicly due to:**

- **Proprietary Restrictions**: The satellite imagery was obtained from Bhuvan Panchayat (ISRO), which has strict usage and distribution policies.
- **National Security Considerations**: As per ISRO and Indian government regulations, high-resolution satellite data is classified as sensitive information.
- **Non-Disclosure Agreement**: This project was conducted under an internship agreement that prohibits sharing the original data.

## 📊 Dataset Specifications

The original dataset consisted of:
- **135 high-resolution satellite images** (3313×2912 pixels) of Punjab, India
- **7 LULC classes**: Built-up Areas, Agriculture, Water Bodies, Wasteland, Forests, Roads, and Others
- **Corresponding segmentation masks** manually annotated using QGIS
- **Data format**: GeoTIFF for images, PNG for masks

## 🌐 Alternative Public Datasets

For replication and further research, here are publicly available datasets suitable for similar LULC classification tasks:

### Suggested alternative dataset to work with - LandCover.ai
- **Description**: High-resolution aerial imagery with land cover annotations, focused on Poland
- **Resolution**: 25cm or 50cm per pixel
- **Classes**: Building, Woodland, Water, Road
- **Link**: [LandCover.ai Dataset](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai)

### 1. EuroSAT
- **Description**: Sentinel-2 satellite images covering 13 spectral bands and 10 classes
- **Resolution**: 64×64 pixels, 10-20m resolution
- **Classes**: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, Sea/Lake
- **Link**: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

### 2. DeepGlobe Land Cover Classification Challenge
- **Description**: High-resolution satellite imagery with pixel-level annotations
- **Resolution**: 2448×2448 pixels, 50cm resolution
- **Classes**: Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown
- **Link**: [DeepGlobe Challenge](http://deepglobe.org/challenge.html)

### 3. UC Merced Land Use Dataset
- **Description**: Aerial orthoimagery with 21 land use classes
- **Resolution**: 256×256 pixels, 30cm resolution
- **Classes**: Agricultural, Airplane, Baseball diamond, Beach, Buildings, Chaparral, Dense residential, Forest, Freeway, Golf course, Harbor, Intersection, Medium residential, Mobile home park, Overpass, Parking lot, River, Runway, Sparse residential, Storage tanks, Tennis courts
- **Link**: [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## 🛠️ Data Preparation
Suggested Data foramt - .tiff (This format is good at dealing with the satellite data )
**Organize your data** in the following structure:
your_dataset/
 images/
 masks/

