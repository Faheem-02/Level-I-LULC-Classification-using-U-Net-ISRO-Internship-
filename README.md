# Level-I Land Use/Land Cover Classification using U-Net

![Python](https://img.shields.io/badge/Python-3.8%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-blue)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![ISRO](https://img.shields.io/badge/Project-ISRO%20Internship-purple)

A deep learning-based semantic segmentation project for automated Land Use and Land Cover (LULC) classification from high-resolution satellite imagery, developed during my internship at the National Remote Sensing Centre (NRSC), Indian Space Research Organisation (ISRO).

##  Project Overview

This project implements a U-Net convolutional neural network for pixel-wise classification of satellite imagery into 7 distinct land cover categories. The model was trained on high-resolution satellite data from Punjab, India, to automate the creation of accurate LULC maps for applications in urban planning, environmental monitoring, and resource management.

**Key Features:**
- Semantic segmentation using U-Net architecture
- Processing of high-resolution satellite imagery (3313×2912 pixels)
- Classification into 7 LULC categories
- Custom data preprocessing and patchification pipeline
- Comprehensive model evaluation metrics

##  LULC Classification Categories

The model classifies each pixel into one of the following categories:
1. **Built-up Areas** - Human-made structures and urban developments
2. **Agriculture** - Crop fields, orchards, and cultivated lands
3. **Water Bodies** - Lakes, rivers, ponds, and reservoirs
4. **Wasteland** - Barren or unutilized land areas
5. **Forests** - Dense vegetation and forest covers
6. **Roads** - Transportation networks and highways
7. **Others** - Unclassified or ambiguous regions

## Technical Implementation

### Architecture
The project utilizes a U-Net architecture with:
- **Encoder Path**: Feature extraction through convolutional and pooling layers
- **Bottleneck**: Capturing high-level abstract features
- **Decoder Path**: Feature reconstruction with transposed convolutions
- **Skip Connections**: Preserving spatial information across layers

### Tech Stack
- **Framework**: TensorFlow 2.x, Keras
- **Image Processing**: OpenCV, Pillow, Patchify
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Geospatial Processing**: QGIS (for ground truth annotation)

## ⚠️ Important Note on Data and Results

**Confidentiality Notice**: 
Due to strict security protocols and non-disclosure agreements with ISRO:
- The original satellite imagery and ground truth data cannot be shared publicly
- Model training results and performance metrics are confidential
- Pre-trained weights and specific evaluation results are not available for distribution

This repository contains only the implementation code and architecture details for educational purposes.

### Installation
Clone the repository:
```bash
git clone https://github.com/Faheem-02/Level-I-LULC-Classification-using-U-Net-ISRO-Internship.git
cd Level-I-LULC-Classification-using-U-Net-ISRO-Internship

##  Learning Outcomes

Through this internship project at NRSC, ISRO, I gained comprehensive expertise in:

- **Satellite Image Processing**: Working with high-resolution geospatial data from Bhuvan Panchayat platform
- **Deep Learning Architecture**: Implementing and customizing U-Net for semantic segmentation tasks
- **Data Preprocessing**: Developing pipelines for large-scale image patchification and normalization
- **Geospatial Analysis**: Understanding LULC classification challenges and applications
- **Model Optimization**: Tuning hyperparameters and addressing class imbalance issues
- **TensorFlow/Keras**: Advanced implementation of complex neural networks
- **Remote Sensing Applications**: Practical experience in solving real-world environmental monitoring problems
- **Project Management**: Handling constraints of proprietary data and security protocols

## Acknowledgments

This project was successfully completed thanks to the guidance and support of:

- **Dr. S.S. Raja Shekhar** - Head of Applications and Scientist-SG, NRSC, ISRO
- **Dr. Jyothi B.** - Project Mentor and Guide, NRSC, ISRO  
- **Smt. Savitha Sunkari** & **Dr. Jaya Saxena** - Internship Coordinators, NRSC
- **Balanagar RRSC Team** - For their unwavering support and technical assistance
- **Indian Space Research Organisation (ISRO)** - For providing this incredible opportunity
- **NRSC (National Remote Sensing Centre)** - For the resources, data access, and workstations

Special gratitude to my parents for their constant encouragement and support throughout this journey.
