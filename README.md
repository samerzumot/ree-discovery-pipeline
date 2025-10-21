# Rare Earth Element Discovery System

A comprehensive machine learning pipeline for identifying potential Rare Earth Element (REE) deposits in California using Google Earth Engine and USGS data.

## 🎯 Objective

This system creates an end-to-end pipeline that:
1. Downloads and filters real USGS REE occurrence data for California
2. Extracts real environmental and spectral features from Google Earth Engine
3. Trains a machine learning model (Random Forest) on those features
4. Generates REE potential predictions for new areas in California
5. Visualizes results in an interactive Streamlit + Folium web app

## 🏗️ Project Structure

```
/ree_discovery_real/
├── data/
│   ├── usgs_ree_occurrences_ca.csv
│   ├── gee_features_california.csv
│   └── ree_predictions_california.geojson
├── src/
│   ├── gee_feature_extractor.py
│   ├── model_training.py
│   ├── predict_new_areas.py
│   └── visualization_app.py
├── models/
│   ├── model_ree_california.pkl
│   ├── model_evaluation_curves.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd geo-graph-ai

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine
earthengine authenticate
```

### 2. Run the Pipeline

```bash
# Step 1: Extract features from Google Earth Engine
python src/gee_feature_extractor.py

# Step 2: Train the machine learning model
python src/model_training.py

# Step 3: Generate predictions for new areas
python src/predict_new_areas.py

# Step 4: Launch the visualization app
streamlit run src/visualization_app.py
```

## 📊 Data Sources

### USGS REE Occurrences
- **Source**: USGS Mineral Resources Data System
- **URL**: https://mrdata.usgs.gov/ree/ree-csv.zip
- **Filter**: California bounding box (32.5°N to 42.0°N, -124.5°W to -114.0°W)

### Google Earth Engine Datasets

| Feature | GEE ID | Details |
|---------|--------|---------|
| Elevation | `USGS/SRTMGL1_003` | SRTM elevation data |
| Terrain | `ee.Terrain.products` | Slope and aspect derived |
| Landcover | `ESA/WorldCover/v100` | Global land cover classification |
| Optical | `COPERNICUS/S2_SR_HARMONIZED` | Sentinel-2 surface reflectance |
| NDVI | Calculated | (B8 - B4) / (B8 + B4) |
| NDCI | Calculated | (B11 - B8) / (B11 + B8) |
| IronRatio | Calculated | B4 / B2 |

## 🔬 Features Extracted

For each point (REE occurrence or background), the system extracts:

- **Elevation**: Mean and standard deviation
- **Slope**: Mean and standard deviation  
- **NDVI**: Mean and standard deviation
- **NDCI**: Mean and standard deviation
- **Iron Ratio**: Mean and standard deviation
- **Landcover**: Mode classification

## 🤖 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - `n_estimators=200`
  - `max_depth=8`
  - `min_samples_leaf=2`
  - `class_weight="balanced"`
- **Validation**: 80/20 train/test split
- **Metrics**: ROC-AUC, F1-Score, Precision, Recall

## 🗺️ Visualization Features

The Streamlit app provides:

- **Interactive Map**: Folium-based map with multiple tile layers
- **Layer Controls**: Toggle REE occurrences, heatmap, and predictions
- **Heatmap**: Color-coded REE potential visualization
- **Feature Analysis**: Statistical plots and distributions
- **Prediction Analysis**: Probability distributions and correlations
- **Model Performance**: ROC curves and feature importance

## 📈 Expected Results

### Validation Targets
- **Mountain Pass Mine**: Should show high REE potential scores
- **ROC-AUC > 0.7**: Indicates useful predictive signal
- **Geological Plausibility**: Results should highlight known REE belts:
  - Sierra Nevada foothills
  - Mojave Desert region
  - Peninsular Ranges

### Performance Metrics
- **Training Time**: ~5-10 minutes for feature extraction
- **Prediction Time**: ~2-5 minutes for full California grid
- **Model Accuracy**: Target ROC-AUC > 0.7

## 🔧 Technical Requirements

### Dependencies
- Python 3.8+
- Google Earth Engine API access
- Internet connection for data downloads

### System Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for data and models
- **Processing**: Multi-core CPU recommended

## 🚨 Important Notes

### Authentication
- Google Earth Engine authentication required
- Run `earthengine authenticate` before first use
- Ensure you have GEE access permissions

### Rate Limiting
- GEE API has rate limits
- Scripts include built-in delays
- Large extractions may take time

### Data Quality
- All data is real (no simulations)
- Features extracted from actual satellite imagery
- Model trained on real geological occurrences

## 🐛 Troubleshooting

### Common Issues

1. **GEE Authentication Error**
   ```bash
   earthengine authenticate
   ```

2. **Missing Data Files**
   - Ensure you run scripts in sequence
   - Check internet connection for downloads

3. **Memory Issues**
   - Reduce grid resolution in prediction script
   - Process data in smaller batches

4. **Rate Limiting**
   - Increase delays in feature extraction
   - Process fewer points at once

## 📚 References

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [USGS Mineral Resources Data System](https://mrdata.usgs.gov/)
- [Sentinel-2 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Earth Engine team for satellite data access
- USGS for mineral occurrence data
- Streamlit and Folium communities for visualization tools
