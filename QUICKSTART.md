# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+
- Google Earth Engine account
- Internet connection

## 1. Setup (5 minutes)

```bash
# Clone and navigate to project
cd geo-graph-ai

# Run setup script
python setup.py

# Authenticate with Google Earth Engine
earthengine authenticate
```

## 2. Run Complete Pipeline (15-30 minutes)

```bash
# Run the entire pipeline
python run_pipeline.py
```

This will:
- Download real USGS REE data for California
- Extract features from Google Earth Engine
- Train a Random Forest model
- Generate predictions across California
- Validate results against known sites

## 3. Launch Visualization App

```bash
# Start the interactive dashboard
streamlit run src/visualization_app.py
```

Open your browser to `http://localhost:8501`

## 4. Expected Results

### Known REE Sites Should Show High Scores:
- **Mountain Pass Mine**: Should have high REE probability
- **Searles Lake**: Should show elevated potential
- **Other known sites**: Validated against real occurrences

### Geological Regions:
- **Sierra Nevada**: Granitic terranes with REE potential
- **Mojave Desert**: Known REE mining regions
- **Peninsular Ranges**: Metamorphic belts

## 5. Troubleshooting

### GEE Authentication Issues:
```bash
earthengine authenticate
# Follow the browser authentication
```

### Missing Dependencies:
```bash
pip install -r requirements.txt
```

### Memory Issues:
- Reduce grid resolution in `predict_new_areas.py`
- Process smaller batches

## 6. File Structure After Running

```
data/
├── usgs_ree_occurrences_ca.csv      # Real USGS REE data
├── gee_features_california.csv       # Extracted features
└── ree_predictions_california.geojson # Predictions

models/
├── model_ree_california.pkl          # Trained model
├── model_evaluation_curves.png       # ROC/PR curves
├── feature_importance.png            # Feature importance
└── validation_analysis.png           # Validation plots
```

## 7. Validation Checklist

- [ ] Mountain Pass Mine shows high probability (>0.5)
- [ ] ROC-AUC > 0.7 (indicates good model performance)
- [ ] High potential areas align with known geological regions
- [ ] Visualization app loads and displays map correctly

## 8. Next Steps

- Explore different prediction thresholds
- Analyze feature importance
- Investigate high-potential areas
- Export results for further analysis

## 🆘 Need Help?

Check the main README.md for detailed documentation and troubleshooting.
