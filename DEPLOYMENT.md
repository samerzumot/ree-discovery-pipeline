# REE Discovery Pipeline - Deployment Guide

## 🚀 Streamlit Cloud Deployment

This project is ready for deployment on Streamlit Cloud.

### 📋 Prerequisites

1. **GitHub Account**: Required for Streamlit Cloud
2. **Google Earth Engine Account**: For real-time data access
3. **Streamlit Cloud Account**: Free tier available

### 🔧 Deployment Steps

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Deploy!

### 🌐 Live App Features

- **Interactive REE potential map**
- **Real-time predictions** for California locations
- **Environmental feature analysis**
- **Machine learning model integration**

### 📊 Data Sources

- **USGS REE occurrences**: Real mineral deposit data
- **Google Earth Engine**: Environmental and spectral features
- **Machine Learning**: Trained Random Forest model

### 🔑 Environment Variables

No special environment variables required for basic deployment.

### 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit entry point
├── src/
│   ├── visualization_app.py  # Core visualization logic
│   ├── gee_feature_extractor.py
│   ├── model_training.py
│   └── predict_new_areas.py
├── data/                     # Sample data files
├── models/                   # Trained ML models
├── requirements.txt          # Python dependencies
└── .streamlit/config.toml   # Streamlit configuration
```

### 🎯 Access Your App

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## 📞 Support

For issues or questions, please check the GitHub repository or contact the development team.
