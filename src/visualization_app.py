#!/usr/bin/env python3
"""
REE Discovery Visualization App
Interactive Streamlit + Folium app for exploring REE predictions and features.
"""

import streamlit as st
import folium
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

class REEVisualizationApp:
    def __init__(self):
        """Initialize the visualization app."""
        self.ree_occurrences_path = 'data/usgs_ree_occurrences_ca.csv'
        self.predictions_path = 'data/ree_predictions_california.geojson'
        self.features_path = 'data/gee_features_california.csv'
        
        # California center coordinates
        self.california_center = [36.7783, -119.4179]
        
    def load_data(self):
        """Load all required datasets."""
        # Load REE occurrences
        if os.path.exists(self.ree_occurrences_path):
            self.ree_occurrences = pd.read_csv(self.ree_occurrences_path)
        else:
            self.ree_occurrences = pd.DataFrame()
        
        # Load predictions (prefer our test data for clear visualization)
        test_path = 'data/test_predictions.csv'
        demo_path = 'data/demo_predictions.csv'
        
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            # Convert to GeoDataFrame
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            self.predictions = gpd.GeoDataFrame(df, geometry=geometry)
            print(f"✅ Loaded test data: {len(self.predictions)} points")
        elif os.path.exists(demo_path):
            df = pd.read_csv(demo_path)
            # Convert to GeoDataFrame
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            self.predictions = gpd.GeoDataFrame(df, geometry=geometry)
            print(f"✅ Loaded demo data: {len(self.predictions)} points")
        else:
            self.predictions = gpd.GeoDataFrame()
            print("⚠️ No test data found")
        
        # Load features
        if os.path.exists(self.features_path):
            self.features = pd.read_csv(self.features_path)
        else:
            self.features = pd.DataFrame()
    
    def create_base_map(self):
        """Create the base Folium map."""
        m = folium.Map(
            location=self.california_center,
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Topographic',
            overlay=False,
            control=True
        ).add_to(m)
        
        return m
    
    def add_ree_occurrences(self, m, ree_df):
        """Add known REE occurrence points to the map."""
        if len(ree_df) == 0:
            return m
        
        # Create feature group for known REE occurrences
        ree_group = folium.FeatureGroup(name="✅ Known REE Sites", show=True)
        
        for idx, row in ree_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                popup=f"<b>✅ KNOWN REE SITE</b><br><b>{row.get('name', 'REE Occurrence')}</b><br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}<br>Status: Verified Deposit",
                color='darkred',
                fillColor='red',
                fillOpacity=0.9,
                weight=3
            ).add_to(ree_group)
        
        ree_group.add_to(m)
        return m
    
    def add_prediction_heatmap(self, m, predictions_gdf, opacity=0.6):
        """Add REE prediction heatmap to the map."""
        if len(predictions_gdf) == 0:
            return m
        
        # Create heatmap data
        heat_data = []
        for idx, row in predictions_gdf.iterrows():
            if row['ree_probability'] > 0.1:  # Only show points with some potential
                heat_data.append([row.geometry.y, row.geometry.x, row['ree_probability']])
        
        if heat_data:
            from folium.plugins import HeatMap
            HeatMap(
                heat_data,
                name="REE Potential Heatmap",
                min_opacity=0.2,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={0.0: 'blue', 0.3: 'cyan', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'}
            ).add_to(m)
        
        return m
    
    def add_prediction_points(self, m, predictions_gdf, threshold=0.5, show_training=True):
        """Add ML-predicted REE potential points to the map."""
        if len(predictions_gdf) == 0:
            return m
        
        # Load training data to check for known sites
        training_data = self.load_training_data()
        
        # Filter high-potential points
        high_potential = predictions_gdf[predictions_gdf['ree_probability'] >= threshold]
        
        if len(high_potential) == 0:
            return m
        
        # Create feature groups for different types of points
        known_group = folium.FeatureGroup(name=f"✅ Training Sites (>{threshold})", show=True)
        pred_group = folium.FeatureGroup(name=f"🔮 New Predictions (>{threshold})", show=True)
        
        for idx, row in high_potential.iterrows():
            # Check if this point matches training data coordinates
            is_training_site = self.is_training_site(row.geometry.x, row.geometry.y, training_data)
            
            # Debug output
            print(f"Point {idx}: ({row.geometry.x:.3f}, {row.geometry.y:.3f}) - Training site: {is_training_site}")
            
            if is_training_site and show_training:
                # This is a training site - mark as known
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    popup=f"<b>✅ TRAINING SITE</b><br>Probability: {row['ree_probability']:.3f}<br>Lat: {row.geometry.y:.4f}<br>Lon: {row.geometry.x:.4f}<br>Status: Known REE Site<br><i>Used for model training</i>",
                    color='darkred',
                    fillColor='orange',
                    fillOpacity=0.9,
                    weight=3
                ).add_to(known_group)
            elif not is_training_site:
                # This is a new prediction - mark as ML prediction
                if row['ree_probability'] >= 0.8:
                    color = 'purple'
                    status = '🔥 Very High Potential'
                elif row['ree_probability'] >= 0.6:
                    color = 'blue'
                    status = '⚠️ High Potential'
                else:
                    color = 'green'
                    status = '🔍 Moderate Potential'
                
                size = 6 if row['ree_probability'] >= 0.8 else 5
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=size,
                    popup=f"<b>🔮 NEW PREDICTION</b><br>Probability: {row['ree_probability']:.3f}<br>Lat: {row.geometry.y:.4f}<br>Lon: {row.geometry.x:.4f}<br>Status: {status}<br><i>AI-generated prediction</i>",
                    color='white',
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(pred_group)
        
        if show_training:
            known_group.add_to(m)
        pred_group.add_to(m)
        return m
    
    def load_training_data(self):
        """Load training data to check for known sites."""
        try:
            if os.path.exists('data/gee_features_california.csv'):
                training_df = pd.read_csv('data/gee_features_california.csv')
                return training_df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def is_training_site(self, lon, lat, training_data, tolerance=0.1):
        """Check if coordinates match training data within tolerance."""
        if len(training_data) == 0:
            return False
        
        # Check if coordinates are within tolerance of any training site
        for _, row in training_data.iterrows():
            if (abs(row['lon'] - lon) <= tolerance and 
                abs(row['lat'] - lat) <= tolerance):
                return True
        return False
    
    def create_feature_analysis_plots(self, features_df):
        """Create feature analysis plots."""
        if len(features_df) == 0:
            return None
        
        # Feature distribution plots
        numeric_features = [
            'elevation_mean', 'slope_mean', 'NDVI_mean', 
            'NDCI_mean', 'IronRatio_mean'
        ]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=numeric_features,
            specs=[[{"secondary_y": False}] * 3, [{"secondary_y": False}] * 3]
        )
        
        for i, feature in enumerate(numeric_features):
            if feature in features_df.columns:
                row = i // 3 + 1
                col = i % 3 + 1
                
                # Create histogram
                fig.add_trace(
                    go.Histogram(
                        x=features_df[feature],
                        name=feature,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Feature Distributions",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_prediction_analysis_plots(self, predictions_gdf):
        """Create prediction analysis plots."""
        if len(predictions_gdf) == 0:
            return None
        
        # Probability distribution
        fig1 = px.histogram(
            predictions_gdf, 
            x='ree_probability',
            title='REE Probability Distribution',
            labels={'ree_probability': 'REE Probability', 'count': 'Number of Points'}
        )
        
        # Probability vs features
        feature_cols = ['elevation_mean', 'slope_mean', 'NDVI_mean', 'NDCI_mean', 'IronRatio_mean']
        available_features = [col for col in feature_cols if col in predictions_gdf.columns]
        
        if available_features:
            fig2 = make_subplots(
                rows=2, cols=3,
                subplot_titles=available_features[:6]
            )
            
            for i, feature in enumerate(available_features[:6]):
                row = i // 3 + 1
                col = i % 3 + 1
                
                fig2.add_trace(
                    go.Scatter(
                        x=predictions_gdf[feature],
                        y=predictions_gdf['ree_probability'],
                        mode='markers',
                        name=feature,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig2.update_layout(
                title='REE Probability vs Features',
                height=600
            )
        else:
            fig2 = None
        
        return fig1, fig2
    
    def run_app(self):
        """Run the Streamlit app."""
        st.set_page_config(
            page_title="REE Discovery Dashboard",
            page_icon="🗺️",
            layout="wide"
        )
        
        st.title("🗺️ Rare Earth Element Discovery Dashboard")
        st.markdown("Interactive exploration of REE potential in California using Google Earth Engine and Machine Learning")
        
        # Load data
        with st.spinner("Loading data..."):
            data = self.load_data()
        
        # Sidebar controls
        st.sidebar.header("Map Controls")
        
        # Layer visibility
        show_occurrences = st.sidebar.checkbox("✅ Show Known REE Sites", value=True)
        show_heatmap = st.sidebar.checkbox("🌡️ Show Prediction Heatmap", value=True)
        show_predictions = st.sidebar.checkbox("🔮 Show ML Predictions", value=True)
        show_training = st.sidebar.checkbox("🟠 Show Training Sites", value=True)
        
        # Heatmap opacity
        if show_heatmap:
            opacity = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.6)
        else:
            opacity = 0.6
        
        # Prediction threshold
        if show_predictions:
            threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5)
        else:
            threshold = 0.5
        
        # Create map
        m = self.create_base_map()
        
        # Add layers based on controls
        if show_occurrences and len(self.ree_occurrences) > 0:
            m = self.add_ree_occurrences(m, self.ree_occurrences)
        
        if show_heatmap and len(self.predictions) > 0:
            m = self.add_prediction_heatmap(m, self.predictions, opacity)
        
        if show_predictions and len(self.predictions) > 0:
            m = self.add_prediction_points(m, self.predictions, threshold, show_training)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display map
        st.subheader("Interactive Map")
        
        # Add legend
        st.markdown("""
        **Map Legend:**
        - ✅ **Red circles**: Known REE sites (verified deposits)
        - 🟠 **Orange circles**: Training sites (used for model training)
        - 🔮 **Purple/Blue/Green circles**: New ML predictions (AI-generated potential)
        - 🌡️ **Heatmap**: Overall REE potential across the region
        """)
        
        map_data = st_folium(m, width=1200, height=600)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Known REE Sites", len(self.ree_occurrences))
        
        with col2:
            if len(self.predictions) > 0:
                st.metric("🔮 ML Predictions", len(self.predictions))
            else:
                st.metric("🔮 ML Predictions", 0)
        
        with col3:
            if len(self.predictions) > 0:
                high_potential = len(self.predictions[self.predictions['ree_probability'] > threshold])
                st.metric(f"🔮 High Potential (>{threshold})", high_potential)
            else:
                st.metric(f"High Potential (>{threshold})", 0)
        
        with col4:
            if len(self.predictions) > 0:
                max_prob = self.predictions['ree_probability'].max()
                st.metric("Max Probability", f"{max_prob:.3f}")
            else:
                st.metric("Max Probability", "N/A")
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Prediction Analysis", "Model Performance"])
        
        with tab1:
            st.subheader("Feature Analysis")
            if len(self.features) > 0:
                # Feature statistics
                st.write("Feature Statistics:")
                st.dataframe(self.features.describe())
                
                # Feature plots
                feature_plot = self.create_feature_analysis_plots(self.features)
                if feature_plot:
                    st.plotly_chart(feature_plot, use_container_width=True)
            else:
                st.info("No feature data available. Run the feature extraction script first.")
        
        with tab2:
            st.subheader("Prediction Analysis")
            if len(self.predictions) > 0:
                # Prediction statistics
                st.write("Prediction Statistics:")
                pred_stats = self.predictions['ree_probability'].describe()
                st.dataframe(pred_stats.to_frame().T)
                
                # Prediction plots
                prob_plot, feature_plot = self.create_prediction_analysis_plots(self.predictions)
                if prob_plot:
                    st.plotly_chart(prob_plot, use_container_width=True)
                if feature_plot:
                    st.plotly_chart(feature_plot, use_container_width=True)
            else:
                st.info("No prediction data available. Run the prediction script first.")
        
        with tab3:
            st.subheader("Model Performance")
            st.info("Model performance metrics will be displayed here after training.")
            
            # Check for model evaluation files
            if os.path.exists('models/model_evaluation_curves.png'):
                st.image('models/model_evaluation_curves.png', caption="Model Evaluation Curves")
            
            if os.path.exists('models/feature_importance.png'):
                st.image('models/feature_importance.png', caption="Feature Importance")
        
        # Footer
        st.markdown("---")
        st.markdown("**Data Sources:** USGS REE Occurrences, Google Earth Engine, Sentinel-2")
        st.markdown("**Model:** Random Forest Classifier trained on environmental and spectral features")

def main():
    """Main execution function."""
    app = REEVisualizationApp()
    app.run_app()

if __name__ == "__main__":
    main()
