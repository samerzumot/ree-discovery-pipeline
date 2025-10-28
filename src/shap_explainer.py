#!/usr/bin/env python3
"""
SHAP Model Interpretability for REE Discovery
Provides explanations for model predictions using SHAP values.
"""

import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import streamlit as st
import os

class SHAPExplainer:
    """SHAP explainer for REE discovery model."""
    
    def __init__(self, model_path='models/model_ree_california.pkl'):
        """Initialize SHAP explainer with trained model."""
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.explainer = None
        self.feature_names = None
        
        # Load model and create explainer
        self._load_model()
        self._create_explainer()
    
    def _load_model(self):
        """Load the trained model and feature columns."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            
            # Create human-readable feature names
            self.feature_names = self._create_feature_names()
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def _create_feature_names(self):
        """Create human-readable feature names for SHAP explanations."""
        feature_mapping = {
            'elevation_mean': 'Elevation',
            'elevation_std': 'Elevation Variability',
            'slope_mean': 'Terrain Slope',
            'slope_std': 'Slope Variability',
            'NDVI_mean': 'Vegetation Index',
            'NDVI_std': 'Vegetation Variability',
            'NDCI_mean': 'Chlorophyll Index',
            'NDCI_std': 'Chlorophyll Variability',
            'IronRatio_mean': 'Iron Content',
            'IronRatio_std': 'Iron Variability',
            'landcover_mode': 'Land Cover Type'
        }
        
        return [feature_mapping.get(col, col) for col in self.feature_columns]
    
    def _create_explainer(self):
        """Create SHAP explainer for the model."""
        # Use TreeExplainer for Random Forest
        self.explainer = shap.TreeExplainer(self.model)
    
    def explain_prediction(self, features):
        """Explain a single prediction using SHAP values."""
        if isinstance(features, dict):
            # Convert dict to array
            feature_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        else:
            feature_array = features.reshape(1, -1) if features.ndim == 1 else features
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(feature_array)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Positive class
        
        # Ensure shap_values is a proper array
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # Get base value (expected output)
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray) and len(base_value) == 2:
            base_value = base_value[1]  # Positive class
        elif isinstance(base_value, (list, tuple)) and len(base_value) == 2:
            base_value = base_value[1]  # Positive class
        
        # Get prediction
        prediction = self.model.predict_proba(feature_array)[0][1]
        
        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'prediction': prediction,
            'feature_values': feature_array[0],
            'feature_names': self.feature_names
        }
    
    def create_waterfall_chart(self, explanation, title="SHAP Explanation"):
        """Create a waterfall chart showing SHAP contributions."""
        shap_values = explanation['shap_values']
        base_value = explanation['base_value']
        prediction = explanation['prediction']
        feature_names = explanation['feature_names']
        
        # Ensure shap_values is iterable
        if isinstance(shap_values, np.ndarray):
            shap_values = shap_values.flatten()
        
        # Sort features by absolute SHAP value
        feature_importance = list(zip(feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create waterfall data
        cumulative = base_value
        waterfall_data = []
        
        # Add base value
        waterfall_data.append({
            'Feature': 'Base Value',
            'SHAP Value': base_value,
            'Cumulative': base_value,
            'Color': 'blue'
        })
        
        # Add feature contributions
        for feature, shap_val in feature_importance:
            cumulative += shap_val
            color = 'green' if shap_val > 0 else 'red'
            waterfall_data.append({
                'Feature': feature,
                'SHAP Value': shap_val,
                'Cumulative': cumulative,
                'Color': color
            })
        
        # Add final prediction
        waterfall_data.append({
            'Feature': 'Final Prediction',
            'SHAP Value': prediction - base_value,
            'Cumulative': prediction,
            'Color': 'purple'
        })
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Add bars
        for i, data in enumerate(waterfall_data):
            fig.add_trace(go.Bar(
                x=[data['Feature']],
                y=[data['SHAP Value']],
                marker_color=data['Color'],
                text=[f"{data['SHAP Value']:.3f}"],
                textposition='auto',
                name=data['Feature']
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_feature_importance_plot(self, explanation):
        """Create a horizontal bar chart of feature importance."""
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        
        # Ensure shap_values is iterable
        if isinstance(shap_values, np.ndarray):
            shap_values = shap_values.flatten()
        
        # Sort by absolute SHAP value
        feature_importance = list(zip(feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        features, values = zip(*feature_importance)
        
        # Create colors based on positive/negative values
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    
    def create_summary_plot(self, X_data, max_display=10):
        """Create SHAP summary plot for multiple predictions."""
        shap_values = self.explainer.shap_values(X_data)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # All samples, all features, positive class
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Create summary plot
        fig = shap.summary_plot(
            shap_values, 
            X_data, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        return fig
    
    def get_explanation_text(self, explanation, top_n=3):
        """Generate human-readable explanation text."""
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        prediction = explanation['prediction']
        
        # Ensure shap_values is iterable
        if isinstance(shap_values, np.ndarray):
            shap_values = shap_values.flatten()
        
        # Sort features by absolute SHAP value
        feature_importance = list(zip(feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation
        explanation_parts = []
        
        if prediction > 0.5:
            explanation_parts.append(f"This location has a <strong>{prediction:.1%}</strong> probability of containing REE deposits.")
            explanation_parts.append("The high prediction is primarily due to:")
        else:
            explanation_parts.append(f"This location has a <strong>{prediction:.1%}</strong> probability of containing REE deposits.")
            explanation_parts.append("The low prediction is primarily due to:")
        
        # Add top contributing factors
        for i, (feature, shap_val) in enumerate(feature_importance[:top_n]):
            direction = "increases" if shap_val > 0 else "decreases"
            explanation_parts.append(f"• <strong>{feature}</strong> {direction} the prediction by {abs(shap_val):.3f}")
        
        return "<br>".join(explanation_parts)

def create_shap_sidebar(explanation, shap_explainer):
    """Create SHAP explanation sidebar for Streamlit."""
    st.sidebar.markdown("## 🔍 Model Explanation")
    
    # Show prediction probability
    prediction = explanation['prediction']
    st.sidebar.metric("REE Probability", f"{prediction:.1%}")
    
    # Show explanation text
    explanation_text = shap_explainer.get_explanation_text(explanation)
    st.sidebar.markdown(explanation_text, unsafe_allow_html=True)
    
    # Show waterfall chart
    st.sidebar.markdown("### Feature Contributions")
    waterfall_fig = shap_explainer.create_waterfall_chart(explanation)
    st.sidebar.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Show feature importance
    st.sidebar.markdown("### Feature Importance")
    importance_fig = shap_explainer.create_feature_importance_plot(explanation)
    st.sidebar.plotly_chart(importance_fig, use_container_width=True)

def main():
    """Test the SHAP explainer."""
    try:
        explainer = SHAPExplainer()
        
        # Test with sample data
        sample_features = {
            'elevation_mean': 1200.0,
            'elevation_std': 25.0,
            'slope_mean': 8.5,
            'slope_std': 5.2,
            'NDVI_mean': 0.0,
            'NDVI_std': 0.0,
            'NDCI_mean': 0.0,
            'NDCI_std': 0.0,
            'IronRatio_mean': 0.0,
            'IronRatio_std': 0.0,
            'landcover_mode': 0.0
        }
        
        explanation = explainer.explain_prediction(sample_features)
        
        print("✅ SHAP Explainer initialized successfully!")
        print(f"Prediction: {explanation['prediction']:.3f}")
        print(f"Base value: {explanation['base_value']:.3f}")
        print("\nTop contributing features:")
        
        # Ensure shap_values is properly formatted
        shap_values = explanation['shap_values']
        if isinstance(shap_values, np.ndarray):
            shap_values = shap_values.flatten()
        
        feature_importance = list(zip(explanation['feature_names'], shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, shap_val in feature_importance[:5]:
            print(f"  {feature}: {shap_val:.3f}")
        
    except Exception as e:
        print(f"❌ Error initializing SHAP explainer: {e}")

if __name__ == "__main__":
    main()
