#!/usr/bin/env python3
"""
Validation Script for REE Discovery System
Validates model performance against known REE sites and geological expectations.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class REEValidation:
    def __init__(self):
        """Initialize the validation system."""
        self.known_ree_sites = {
            'Mountain Pass': {'lat': 35.4667, 'lon': -115.4667, 'type': 'Major REE Mine'},
            'Searles Lake': {'lat': 35.7667, 'lon': -117.3667, 'type': 'REE Brine'},
            'Bastnaesite': {'lat': 35.5, 'lon': -115.5, 'type': 'REE Mineral'},
            'Monazite': {'lat': 35.4, 'lon': -115.4, 'type': 'REE Mineral'}
        }
        
        self.geological_regions = {
            'Sierra Nevada': {'lat': 37.0, 'lon': -119.0, 'type': 'Granitic Terrane'},
            'Mojave Desert': {'lat': 35.0, 'lon': -116.0, 'type': 'Desert Basin'},
            'Peninsular Ranges': {'lat': 33.0, 'lon': -117.0, 'type': 'Metamorphic Belt'}
        }
    
    def load_predictions(self, predictions_path='data/ree_predictions_california.geojson'):
        """Load prediction results."""
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        
        predictions = gpd.read_file(predictions_path)
        print(f"✓ Loaded {len(predictions)} prediction points")
        return predictions
    
    def load_features(self, features_path='data/gee_features_california.csv'):
        """Load feature data for analysis."""
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        features = pd.read_csv(features_path)
        print(f"✓ Loaded {len(features)} feature samples")
        return features
    
    def validate_known_sites(self, predictions):
        """Validate predictions against known REE sites."""
        print("\n" + "="*50)
        print("VALIDATING KNOWN REE SITES")
        print("="*50)
        
        results = {}
        
        for site_name, site_info in self.known_ree_sites.items():
            # Find nearest prediction point
            site_point = Point(site_info['lon'], site_info['lat'])
            
            # Calculate distances
            distances = predictions.geometry.distance(site_point)
            nearest_idx = distances.idxmin()
            nearest_point = predictions.iloc[nearest_idx]
            
            # Get prediction details
            distance_km = distances[nearest_idx] * 111  # Rough conversion to km
            probability = nearest_point['ree_probability']
            
            results[site_name] = {
                'distance_km': distance_km,
                'probability': probability,
                'type': site_info['type'],
                'coordinates': (site_info['lat'], site_info['lon'])
            }
            
            print(f"\n{site_name} ({site_info['type']}):")
            print(f"  Coordinates: {site_info['lat']:.4f}, {site_info['lon']:.4f}")
            print(f"  Nearest prediction: {distance_km:.2f} km away")
            print(f"  REE Probability: {probability:.4f}")
            print(f"  Status: {'✓ HIGH' if probability > 0.5 else '⚠ LOW' if probability > 0.3 else '✗ VERY LOW'}")
        
        return results
    
    def validate_geological_regions(self, predictions):
        """Validate predictions against geological regions known for REE potential."""
        print("\n" + "="*50)
        print("VALIDATING GEOLOGICAL REGIONS")
        print("="*50)
        
        region_results = {}
        
        for region_name, region_info in self.geological_regions.items():
            # Find points within region (rough bounding box)
            region_lat = region_info['lat']
            region_lon = region_info['lon']
            
            # Define region bounds (±0.5 degrees)
            lat_min, lat_max = region_lat - 0.5, region_lat + 0.5
            lon_min, lon_max = region_lon - 0.5, region_lon + 0.5
            
            # Filter predictions in region
            region_predictions = predictions[
                (predictions.geometry.y >= lat_min) & 
                (predictions.geometry.y <= lat_max) &
                (predictions.geometry.x >= lon_min) & 
                (predictions.geometry.x <= lon_max)
            ]
            
            if len(region_predictions) > 0:
                mean_prob = region_predictions['ree_probability'].mean()
                max_prob = region_predictions['ree_probability'].max()
                high_potential = len(region_predictions[region_predictions['ree_probability'] > 0.5])
                
                region_results[region_name] = {
                    'mean_probability': mean_prob,
                    'max_probability': max_prob,
                    'high_potential_count': high_potential,
                    'total_points': len(region_predictions),
                    'type': region_info['type']
                }
                
                print(f"\n{region_name} ({region_info['type']}):")
                print(f"  Points in region: {len(region_predictions)}")
                print(f"  Mean probability: {mean_prob:.4f}")
                print(f"  Max probability: {max_prob:.4f}")
                print(f"  High potential points: {high_potential}")
                print(f"  Status: {'✓ PROMISING' if mean_prob > 0.4 else '⚠ MODERATE' if mean_prob > 0.2 else '✗ LOW'}")
            else:
                print(f"\n{region_name}: No prediction points in region")
                region_results[region_name] = None
        
        return region_results
    
    def analyze_feature_importance(self, features):
        """Analyze feature importance and distributions."""
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        
        # Separate REE sites from background
        ree_sites = features[features['label'] == 1]
        background = features[features['label'] == 0]
        
        print(f"REE sites: {len(ree_sites)}")
        print(f"Background points: {len(background)}")
        
        # Feature comparison
        numeric_features = [
            'elevation_mean', 'slope_mean', 'NDVI_mean', 
            'NDCI_mean', 'IronRatio_mean'
        ]
        
        print("\nFeature Comparison (REE sites vs Background):")
        print("-" * 60)
        
        for feature in numeric_features:
            if feature in features.columns:
                ree_mean = ree_sites[feature].mean()
                bg_mean = background[feature].mean()
                difference = ree_mean - bg_mean
                percent_diff = (difference / bg_mean) * 100 if bg_mean != 0 else 0
                
                print(f"{feature:15s}: REE={ree_mean:8.3f}, BG={bg_mean:8.3f}, Diff={difference:8.3f} ({percent_diff:6.1f}%)")
        
        return ree_sites, background
    
    def create_validation_plots(self, predictions, features):
        """Create validation plots."""
        print("\nCreating validation plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('REE Discovery System Validation', fontsize=16)
        
        # 1. Probability distribution
        axes[0, 0].hist(predictions['ree_probability'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('REE Probability Distribution')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='High Potential Threshold')
        axes[0, 0].legend()
        
        # 2. Elevation vs Probability
        axes[0, 1].scatter(predictions['elevation_mean'], predictions['ree_probability'], alpha=0.6)
        axes[0, 1].set_title('Elevation vs REE Probability')
        axes[0, 1].set_xlabel('Elevation (m)')
        axes[0, 1].set_ylabel('REE Probability')
        
        # 3. NDVI vs Probability
        axes[0, 2].scatter(predictions['NDVI_mean'], predictions['ree_probability'], alpha=0.6)
        axes[0, 2].set_title('NDVI vs REE Probability')
        axes[0, 2].set_xlabel('NDVI')
        axes[0, 2].set_ylabel('REE Probability')
        
        # 4. Feature importance (if available)
        if 'elevation_mean' in features.columns:
            feature_means = features.groupby('label')[numeric_features].mean()
            feature_means.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Feature Means by Class')
            axes[1, 0].set_xlabel('Class (0=Background, 1=REE)')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. Spatial distribution of high potential areas
        high_potential = predictions[predictions['ree_probability'] > 0.5]
        axes[1, 1].scatter(predictions.geometry.x, predictions.geometry.y, 
                          c=predictions['ree_probability'], cmap='viridis', alpha=0.6, s=1)
        axes[1, 1].scatter(high_potential.geometry.x, high_potential.geometry.y, 
                          c='red', alpha=0.8, s=2, label='High Potential')
        axes[1, 1].set_title('Spatial Distribution of REE Potential')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].legend()
        
        # 6. Model performance summary
        axes[1, 2].text(0.1, 0.8, f"Total Prediction Points: {len(predictions)}", fontsize=12)
        axes[1, 2].text(0.1, 0.7, f"High Potential Points: {len(high_potential)}", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Max Probability: {predictions['ree_probability'].max():.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.5, f"Mean Probability: {predictions['ree_probability'].mean():.3f}", fontsize=12)
        axes[1, 2].set_title('Model Performance Summary')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Validation plots saved to models/validation_analysis.png")
    
    def generate_validation_report(self, site_results, region_results, ree_sites, background):
        """Generate a comprehensive validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        # Overall assessment
        high_prob_sites = sum(1 for result in site_results.values() if result['probability'] > 0.5)
        total_sites = len(site_results)
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Known REE sites with high predictions: {high_prob_sites}/{total_sites}")
        print(f"  Success rate: {high_prob_sites/total_sites*100:.1f}%")
        
        # Geological validation
        promising_regions = sum(1 for result in region_results.values() 
                              if result and result['mean_probability'] > 0.4)
        total_regions = len([r for r in region_results.values() if r is not None])
        
        print(f"\nGEOLOGICAL VALIDATION:")
        print(f"  Promising regions: {promising_regions}/{total_regions}")
        
        # Feature analysis
        print(f"\nFEATURE ANALYSIS:")
        print(f"  REE sites analyzed: {len(ree_sites)}")
        print(f"  Background points: {len(background)}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if high_prob_sites/total_sites >= 0.5:
            print("  ✓ Model shows good performance on known sites")
        else:
            print("  ⚠ Model may need improvement for known sites")
        
        if promising_regions > 0:
            print("  ✓ Model identifies geologically promising regions")
        else:
            print("  ⚠ Model may not capture geological patterns well")
        
        # Save report
        report_path = 'models/validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("REE Discovery System Validation Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Validation Date: {pd.Timestamp.now()}\n\n")
            
            f.write("KNOWN SITE VALIDATION:\n")
            for site, result in site_results.items():
                f.write(f"  {site}: {result['probability']:.3f} ({result['distance_km']:.1f}km)\n")
            
            f.write("\nGEOLOGICAL REGION VALIDATION:\n")
            for region, result in region_results.items():
                if result:
                    f.write(f"  {region}: {result['mean_probability']:.3f} mean, {result['high_potential_count']} high-potential points\n")
            
            f.write(f"\nOVERALL SUCCESS RATE: {high_prob_sites/total_sites*100:.1f}%\n")
        
        print(f"✓ Validation report saved to {report_path}")
    
    def run_validation(self):
        """Run the complete validation pipeline."""
        print("="*60)
        print("REE DISCOVERY SYSTEM VALIDATION")
        print("="*60)
        
        # Load data
        predictions = self.load_predictions()
        features = self.load_features()
        
        # Validate known sites
        site_results = self.validate_known_sites(predictions)
        
        # Validate geological regions
        region_results = self.validate_geological_regions(predictions)
        
        # Analyze features
        ree_sites, background = self.analyze_feature_importance(features)
        
        # Create plots
        self.create_validation_plots(predictions, features)
        
        # Generate report
        self.generate_validation_report(site_results, region_results, ree_sites, background)
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE!")
        print("="*60)

def main():
    """Main execution function."""
    validator = REEValidation()
    validator.run_validation()

if __name__ == "__main__":
    main()
