#!/usr/bin/env python3
"""
Large-scale feature extraction for REE discovery using Google Earth Engine.
Handles expanded geographic coverage and larger datasets.
"""

import os
import pandas as pd
import numpy as np
import ee
import time
import requests
import zipfile
import io
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeScaleREEFeatureExtractor:
    """Extract features for REE discovery using Google Earth Engine with large-scale processing."""
    
    def __init__(self, project_id: str = 'robotic-rampart-474204-c0'):
        """Initialize the feature extractor with GEE project."""
        self.project_id = project_id
        self.initialize_gee()
        
    def initialize_gee(self):
        """Initialize Google Earth Engine with error handling."""
        try:
            ee.Initialize(project=self.project_id)
            logger.info(f"✅ GEE initialized with project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ GEE initialization failed: {e}")
            logger.info("Please ensure:")
            logger.info("1. Earth Engine API is enabled in Google Cloud Console")
            logger.info("2. Project is registered for Earth Engine")
            logger.info("3. Authentication is set up")
            raise
    
    def download_usgs_ree_data_expanded(self) -> pd.DataFrame:
        """Download USGS REE data for expanded Western US coverage."""
        logger.info("Downloading USGS REE data for Western US...")
        
        url = "https://mrdata.usgs.gov/ree/ree-csv.zip"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Look for the main CSV file
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                logger.info(f"Found CSV files: {csv_files}")
                
                # Use main.csv which contains the coordinate data
                main_csv = 'ree/main.csv'
                
                logger.info(f"Using CSV file: {main_csv}")
                
                with zip_file.open(main_csv) as csv_file:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded {len(df)} records from {main_csv}")
                    logger.info(f"Columns: {list(df.columns)}")
                    
                    # Use the correct column names from main.csv
                    lat_col = 'latitude'
                    lon_col = 'longitude'
                    
                    logger.info(f"Using latitude column: {lat_col}")
                    logger.info(f"Using longitude column: {lon_col}")
                    
                    # Filter to Western US (expanded coverage)
                    western_us_mask = (
                        (df[lat_col] >= 30.0) & (df[lat_col] <= 45.0) &
                        (df[lon_col] >= -125.0) & (df[lon_col] <= -110.0)
                    )
                    
                    filtered_df = df[western_us_mask].copy()
                    logger.info(f"Filtered to Western US: {len(filtered_df)} records")
                    
                    # Clean and standardize
                    filtered_df = filtered_df.dropna(subset=[lat_col, lon_col])
                    filtered_df = filtered_df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                    
                    # Add name and mineral columns using available columns
                    filtered_df['name'] = filtered_df['depname']  # Use deposit name
                    filtered_df['mineral'] = filtered_df['deptype']  # Use deposit type
                    
                    return filtered_df[['lat', 'lon', 'name', 'mineral']]
                    
        except Exception as e:
            logger.error(f"Failed to download USGS REE data: {e}")
            raise
    
    def download_usgs_background_data_expanded(self, target_count: int = 200) -> pd.DataFrame:
        """Download USGS non-REE mineral deposits for background points."""
        logger.info(f"Downloading USGS non-REE data (target: {target_count} points)...")
        
        url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                logger.info(f"Found CSV files: {csv_files}")
                
                main_csv = csv_files[0]  # Use first CSV
                logger.info(f"Using CSV file: {main_csv}")
                
                with zip_file.open(main_csv) as csv_file:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded {len(df)} records from {main_csv}")
                    
                    # Find coordinate columns
                    lat_cols = [col for col in df.columns if 'lat' in col.lower()]
                    lon_cols = [col for col in df.columns if 'lon' in col.lower()]
                    
                    if not lat_cols or not lon_cols:
                        raise ValueError("Could not find latitude/longitude columns")
                    
                    lat_col = lat_cols[0]
                    lon_col = lon_cols[0]
                    
                    # Filter to Western US
                    western_us_mask = (
                        (df[lat_col] >= 30.0) & (df[lat_col] <= 45.0) &
                        (df[lon_col] >= -125.0) & (df[lon_col] <= -110.0)
                    )
                    
                    filtered_df = df[western_us_mask].copy()
                    logger.info(f"Filtered to Western US: {len(filtered_df)} records")
                    
                    # Clean and standardize
                    filtered_df = filtered_df.dropna(subset=[lat_col, lon_col])
                    filtered_df = filtered_df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                    
                    # Sample if we have too many
                    if len(filtered_df) > target_count:
                        filtered_df = filtered_df.sample(n=target_count, random_state=42)
                        logger.info(f"Sampled to {target_count} points")
                    
                    # Add labels
                    filtered_df['name'] = f"Non-REE Site {range(len(filtered_df))}"
                    filtered_df['mineral'] = 'Non-REE'
                    
                    return filtered_df[['lat', 'lon', 'name', 'mineral']]
                    
        except Exception as e:
            logger.error(f"Failed to download USGS non-REE data: {e}")
            raise
    
    def extract_gee_features_batch(self, points_df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """Extract GEE features for points in batches to handle large datasets."""
        logger.info(f"Extracting GEE features for {len(points_df)} points in batches of {batch_size}")
        
        all_features = []
        
        for i in range(0, len(points_df), batch_size):
            batch = points_df.iloc[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(points_df)-1)//batch_size + 1}")
            
            batch_features = []
            for idx, row in batch.iterrows():
                try:
                    features = self.extract_features_for_point(row['lat'], row['lon'])
                    features['lat'] = row['lat']
                    features['lon'] = row['lon']
                    features['name'] = row['name']
                    features['mineral'] = row['mineral']
                    batch_features.append(features)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features for point {idx}: {e}")
                    # Add default features for failed points
                    default_features = self._get_default_features()
                    default_features['lat'] = row['lat']
                    default_features['lon'] = row['lon']
                    default_features['name'] = row['name']
                    default_features['mineral'] = row['mineral']
                    batch_features.append(default_features)
            
            all_features.extend(batch_features)
            logger.info(f"Completed batch {i//batch_size + 1}")
            
            # Longer delay between batches
            time.sleep(1)
        
        return pd.DataFrame(all_features)
    
    def extract_features_for_point(self, lat: float, lon: float) -> Dict:
        """Extract comprehensive GEE features for a single point."""
        try:
            point = ee.Geometry.Point([lon, lat])
            
            # Elevation data
            elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
            elevation_stats = elevation.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=point.buffer(1000),  # 1km buffer
                scale=30,
                maxPixels=1e9
            )
            
            # Terrain data
            terrain = ee.Terrain.products(elevation)
            slope_stats = terrain.select('slope').reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=point.buffer(1000),
                scale=30,
                maxPixels=1e9
            )
            
            # Landcover data
            landcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
            landcover_mode = landcover.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=point.buffer(1000),
                scale=30,
                maxPixels=1e9
            )
            
            # Optical data (Sentinel-2)
            optical = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2020-01-01', '2020-12-31').filterBounds(point).first()
            
            if optical:
                # Calculate NDVI
                ndvi = optical.normalizedDifference(['B8', 'B4']).rename('NDVI')
                ndvi_stats = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=point.buffer(1000),
                    scale=30,
                    maxPixels=1e9
                )
                
                # Calculate NDCI (Normalized Difference Chlorophyll Index)
                ndci = optical.normalizedDifference(['B5', 'B4']).rename('NDCI')
                ndci_stats = ndci.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=point.buffer(1000),
                    scale=30,
                    maxPixels=1e9
                )
                
                # Calculate Iron Ratio
                iron_ratio = optical.select('B12').divide(optical.select('B4')).rename('IronRatio')
                iron_stats = iron_ratio.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=point.buffer(1000),
                    scale=30,
                    maxPixels=1e9
                )
            else:
                ndvi_stats = {'NDVI_mean': 0.0, 'NDVI_stdDev': 0.0}
                ndci_stats = {'NDCI_mean': 0.0, 'NDCI_stdDev': 0.0}
                iron_stats = {'IronRatio_mean': 0.0, 'IronRatio_stdDev': 0.0}
            
            # Get values
            elevation_values = elevation_stats.getInfo()
            slope_values = slope_stats.getInfo()
            landcover_value = landcover_mode.getInfo()
            ndvi_values = ndvi_stats
            ndci_values = ndci_stats
            iron_values = iron_stats
            
            # Convert to safe floats
            features = {
                'elevation_mean': self.safe_float(elevation_values.get('elevation_mean', 0)),
                'elevation_std': self.safe_float(elevation_values.get('elevation_stdDev', 0)),
                'slope_mean': self.safe_float(slope_values.get('slope_mean', 0)),
                'slope_std': self.safe_float(slope_values.get('slope_stdDev', 0)),
                'NDVI_mean': self.safe_float(ndvi_values.get('NDVI_mean', 0)),
                'NDVI_std': self.safe_float(ndvi_values.get('NDVI_stdDev', 0)),
                'NDCI_mean': self.safe_float(ndci_values.get('NDCI_mean', 0)),
                'NDCI_std': self.safe_float(ndci_values.get('NDCI_stdDev', 0)),
                'IronRatio_mean': self.safe_float(iron_values.get('IronRatio_mean', 0)),
                'IronRatio_std': self.safe_float(iron_values.get('IronRatio_stdDev', 0)),
                'landcover_mode': self.safe_float(landcover_value.get('Map', 0))
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"GEE extraction failed for point ({lat}, {lon}): {e}")
            return self._get_default_features()
    
    def safe_float(self, value) -> float:
        """Convert value to float safely."""
        if value is None:
            return 0.0
        if isinstance(value, str) and 'ComputedObject' in value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _get_default_features(self) -> Dict:
        """Get default features when GEE extraction fails."""
        return {
            'elevation_mean': 0.0,
            'elevation_std': 0.0,
            'slope_mean': 0.0,
            'slope_std': 0.0,
            'NDVI_mean': 0.0,
            'NDVI_std': 0.0,
            'NDCI_mean': 0.0,
            'NDCI_std': 0.0,
            'IronRatio_mean': 0.0,
            'IronRatio_std': 0.0,
            'landcover_mode': 0.0
        }
    
    def generate_training_data(self, target_ree_count: int = 50, target_background_count: int = 200) -> pd.DataFrame:
        """Generate comprehensive training data for Western US."""
        logger.info("Generating large-scale training data...")
        
        # Download REE data
        try:
            ree_df = self.download_usgs_ree_data_expanded()
            logger.info(f"Downloaded {len(ree_df)} REE sites")
        except Exception as e:
            logger.error(f"Failed to download REE data: {e}")
            raise
        
        # Download background data
        try:
            background_df = self.download_usgs_background_data_expanded(target_background_count)
            logger.info(f"Downloaded {len(background_df)} background sites")
        except Exception as e:
            logger.error(f"Failed to download background data: {e}")
            raise
        
        # Sample if we have too many REE sites
        if len(ree_df) > target_ree_count:
            ree_df = ree_df.sample(n=target_ree_count, random_state=42)
            logger.info(f"Sampled to {target_ree_count} REE sites")
        
        # Combine datasets
        ree_df['label'] = 1  # REE sites
        background_df['label'] = 0  # Background sites
        
        all_points = pd.concat([ree_df, background_df], ignore_index=True)
        logger.info(f"Combined dataset: {len(all_points)} total points")
        
        # Extract GEE features
        logger.info("Extracting GEE features...")
        features_df = self.extract_gee_features_batch(all_points, batch_size=25)
        
        # Add labels back to the features dataframe
        features_df['label'] = all_points['label'].values
        
        # Save results
        os.makedirs('data', exist_ok=True)
        output_path = 'data/large_scale_features.csv'
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        return features_df

def main():
    """Main function to run large-scale feature extraction."""
    extractor = LargeScaleREEFeatureExtractor()
    
    try:
        # Generate training data
        features_df = extractor.generate_training_data(
            target_ree_count=50,
            target_background_count=200
        )
        
        print(f"\n✅ Large-scale feature extraction completed!")
        print(f"Total samples: {len(features_df)}")
        print(f"REE sites: {len(features_df[features_df['label'] == 1])}")
        print(f"Background sites: {len(features_df[features_df['label'] == 0])}")
        print(f"Features extracted: {len(features_df.columns) - 4}")  # Exclude lat, lon, name, mineral
        
    except Exception as e:
        logger.error(f"Large-scale extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
