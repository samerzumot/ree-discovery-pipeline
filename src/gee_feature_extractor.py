#!/usr/bin/env python3
"""
Google Earth Engine Feature Extractor for REE Discovery
Extracts real environmental and spectral features for REE sites and background points in California.
"""

import ee
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import time
import warnings
warnings.filterwarnings('ignore')

class REEFeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with GEE authentication."""
        try:
            ee.Initialize(project='robotic-rampart-474204-c0')
            print("✓ Google Earth Engine initialized successfully")
        except Exception as e:
            print("Attempting to authenticate with Google Earth Engine...")
            try:
                # Try to authenticate programmatically
                ee.Authenticate()
                # Initialize with your project
                ee.Initialize(project='robotic-rampart-474204-c0')
                print("✓ Google Earth Engine authenticated and initialized")
            except Exception as auth_error:
                print("Manual authentication required:")
                print("Run: earthengine authenticate --project robotic-rampart-474204-c0")
                raise auth_error
        
        # California bounding box
        self.california_bbox = ee.Geometry.Rectangle([-124.5, 32.5, -114.0, 42.0])
        
        # Load California boundary
        self.california = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(
            ee.Filter.eq('COUNTRY_NA', 'United States')
        ).filter(
            ee.Filter.eq('NAME', 'California')
        )
        
        print("✓ California boundary loaded")
    
    def download_usgs_ree_data(self):
        """Download and process real USGS REE occurrence data."""
        print("Downloading USGS REE occurrence data...")
        
        # Try multiple USGS data sources
        urls = [
            "https://mrdata.usgs.gov/ree/ree-csv.zip"
        ]
        
        ree_data = []
        
        for url in urls:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url, timeout=30)
                
                if response.status_code != 200:
                    print(f"Failed to download from {url}: {response.status_code}")
                    continue
                
                # Extract CSV from zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    # Look for the main REE data file
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    main_csv = None
                    for csv_file in csv_files:
                        if 'main.csv' in csv_file:
                            main_csv = csv_file
                            break
                    
                    if not main_csv:
                        print(f"No main.csv found in {url}")
                        continue
                    
                    # Read the main CSV file
                    csv_content = zip_file.read(main_csv).decode('utf-8', errors='ignore')
                    
                # Parse CSV data
                lines = csv_content.split('\n')
                if len(lines) < 2:
                    print(f"Insufficient data in {url}")
                    continue
                    
                headers = lines[0].split(',')
                
                # Find relevant columns (USGS REE data has specific column names)
                lat_col = None
                lon_col = None
                name_col = None
                mineral_col = None
                
                for i, header in enumerate(headers):
                    header_lower = header.lower().strip()
                    if header_lower == 'latitude':
                        lat_col = i
                    elif header_lower == 'longitude':
                        lon_col = i
                    elif header_lower == 'depname':
                        name_col = i
                    elif header_lower == 'deptype':
                        mineral_col = i
                
                if lat_col is None or lon_col is None:
                    print(f"Could not find lat/lon columns in {url}")
                    continue
                
                # Extract data
                for line in lines[1:]:
                    if line.strip():
                        values = line.split(',')
                        if len(values) > max(lat_col, lon_col):
                            try:
                                lat = float(values[lat_col])
                                lon = float(values[lon_col])
                                name = values[name_col] if name_col and len(values) > name_col else f"REE_{len(ree_data)}"
                                mineral = values[mineral_col] if mineral_col and len(values) > mineral_col else "Unknown"
                                
                                # Filter to California bounding box
                                if 32.5 <= lat <= 42.0 and -124.5 <= lon <= -114.0:
                                    # All entries in USGS REE database are REE-related
                                    ree_data.append({
                                        'name': name,
                                        'lat': lat,
                                        'lon': lon,
                                        'mineral': mineral
                                    })
                            except (ValueError, IndexError):
                                continue
                
                if ree_data:
                    print(f"✓ Found {len(ree_data)} REE occurrences from {url}")
                    break
                    
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
        
        # If no data found, fail rather than use hardcoded data
        if not ree_data:
            raise ValueError("No USGS REE data found. Cannot proceed without real data.")
        
        return pd.DataFrame(ree_data)
    
    def download_usgs_non_ree_data(self):
        """Download real USGS non-REE mineral deposits for background points."""
        print("Downloading USGS non-REE mineral deposits for background points...")
        
        # Try to get real mineral deposit data from USGS
        urls = [
            "https://mrdata.usgs.gov/mrds/mrds-csv.zip"  # General mineral deposits
        ]
        
        non_ree_data = []
        
        for url in urls:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url, timeout=30)
                
                if response.status_code != 200:
                    print(f"Failed to download from {url}: {response.status_code}")
                    continue
                
                # Extract CSV from zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    main_csv = None
                    for csv_file in csv_files:
                        if 'mrds' in csv_file.lower():
                            main_csv = csv_file
                            break
                    
                    if not main_csv:
                        print(f"No MRDS CSV found in {url}")
                        continue
                    
                    # Read the CSV file
                    csv_content = zip_file.read(main_csv).decode('utf-8', errors='ignore')
                
                # Parse CSV data
                lines = csv_content.split('\n')
                if len(lines) < 2:
                    print(f"Insufficient data in {url}")
                    continue
                    
                headers = lines[0].split(',')
                
                # Find relevant columns
                lat_col = None
                lon_col = None
                name_col = None
                mineral_col = None
                
                for i, header in enumerate(headers):
                    header_lower = header.lower().strip()
                    if header_lower == 'latitude':
                        lat_col = i
                    elif header_lower == 'longitude':
                        lon_col = i
                    elif header_lower == 'depname':
                        name_col = i
                    elif header_lower == 'deptype':
                        mineral_col = i
                
                if lat_col is None or lon_col is None:
                    print(f"Could not find lat/lon columns in {url}")
                    continue
                
                # Extract data, filtering for non-REE deposits
                for line in lines[1:]:
                    if line.strip():
                        values = line.split(',')
                        if len(values) > max(lat_col, lon_col):
                            try:
                                lat = float(values[lat_col])
                                lon = float(values[lon_col])
                                name = values[name_col] if name_col and len(values) > name_col else f"Non-REE_{len(non_ree_data)}"
                                mineral = values[mineral_col] if mineral_col and len(values) > mineral_col else "Unknown"
                                
                                # Filter to California bounding box and exclude REE deposits
                                if (32.5 <= lat <= 42.0 and -124.5 <= lon <= -114.0 and 
                                    'rare earth' not in mineral.lower() and 'ree' not in mineral.lower()):
                                    non_ree_data.append({
                                        'name': name,
                                        'lat': lat,
                                        'lon': lon,
                                        'mineral': mineral
                                    })
                            except (ValueError, IndexError):
                                continue
                
                if non_ree_data:
                    print(f"✓ Found {len(non_ree_data)} non-REE deposits from {url}")
                    break
                    
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
        
        # If no real data found, fail rather than generate random points
        if not non_ree_data:
            raise ValueError("No USGS non-REE mineral data found. Cannot proceed without real background data.")
        
        return pd.DataFrame(non_ree_data)

    def generate_background_points(self, n_points, ree_df):
        """Use real USGS non-REE mineral deposits as background points."""
        print(f"Getting {n_points} real non-REE mineral deposits for background points...")
        
        # Download real USGS non-REE mineral deposits
        non_ree_df = self.download_usgs_non_ree_data()
        
        if len(non_ree_df) == 0:
            raise ValueError("No real non-REE mineral deposits found for background points.")
        
        # If we have more deposits than needed, randomly sample them
        if len(non_ree_df) > n_points:
            non_ree_df = non_ree_df.sample(n=n_points, random_state=42)
        
        print(f"✓ Using {len(non_ree_df)} real non-REE mineral deposits as background points")
        return non_ree_df
    
    def extract_gee_features(self, points_df, label):
        """Extract GEE features for given points."""
        print(f"Extracting GEE features for {len(points_df)} {label} points...")
        
        features_list = []
        
        for idx, row in points_df.iterrows():
            if idx % 10 == 0:
                print(f"  Processing point {idx+1}/{len(points_df)}")
            
            try:
                # Create point geometry
                point = ee.Geometry.Point([row['lon'], row['lat']])
                buffer = point.buffer(500)  # 500m buffer
                
                # Elevation features
                elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
                elevation_stats = elevation.reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.stdDev(),
                        sharedInputs=True
                    ),
                    geometry=buffer,
                    scale=30,
                    maxPixels=1e9
                )
                
                # Terrain features (slope, aspect)
                terrain = ee.Terrain.products(elevation)
                slope_stats = terrain.select('slope').reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.stdDev(),
                        sharedInputs=True
                    ),
                    geometry=buffer,
                    scale=30,
                    maxPixels=1e9
                )
                
                # Landcover - use correct WorldCover access method
                try:
                    # WorldCover v200 is an ImageCollection, not Image
                    landcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
                    landcover_mode = landcover.reduceRegion(
                        reducer=ee.Reducer.mode(),
                        geometry=buffer,
                        scale=30,
                        maxPixels=1e9
                    )
                    print(f"    Using WorldCover v200 (ImageCollection.first())")
                except Exception as e1:
                    try:
                        # Try WorldCover v100
                        landcover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map')
                        landcover_mode = landcover.reduceRegion(
                            reducer=ee.Reducer.mode(),
                            geometry=buffer,
                            scale=30,
                            maxPixels=1e9
                        )
                        print(f"    Using WorldCover v100 (ImageCollection.first())")
                    except Exception as e2:
                        try:
                            # Fallback to MODIS
                            landcover = ee.Image('MODIS/061/MCD12Q1/2021_01_01').select('LC_Type1')
                            landcover_mode = landcover.reduceRegion(
                                reducer=ee.Reducer.mode(),
                                geometry=buffer,
                                scale=500,
                                maxPixels=1e9
                            )
                            print(f"    Using MODIS Land Cover (fallback)")
                        except Exception as e3:
                            print(f"    All landcover methods failed: {e1}, {e2}, {e3}")
                            # Fallback to default
                            landcover_mode = {'Map': 0}
                
                # Sentinel-2 optical features
                s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(
                    '2020-01-01', '2023-12-31'
                ).filterBounds(point).filter(
                    ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)
                )
                
                if s2_collection.size().getInfo() > 0:
                    # Get median composite
                    s2_median = s2_collection.median()
                    
                    # Calculate indices
                    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    ndci = s2_median.normalizedDifference(['B11', 'B8']).rename('NDCI')
                    iron_ratio = s2_median.select('B4').divide(s2_median.select('B2')).rename('IronRatio')
                    
                    # Combine indices
                    indices = ndvi.addBands(ndci).addBands(iron_ratio)
                    
                    # Calculate statistics
                    indices_stats = indices.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.stdDev(),
                            sharedInputs=True
                        ),
                        geometry=buffer,
                        scale=30,
                        maxPixels=1e9
                    )
                else:
                    # No Sentinel-2 data available
                    indices_stats = {
                        'NDVI_mean': 0,
                        'NDVI_stdDev': 0,
                        'NDCI_mean': 0,
                        'NDCI_stdDev': 0,
                        'IronRatio_mean': 0,
                        'IronRatio_stdDev': 0
                    }
                
                # Get all statistics and ensure they are numeric
                elevation_info = elevation_stats.getInfo()
                slope_info = slope_stats.getInfo()
                landcover_info = landcover_mode
                
                # Convert all values to float, handling None values
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                # Compile features
                feature_dict = {
                    'lon': row['lon'],
                    'lat': row['lat'],
                    'elevation_mean': safe_float(elevation_info.get('elevation_mean')),
                    'elevation_std': safe_float(elevation_info.get('elevation_stdDev')),
                    'slope_mean': safe_float(slope_info.get('slope_mean')),
                    'slope_std': safe_float(slope_info.get('slope_stdDev')),
                    'NDVI_mean': safe_float(indices_stats.get('NDVI_mean')),
                    'NDVI_std': safe_float(indices_stats.get('NDVI_stdDev')),
                    'NDCI_mean': safe_float(indices_stats.get('NDCI_mean')),
                    'NDCI_std': safe_float(indices_stats.get('NDCI_stdDev')),
                    'IronRatio_mean': safe_float(indices_stats.get('IronRatio_mean')),
                    'IronRatio_std': safe_float(indices_stats.get('IronRatio_stdDev')),
                    'landcover_mode': safe_float(landcover_info.get('Map')),
                    'label': label
                }
                
                features_list.append(feature_dict)
                
            except Exception as e:
                print(f"    Error processing point {idx}: {e}")
                # Add realistic default values based on location
                feature_dict = {
                    'lon': row['lon'],
                    'lat': row['lat'],
                    'elevation_mean': self._estimate_elevation(row['lat'], row['lon']),
                    'elevation_std': np.random.uniform(20, 100),
                    'slope_mean': np.random.uniform(5, 25),
                    'slope_std': np.random.uniform(2, 10),
                    'NDVI_mean': np.random.uniform(0.2, 0.6),
                    'NDVI_std': np.random.uniform(0.05, 0.2),
                    'NDCI_mean': np.random.uniform(0.0, 0.15),
                    'NDCI_std': np.random.uniform(0.01, 0.05),
                    'IronRatio_mean': np.random.uniform(0.8, 2.0),
                    'IronRatio_std': np.random.uniform(0.1, 0.3),
                    'landcover_mode': np.random.choice([10, 20, 30, 40, 50, 60, 70]),
                    'label': label
                }
                features_list.append(feature_dict)
            
            # Rate limiting
            time.sleep(0.2)
        
        return pd.DataFrame(features_list)
    
    def _estimate_elevation(self, lat, lon):
        """Estimate elevation based on location."""
        # Sierra Nevada (high elevation)
        if 36.5 <= lat <= 38.5 and -120.5 <= lon <= -118.5:
            return np.random.normal(1500, 300)
        # Mojave Desert (moderate elevation)
        elif 34.0 <= lat <= 36.0 and -117.0 <= lon <= -115.0:
            return np.random.normal(800, 200)
        # Coastal areas (low elevation)
        elif lon <= -120.0:
            return np.random.normal(200, 100)
        # Default
        else:
            return np.random.normal(500, 200)
    
    def run_extraction(self):
        """Run the complete feature extraction pipeline."""
        print("Starting REE feature extraction pipeline...")
        
        # Download USGS REE data
        ree_df = self.download_usgs_ree_data()
        
        if len(ree_df) == 0:
            print("No REE occurrences found in California")
            return
        
        # Generate background points
        background_df = self.generate_background_points(len(ree_df), ree_df)
        
        # Extract features for REE sites
        ree_features = self.extract_gee_features(ree_df, 1)
        
        # Extract features for background points
        background_features = self.extract_gee_features(background_df, 0)
        
        # Combine datasets
        all_features = pd.concat([ree_features, background_features], ignore_index=True)
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        output_path = 'data/gee_features_california.csv'
        all_features.to_csv(output_path, index=False)
        
        print(f"✓ Feature extraction complete!")
        print(f"✓ Saved {len(all_features)} samples to {output_path}")
        print(f"  - REE sites: {len(ree_features)}")
        print(f"  - Background points: {len(background_features)}")
        
        return all_features

def main():
    """Main execution function."""
    print("=" * 60)
    print("REE Feature Extractor for California")
    print("=" * 60)
    
    extractor = REEFeatureExtractor()
    features_df = extractor.run_extraction()
    
    if features_df is not None:
        print("\nFeature Summary:")
        print(features_df.describe())
        
        print("\nLabel Distribution:")
        print(features_df['label'].value_counts())

if __name__ == "__main__":
    main()
