#!/usr/bin/env python3
"""
Model Training for REE Discovery
Trains a Random Forest classifier on real GEE features for REE prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class REEModelTrainer:
    def __init__(self, data_path='data/gee_features_california.csv'):
        """Initialize the model trainer."""
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess the feature data."""
        print("Loading feature data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Feature data not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} samples")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill missing values with median for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Define feature columns (exclude coordinates and label)
        self.feature_columns = [
            'elevation_mean', 'elevation_std',
            'slope_mean', 'slope_std',
            'NDVI_mean', 'NDVI_std',
            'NDCI_mean', 'NDCI_std',
            'IronRatio_mean', 'IronRatio_std',
            'landcover_mode'
        ]
        
        # Check if all feature columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Prepare features and labels
        X = df[self.feature_columns]
        y = df['label']
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the Random Forest model."""
        print("Training Random Forest model...")
        
        # Initialize Random Forest with specified parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        print("✓ Model training completed")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\nEvaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = self.model.score(self.X_test, self.y_test)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✓ Accuracy: {accuracy:.4f}")
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")
        print(f"✓ ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save evaluation plots
        self._plot_evaluation_curves(y_pred_proba)
        self._plot_feature_importance()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def _plot_evaluation_curves(self, y_pred_proba):
        """Plot ROC and Precision-Recall curves."""
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/model_evaluation_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Evaluation curves saved to models/model_evaluation_curves.png")
    
    def _plot_feature_importance(self):
        """Plot feature importance."""
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Feature importance plot saved to models/feature_importance.png")
    
    def save_model(self, model_path='models/model_ree_california.pkl'):
        """Save the trained model."""
        os.makedirs('models', exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        joblib.dump(model_data, model_path)
        print(f"✓ Model saved to {model_path}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("=" * 60)
        print("REE Model Training Pipeline")
        print("=" * 60)
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        self.split_data(X, y)
        
        # Train model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        
        return metrics

def main():
    """Main execution function."""
    trainer = REEModelTrainer()
    metrics = trainer.run_training_pipeline()
    
    print(f"\nFinal Model Performance:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    if metrics['roc_auc'] > 0.7:
        print("✓ Model shows good predictive performance (ROC-AUC > 0.7)")
    else:
        print("⚠ Model performance is below target (ROC-AUC < 0.7)")

if __name__ == "__main__":
    main()
