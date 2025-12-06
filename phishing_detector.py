"""
Lightweight Phishing Email Detection System
============================================
A memory-efficient application for detecting phishing emails using machine learning.
Designed for everyday users with minimal storage requirements.

Author: ML Security Team
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                            classification_report, precision_recall_fscore_support)
import json
import csv
from datetime import datetime
import os


class PhishingDetector:
    """Main class for phishing email detection and analysis."""
    
    def __init__(self, csv_file):
        """
        Initialize the detector with a dataset.
        
        Args:
            csv_file (str): Path to the phishing emails CSV file
        """
        self.csv_file = csv_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("📁 Loading dataset...")
        self.data = pd.read_csv(self.csv_file)
        print(f"✓ Loaded {len(self.data)} emails")
        print(f"✓ Features: {self.data.shape[1] - 1} columns")
        return self
    
    def prepare_data(self):
        """Split data into training (80%) and testing (20%) sets."""
        print("\n🔀 Splitting dataset (80% train, 20% test)...")
        
        # Separate features and target
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Testing set: {len(self.X_test)} samples")
        return self
    
    def train_linear_regression(self):
        """Train Linear Regression model with output rounding for classification."""
        print("\n🤖 Training Linear Regression model...")
        
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        
        # Predict and round to 0 or 1
        y_pred_continuous = lr_model.predict(self.X_test)
        y_pred = np.round(y_pred_continuous).astype(int)
        
        # Clip predictions to valid range [0, 1]
        y_pred = np.clip(y_pred, 0, 1)
        
        # Store model and predictions
        self.models['Linear Regression'] = lr_model
        self.predictions['Linear Regression'] = y_pred
        
        print("✓ Linear Regression trained successfully")
        return self
    
    def train_random_forest(self):
        """Train Random Forest Classifier model."""
        print("\n🌲 Training Random Forest Classifier...")
        
        # Train model with optimized parameters for efficiency
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = rf_model.predict(self.X_test)
        
        # Store model and predictions
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = y_pred
        
        print("✓ Random Forest trained successfully")
        return self
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics for both models."""
        print("\n📊 Calculating performance metrics...")
        
        for model_name, y_pred in self.predictions.items():
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='binary'
            )
            
            # Store all metrics
            self.metrics[model_name] = {
                'confusion_matrix': {
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                },
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'classification_report': classification_report(
                    self.y_test, y_pred, output_dict=True
                )
            }
        
        print("✓ Metrics calculated for all models")
        return self
    
    def display_results(self):
        """Display detailed results in console with visual formatting."""
        print("\n" + "="*70)
        print("📈 MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{'─'*70}")
            print(f"🔍 {model_name.upper()}")
            print(f"{'─'*70}")
            
            # Confusion Matrix
            cm = metrics['confusion_matrix']
            print("\n📋 Confusion Matrix:")
            print(f"   True Positives (TP):  {cm['true_positives']:4d}")
            print(f"   True Negatives (TN):  {cm['true_negatives']:4d}")
            print(f"   False Positives (FP): {cm['false_positives']:4d}")
            print(f"   False Negatives (FN): {cm['false_negatives']:4d}")
            
            # Key Metrics
            print(f"\n🎯 Key Metrics:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            
            # Visual accuracy bar
            self._print_ascii_bar("Accuracy", metrics['accuracy'])
            self._print_ascii_bar("Precision", metrics['precision'])
            self._print_ascii_bar("Recall", metrics['recall'])
        
        # Comparison
        self._print_model_comparison()
        
        return self
    
    def _print_ascii_bar(self, metric_name, value, width=40):
        """Print an ASCII bar chart for a metric."""
        filled = int(value * width)
        bar = "█" * filled + "░" * (width - filled)
        print(f"   {metric_name:10s} [{bar}] {value*100:.1f}%")
    
    def _print_model_comparison(self):
        """Print side-by-side comparison of models."""
        print(f"\n{'='*70}")
        print("🏆 MODEL COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.metrics.keys())
        
        # Header
        print(f"{'Metric':<15} {'Linear Regression':<25} {'Random Forest':<25}")
        print(f"{'-'*70}")
        
        # Comparison rows
        for metric in metrics_names:
            lr_val = self.metrics[model_names[0]][metric]
            rf_val = self.metrics[model_names[1]][metric]
            
            lr_str = f"{lr_val:.4f} ({lr_val*100:.2f}%)"
            rf_str = f"{rf_val:.4f} ({rf_val*100:.2f}%)"
            
            # Add winner indicator
            if lr_val > rf_val:
                lr_str += " 🏆"
            elif rf_val > lr_val:
                rf_str += " 🏆"
            
            print(f"{metric.upper():<15} {lr_str:<25} {rf_str:<25}")
    
    def save_predictions(self, output_file='predictions_compact.csv'):
        """
        Save predictions to a compact CSV file.
        Only includes essential columns: Email_ID and Predicted_Label for each model.
        """
        print(f"\n💾 Saving predictions to {output_file}...")
        
        # Create compact dataframe
        results = pd.DataFrame({
            'Email_ID': range(len(self.y_test)),
            'Actual_Label': self.y_test.values,
            'LR_Prediction': self.predictions['Linear Regression'],
            'RF_Prediction': self.predictions['Random Forest']
        })
        
        # Save to CSV
        results.to_csv(output_file, index=False)
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        print(f"✓ Predictions saved ({file_size:,} bytes)")
        
        return self
    
    def save_metrics_json(self, output_file='model_metrics.json'):
        """
        Save all metrics and summary statistics to a compact JSON file.
        """
        print(f"\n💾 Saving metrics to {output_file}...")
        
        # Prepare comprehensive report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(self.data),
                'training_samples': len(self.X_train),
                'testing_samples': len(self.X_test),
                'num_features': self.X_train.shape[1]
            },
            'models': self.metrics,
            'recommendations': self._generate_recommendations()
        }
        
        # Save to JSON with minimal formatting for smaller file size
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        print(f"✓ Metrics saved ({file_size:,} bytes)")
        
        return self
    
    def _generate_recommendations(self):
        """Generate recommendations based on model performance."""
        lr_acc = self.metrics['Linear Regression']['accuracy']
        rf_acc = self.metrics['Random Forest']['accuracy']
        
        recommendations = []
        
        if rf_acc > lr_acc:
            recommendations.append("Random Forest shows superior performance - recommended for deployment")
        else:
            recommendations.append("Linear Regression provides competitive results with faster inference")
        
        # Check for false positives (legitimate emails marked as phishing)
        lr_fp = self.metrics['Linear Regression']['confusion_matrix']['false_positives']
        rf_fp = self.metrics['Random Forest']['confusion_matrix']['false_positives']
        
        if lr_fp < rf_fp:
            recommendations.append("Linear Regression has fewer false positives (better for user experience)")
        elif rf_fp < lr_fp:
            recommendations.append("Random Forest has fewer false positives (better for user experience)")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Execute the complete phishing detection pipeline."""
        print("\n" + "="*70)
        print("🛡️  PHISHING EMAIL DETECTION SYSTEM")
        print("="*70)
        
        # Execute pipeline
        (self.load_data()
            .prepare_data()
            .train_linear_regression()
            .train_random_forest()
            .calculate_metrics()
            .display_results()
            .save_predictions()
            .save_metrics_json())
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\n📂 Output Files Generated:")
        print("   • predictions_compact.csv - Compact predictions file")
        print("   • model_metrics.json - Complete metrics and statistics")
        print("\n💡 Tip: Use the JSON file for quick analysis and reporting!")
        print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize and run the phishing detector
    detector = PhishingDetector('phishing mails.csv')
    detector.run_complete_analysis()
    
    print("🎉 Thank you for using the Phishing Detection System!")
    print("📧 Stay safe from phishing attacks!\n")
