# 🛡️ Phishing Email Detection System

A lightweight, memory-efficient machine learning application for detecting phishing emails. Designed for everyday users with minimal storage requirements and user-friendly reporting.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Output Files](#output-files)
- [Performance Metrics](#performance-metrics)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Dual Model Comparison**: Linear Regression and Random Forest Classifier
- **Memory Efficient**: Optimized for minimal storage and fast execution
- **Comprehensive Metrics**: Confusion matrix, accuracy, precision, recall, F1-score
- **Visual Console Output**: ASCII bar charts and side-by-side model comparison
- **Compact Storage**: 
  - Predictions saved in minimal CSV format
  - Complete metrics in small JSON file
- **User-Friendly**: Clear visualizations and actionable recommendations
- **Production Ready**: Clean, well-documented, fully-commented code

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

1. Place your dataset file named `phishing mails.csv` in the project directory
2. Run the detection system:

```bash
python phishing_detector.py
```

### Custom Dataset Path

Modify the main execution block in `phishing_detector.py`:

```python
detector = PhishingDetector('path/to/your/dataset.csv')
detector.run_complete_analysis()
```

## 📊 Dataset Format

Your CSV file should have the following structure:

| Feature_1 | Feature_2 | ... | Feature_N | Label |
|-----------|-----------|-----|-----------|-------|
| value     | value     | ... | value     | 0/1   |

**Requirements:**
- Last column must be named `Label`
- Label values: `1` = phishing, `0` = legitimate
- All other columns are treated as features
- No missing values (preprocess if needed)

### Example CSV Structure

```csv
Feature1,Feature2,Feature3,Feature4,Label
0.5,1.2,0.8,2.1,1
1.1,0.3,1.5,0.9,0
0.2,2.1,0.4,1.8,1
```

## 📁 Output Files

The system generates two compact output files:

### 1. `predictions_compact.csv`
Minimal CSV with essential prediction data:
```csv
Email_ID,Actual_Label,LR_Prediction,RF_Prediction
0,1,1,1
1,0,0,0
2,1,0,1
```

### 2. `model_metrics.json`
Complete metrics and statistics:
```json
{
  "timestamp": "2025-12-06 10:30:45",
  "dataset_info": {
    "total_samples": 1000,
    "training_samples": 800,
    "testing_samples": 200
  },
  "models": {
    "Linear Regression": { ... },
    "Random Forest": { ... }
  },
  "recommendations": [ ... ]
}
```

## 📈 Performance Metrics

For each model, the system calculates:

- **Confusion Matrix Components**
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)

- **Performance Scores**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

- **Classification Report**
  - Per-class precision, recall, F1-score
  - Support values

## 🖥️ Example Output

```
======================================================================
🛡️  PHISHING EMAIL DETECTION SYSTEM
======================================================================

📁 Loading dataset...
✓ Loaded 1000 emails
✓ Features: 18 columns

🔀 Splitting dataset (80% train, 20% test)...
✓ Training set: 800 samples
✓ Testing set: 200 samples

🤖 Training Linear Regression model...
✓ Linear Regression trained successfully

🌲 Training Random Forest Classifier...
✓ Random Forest trained successfully

📊 Calculating performance metrics...
✓ Metrics calculated for all models

======================================================================
📈 MODEL PERFORMANCE COMPARISON
======================================================================

──────────────────────────────────────────────────────────────────────
🔍 LINEAR REGRESSION
──────────────────────────────────────────────────────────────────────

📋 Confusion Matrix:
   True Positives (TP):   85
   True Negatives (TN):   90
   False Positives (FP):  10
   False Negatives (FN):  15

🎯 Key Metrics:
   Accuracy:  0.8750 (87.50%)
   Precision: 0.8947
   Recall:    0.8500
   F1-Score:  0.8718
   
   Accuracy   [████████████████████████████████████░░░░] 87.5%
   Precision  [█████████████████████████████████████░░░] 89.5%
   Recall     [██████████████████████████████████░░░░░░] 85.0%

======================================================================
🏆 MODEL COMPARISON SUMMARY
======================================================================

Metric          Linear Regression         Random Forest            
----------------------------------------------------------------------
ACCURACY        0.8750 (87.50%)           0.9250 (92.50%) 🏆
PRECISION       0.8947 (89.47%)           0.9500 (95.00%) 🏆
RECALL          0.8500 (85.00%)           0.9000 (90.00%) 🏆
F1_SCORE        0.8718 (87.18%)           0.9247 (92.47%) 🏆

💾 Saving predictions to predictions_compact.csv...
✓ Predictions saved (5,432 bytes)

💾 Saving metrics to model_metrics.json...
✓ Metrics saved (3,891 bytes)

======================================================================
✅ ANALYSIS COMPLETE
======================================================================

📂 Output Files Generated:
   • predictions_compact.csv - Compact predictions file
   • model_metrics.json - Complete metrics and statistics

💡 Tip: Use the JSON file for quick analysis and reporting!
======================================================================
```

## 🛠️ Project Structure

```
phishing-detection/
│
├── phishing_detector.py      # Main application script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
│
├── data/                      # (Optional) Store datasets here
│   └── phishing mails.csv
│
└── output/                    # (Optional) Store results here
    ├── predictions_compact.csv
    └── model_metrics.json
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- Add support for more ML models (SVM, Neural Networks)
- Implement feature importance visualization
- Add email preprocessing utilities
- Create a GUI interface
- Add real-time email scanning capability

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work*

## 🙏 Acknowledgments

- Built with scikit-learn
- Inspired by the need for accessible cybersecurity tools
- Thanks to the open-source community

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

**Note**: This tool is designed for educational and research purposes. Always verify phishing detection results with your organization's security team.