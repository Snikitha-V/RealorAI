# RealorAI - CIFAKE Detection

A comprehensive machine learning project for detecting AI-generated (fake) images vs. real images using the CIFAKE dataset.

## 📊 Project Overview

This project compares **9 different machine learning and deep learning models** to classify images as either:
- **REAL**: Authentic photographs
- **FAKE**: AI-generated images

### Dataset Statistics
- **Total Images**: 120,000
  - **Training Set**: 100,000 images (50,000 REAL + 50,000 FAKE)
  - **Test Set**: 20,000 images (10,000 REAL + 10,000 FAKE)

## 🤖 Models Implemented

1. **Logistic Regression** - Baseline linear classifier
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Support Vector Machine (SVM)** - RBF kernel classifier
4. **Naive Bayes** - Probabilistic classifier
5. **Decision Tree** - Tree-based classifier (max_depth=10)
6. **Random Forest** - Ensemble of 100 decision trees
7. **XGBoost** - Gradient boosting classifier
8. **AdaBoost** - Adaptive boosting ensemble
9. **Multi-Layer Perceptron (MLP)** - Neural network with TensorFlow/Keras

## 📁 Project Structure

```
RealorAI/
├── README.md
├── .gitignore
├── ML Prj CIFAKE/
│   ├── CIFAKE_ML_Project.ipynb       # Main notebook with all models
│   ├── .venv/                         # Python virtual environment
│   └── data/
│       └── CIFAKE/
│           ├── train/
│           │   ├── REAL/              # 50,000 real images
│           │   └── FAKE/              # 50,000 fake images
│           └── test/
│               ├── REAL/              # 10,000 real images
│               └── FAKE/              # 10,000 fake images
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Snikitha-V/RealorAI.git
   cd RealorAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook "ML Prj CIFAKE/CIFAKE_ML_Project.ipynb"
   ```

## 📦 Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- opencv-python (cv2)
- pillow
- xgboost

## 📊 Model Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🔄 Workflow

1. **Data Loading**: Images are loaded from the CIFAKE dataset structure
2. **Preprocessing**: 
   - Pixel normalization (0-1 range)
   - Image flattening for traditional ML models
   - Train-test split with stratification
3. **Training**: Models are trained on the training set
4. **Evaluation**: Comprehensive metrics calculated and compared
5. **Comparison**: Results aggregated in a comparative DataFrame

## 💡 Key Findings

The notebook generates a results comparison table showing performance metrics for all 9 models, helping identify which approach works best for real vs. fake image detection.

## 🔧 Customization

You can modify the following in the notebook:
- KNN k-values: Change `k_values = [3, 5, 7, 9, 11]`
- Decision Tree depth: Modify `max_depth=10`
- Random Forest estimators: Change `n_estimators=100`
- Neural network architecture: Edit layers in MLP model
- Training epochs: Adjust `epochs=50`

## 📝 Notes

- The project uses image flattening for traditional ML algorithms (32×32×3 = 3,072 features)
- Neural networks use the original image shape for better feature extraction
- Cross-validation can be added for more robust evaluation

## 🤝 Contributing

Feel free to:
- Suggest improvements to model architectures
- Add new classification models
- Optimize hyperparameters
- Enhance data preprocessing

## 📄 License

MIT License - feel free to use this project for educational and research purposes.

## 👤 Author

Snikitha V

## 📞 Support

For issues or questions, please open a GitHub issue or contact the maintainer.

---

**Last Updated**: October 23, 2025
