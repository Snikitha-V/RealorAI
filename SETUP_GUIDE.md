# RealorAI Environment Setup Guide

## Quick Start

### 1. Virtual Environment is Ready
The virtual environment has been created at: `c:\Users\sniki\OneDrive\Desktop\RealorAI\venv`

### 2. Activate Virtual Environment (Windows PowerShell)
```powershell
Set-Location -Path 'c:\Users\sniki\OneDrive\Desktop\RealorAI'
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Once the virtual environment is activated, run:
```bash
pip install -r requirements.txt
```

This will install:
- **Data Processing**: numpy, pandas
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: tensorflow, keras
- **Computer Vision**: opencv-python, pillow
- **Jupyter**: jupyter, jupyterlab, ipykernel

### 4. Run Jupyter Notebook
```bash
jupyter notebook "ML Prj CIFAKE/CIFAKE_ML_Project.ipynb"
```

Or use JupyterLab (more feature-rich):
```bash
jupyter lab
```

### 5. Verify Installation
To verify all packages are installed, run:
```bash
pip list
```

## Environment Details

- **Location**: `c:\Users\sniki\OneDrive\Desktop\RealorAI\venv`
- **Python Version**: 3.13.5
- **Package Manager**: pip 25.2
- **Notebook Directory**: `ML Prj CIFAKE/`

## Troubleshooting

### If activation fails:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If pip install is slow:
```bash
pip install --upgrade pip
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

### To deactivate environment:
```bash
deactivate
```

---

**Status**: ✅ Virtual environment created and ready to use!
