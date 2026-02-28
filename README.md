# Diabetic Retinopathy Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
Advanced deep learning system for early detection of Diabetic Retinopathy from retinal fundus images.

## Architecture
- **Backend**: TensorFlow 2.13 with MobileNetV2 transfer learning
- **Frontend**: Streamlit interactive web interface
- **Model**: Custom CNN with transfer learning
- **Dataset**: Synthetic dataset of 10,000 retinal images

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detection
cd diabetic-retinopathy-detection
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Run the app
```bash
streamlit run app.py
```

## Severity Levels
| Level | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy detected |
| 1 | Mild NPDR | Mild non-proliferative diabetic retinopathy |
| 2 | Moderate NPDR | Moderate non-proliferative diabetic retinopathy |
| 3 | Severe NPDR | Severe non-proliferative diabetic retinopathy |
| 4 | Proliferative DR | Proliferative diabetic retinopathy |

## Disclaimer
This is an AI-assisted screening tool. Always consult with healthcare professionals for medical decisions.
