# 🌱 Smart Crop Health Analyzer

## Overview

Smart Crop Health Analyzer is a deep learning-based web application that helps farmers, researchers, and agricultural professionals identify nutrient deficiencies and predict crop growth stages from leaf images.

The application currently supports three major crops:

- 🌽 Maize
- 🌾 Rice
- ☕ Coffee

Users can choose between:

1. **Nutrient Deficiency Detection**
2. **Growth Stage Prediction**

After selecting a task, users upload a crop image and choose the crop type. The model then analyzes the image and provides predictions along with confidence scores and visual explanations.

---

## Features

### 🔬 Nutrient Deficiency Detection

Supported classes:

- Healthy
- Nitrogen Deficiency
- Phosphorus Deficiency
- Potassium Deficiency

Additional Features:

- Confidence Score
- Grad-CAM Visual Explanation
- Fertilizer Recommendations
- Deficiency Reasoning
- Suggested Corrective Actions

### 🌿 Growth Stage Prediction

Supported classes:

- Seedling Stage
- Vegetative Stage
- Flowering Stage

---

## Dataset Collection

One of the biggest challenges during this project was obtaining a balanced and sufficiently large dataset.

### Data Sources

- Publicly available agricultural datasets
- Manual web scraping
- Data augmentation

Since complete datasets for all crops and classes were not available, we manually collected and web-scraped approximately **250 unique images**.

To overcome data scarcity and class imbalance, extensive augmentation techniques were applied, including:

- Rotation
- Flipping
- Zooming
- Brightness adjustments
- Various image transformations

This resulted in approximately:

**1000 images per class**

which significantly improved model performance and generalization.

---

## Model Development

Two separate deep learning models were developed:

### 1. Nutrient Deficiency Model

**Architecture:** EfficientNetV2-S

Classes:

- Healthy
- Nitrogen
- Phosphorus
- Potassium

### 2. Growth Stage Model

**Architecture:** EfficientNetV2-S

Classes:

- Seedling
- Vegetative
- Flowering

---

## Training Experiments

### Initial Training (224×224 Images)

| Model | Accuracy |
|---------|---------|
| Growth Stage | 82.64% |
| Nutrient Deficiency | 76% |

### Improved Training (300×300 Images)

Changes made:

- Increased image size to 300×300
- Removed manual normalization since EfficientNetV2-S performs preprocessing internally

Results:

| Model | Accuracy |
|---------|---------|
| Nutrient Deficiency | 98% |
| Growth Stage | 74% |

### Final Model Selection

After experimentation, the best-performing configuration was:

#### Nutrient Deficiency Model

- EfficientNetV2-S
- Input Size: 300×300
- Accuracy: 98%

#### Growth Stage Model

- EfficientNetV2-S
- Input Size: 224×224
- Accuracy: 82.64%

This hybrid setup produced the best overall results.

---

## Crop Metadata Integration

Besides image inputs, crop type information is also provided to the model.

Supported crop types:

- Rice
- Maize
- Coffee

Including crop metadata helps the model learn crop-specific growth patterns and deficiency characteristics, improving prediction accuracy.

---

## Explainable AI (XAI)

The nutrient deficiency model incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM highlights the regions of the leaf that influenced the model's decision, allowing users to understand:

- Why a prediction was made
- Which leaf regions contributed most to the diagnosis
- Whether the prediction aligns with visible symptoms

---

## Research Contribution

A research paper based on this project was submitted to an Indian Data Science conference and was **accepted for presentation**.

However, due to the conference presentation fee of **₹12,000**, the paper was not presented.

The acceptance nevertheless validates the technical and research contributions of this work.

---

## Model Files

The trained models are too large to be stored directly in this GitHub repository.

Download the models from:

**Google Drive:**  
https://drive.google.com/drive/folders/1i0kvuDpCBKfIqpGPvJPfLbYKPpQkuYQl?usp=sharing

After downloading, place the model files in the project root directory.

Expected files:

```text
deficiency_300px_autosave.keras
best_growth_fusion_model_v2.keras
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Dependencies

- Streamlit
- TensorFlow
- NumPy
- Pillow
- OpenCV

---

## Application Workflow

1. Select a task:
   - Growth Stage Prediction
   - Nutrient Deficiency Detection

2. Select crop type:
   - Rice
   - Maize
   - Coffee

3. Upload a crop image.

4. Click Predict.

5. View:
   - Prediction
   - Confidence Score
   - Visual Explanation (for deficiency prediction)
   - Recommendations and fertilizer suggestions

---

## Future Improvements

- Support additional crop species
- Detect more nutrient deficiencies
- Mobile deployment
- Real-time field diagnosis
- Disease detection integration
- Multi-deficiency prediction

---

## Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetV2-S
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## Authors

Developed as a research-driven project focused on applying:

- Deep Learning
- Computer Vision
- Explainable AI (XAI)
- Precision Agriculture

to create an intelligent crop health monitoring system capable of assisting farmers and agricultural researchers in making data-driven decisions.
