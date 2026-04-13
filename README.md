
# AI-Based Skin Disease Detection

## 📌 Project Overview
This project uses deep learning to detect and classify different types of skin diseases from images using the HAM10000 dataset. It helps in early diagnosis and supports dermatological analysis.

---

## 🎯 Objective
To build an accurate image classification model using transfer learning and provide predictions through a simple and user-friendly web interface.

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Organized HAM10000 dataset using metadata
   - Converted images into class-wise folders

2. **Data Augmentation**
   - Applied transformations like rotation, zoom, and flipping using ImageDataGenerator

3. **Model Building**
   - Used EfficientNetB0 (Transfer Learning)
   - Added custom classification layers

4. **Training & Fine-Tuning**
   - Performed initial training with frozen base layers
   - Fine-tuned deeper layers to improve accuracy

5. **Prediction Interface**
   - Built using Streamlit for easy image upload and prediction

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas

---

## 📊 Dataset
- HAM10000 dataset
- Contains 7 classes of skin diseases

---

## 📁 Project Structure
- `skin_disease.ipynb` → Model training
- `main.py` → Streamlit web application
- `requirements.txt` → Required libraries

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run main.py
