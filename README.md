# Indian House Price Predictor

A machine learning web app that predicts house prices across major Indian cities based on property details like size, location, amenities, and more — built with **Python**, **scikit-learn**, and **Streamlit**, and deployed on **Hugging Face Spaces**.

🔗 **Live App:** [huggingface.co/spaces/Chetankumawat/Indian_house_price_predictor](https://huggingface.co/spaces/Chetankumawat/Indian_house_price_predictor)

---

## 📌 Overview

This project predicts residential property prices (in Lakhs ₹) using a **Linear Regression** model trained on real estate data from multiple Indian cities. Users can interactively input property details — city, size, BHK, furnishing status, amenities, and more — and instantly get a price estimate.

## ✨ Features

- Interactive form-based UI built with Streamlit
- Predicts price based on 14+ property features
- Supports multiple cities across India
- One-hot encoding, ordinal encoding, and multi-label binarization for categorical features
- MinMax-scaled inputs/outputs for consistent model performance
- Fully containerized with Docker and deployed on Hugging Face Spaces

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data preprocessing
- **Scikit-learn** – model training (Linear Regression, OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer, MinMaxScaler)
- **Streamlit** – interactive web UI
- **Docker** – containerized deployment
- **Hugging Face Spaces** – hosting

## 📊 Dataset

The model is trained on the **India Housing Prices** dataset, which includes features such as:

- City, Property Type, BHK, Size (SqFt)
- Floor Number, Age of Property
- Furnished Status, Parking, Security, Facing
- Nearby Schools/Hospitals, Public Transport Accessibility
- Amenities (Gym, Pool, Clubhouse, Garden, Playground)

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/Chetankumawat209/Indian-House-price-prediction.git
cd Indian-House-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                     # Streamlit application
├── house_price_bundle.pkl     # Trained model + preprocessing objects
├── Model_train.ipynb          # Notebook: data prep, training, evaluation
├── requirements.txt           # Python dependencies                
├──Indian_housing_prices.csv   # Original data set
├──Understand data.ipynb
└── README.md
```

## 🧠 Model Details

The trained bundle includes:
- The fitted `LinearRegression` model
- `OneHotEncoder` for City, Property Type, Parking, Security, Facing
- `OrdinalEncoder` for Furnished Status and Public Transport Accessibility
- `MultiLabelBinarizer` for Amenities
- `MinMaxScaler` for numeric features and the target variable

This ensures user input is transformed exactly the same way as the training data before prediction.

## 📈 Model Performance

- **R² Score:** ~0.87 on the test set

## 🙋‍♂️ Author

**Chetan Kumawat**
This was my first end-to-end ML deployment project — from data preprocessing and model training to building an interactive UI and deploying it live with Docker on Hugging Face Spaces.

---
If you found this project useful, consider giving it a star!
