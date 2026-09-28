# Placement Prediction Model

A machine learning project that predicts student placement outcomes using various classification algorithms. This project uses student academic and assessment data to build predictive models.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Author & Contact](#author--contact)

## 🎯 Features

- **Multiple ML Algorithms**: Implements Random Forest, Gradient Boosting, KNN, and Logistic Regression
- **Data Preprocessing**: Handles missing values and feature scaling
- **Model Evaluation**: Comprehensive metrics including Accuracy, Precision, Recall, and F1-Score
- **Interactive Dashboard**: Streamlit-based web interface for visualization
- **Model Persistence**: Saves trained models for production deployment
- **Dynamic Benchmark Loading**: Loads evaluation metrics from JSON for real-time updates

## 📁 Project Structure

```
placement-prediction/
├── data/
│   └── student_placement_data.csv       # Training dataset
├── models/
│   ├── best_pipeline.pkl                # Serialized best model
│   └── benchmark_results.json           # Model evaluation metrics
├── train.py                             # Model training script
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/stutikatiyar/placement-prediction.git
cd placement-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

Run the training script to build and evaluate models:
```bash
python train.py
```

This will:
- Load and preprocess the student data
- Train multiple classification models
- Evaluate each model's performance
- Save the best model and benchmark metrics to JSON
- Display results in the console

### Running the Web Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## 📊 Model Evaluation

The project evaluates models using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of correctly identified positive cases
- **F1-Score**: Harmonic mean of precision and recall

Benchmark results are saved dynamically to `models/benchmark_results.json` and displayed in the web interface.

## 🛠 Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **Streamlit**: Interactive web dashboard
- **joblib**: Model serialization
- **numpy**: Numerical computations

## 👩‍💻 Author & Contact

* **Developers**:  Sujal Ranjan & Stuti Katiyar
* **Repository**: [https://github.com/stutikatiyar/placement-prediction](https://github.com/stutikatiyar/placement-prediction?utm_source=gemini)

