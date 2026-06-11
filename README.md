# ChurnLens 🔍 - Customer Churn Predictor
### An interactive Machine Learning application designed to predict whether a customer is at high risk of churning (leaving a service) based on behavioral and financial patterns. This project was developed during Semester 1 of my Computer Science and Engineering (CSE) degree to demonstrate end-to-end implementation of classical classification algorithms—from raw data preprocessing to web UI deployment.

## 🌟 Project Overview
Customer retention is vital for businesses across subscription-based industries. ChurnLens utilizes a K-Nearest Neighbors (KNN) model trained on processed historic customer behavior patterns to evaluate real-time risks. It enables business managers to input consumer demographics and active service usage metrics to dynamically flag a consumer as either High Risk (Churn) or Low Risk (Stay).

## ✨ Key Features
* **End-to-End Pipeline**: Covers all stages of data science, including null value treatment, structural cleaning, feature mapping, feature normalization, model serialization, and interactive interface builds.

* **Exploratory Data Analysis (EDA)**: Visualizations examining core correlations, specifically highlighting how contract models and life tenures directly manipulate customer churn metrics.

* **Robust Feature Engineering**: Replicates categorical scaling using label encoders to convert custom criteria dynamically within active production scripts.

* **Web App UI**: Interactive web interface utilizing Gradio that allows human operators to quickly query individual records via visual sliders, numerical forms, and multi-option checkboxes.

## 🛠️ Tech Stack
* **Core Language**: Python

* **Data Handling**: Pandas, NumPy

* **Machine Learning Core**: Scikit-Learn

* **Data Serialization**: Joblib

* **Visualizations**: Matplotlib

* **Interactive UI Framework**: Gradio

## 📂 Project Directory Layout
```Plaintext
├── ChurnLens/
│   ├── app.py                     # Main Gradio application script for UI deployment
│   ├── Mini_Project(Sem_1).ipynb  # Jupyter notebook covering EDA, training, and evaluation
│   ├── churn_model.pkl            # Serialized trained KNN Classifier object
│   ├── scaler.pkl                 # Serialized fitted StandardScaler mapping criteria
│   ├── requirements.txt           # Python baseline dependencies file
│   └── README.md                  # Project repository documentation
```

## 📊 Dataset & Feature Description
The model evaluates a set of 10 strategic consumer inputs across individual profiles:

1. Age: Consumer's age group.

2. Gender: Binary classification (Male / Female).

3. Tenure (Months): Number of consecutive active months recorded.

4. Usage Frequency: Monthly index measuring app interactions/logins.

5. Support Calls: Count of technical support queries raised.

6. Payment Delay (Days): Total days an invoice went unpaid beyond standard terms.

7. Subscription Type: Active plan package tier (Basic, Standard, Premium).

8. Contract Length: Commitment structure duration (Monthly, Quarterly, Annual).

9. Total Spend: Total historic spending calculated over the customer lifecycle.

10. Last Interaction: Days since the customer's last recorded service interaction.

## ⚙️ Machine Learning Workflow
1. Preprocessing & Data Cleaning
* Dropped irrelevant lookup trackers (CustomerID) to isolate pure descriptive traits.

* Cleaned dirty fields and dropped rows containing NaN inputs.

* Applied categorical multi-class transformations using label maps to standard metrics (Gender, Subscription Type, Contract Length).

2. Normalization & Modeling
* Standardized data distributions via a fitted StandardScaler to prevent feature dominance during multi-dimensional distance calculations.

* Trained a K-Nearest Neighbors (KNN) classification algorithm to group customers based on distance vectors.

3. App Architecture
* Developed a backend parsing layer in app.py to intercept user UI interactions, evaluate them against active mappings, feed scales through scaler.pkl, and run the resulting arrays through churn_model.pkl.

## 🚀 Installation & Local Setup
Get a local instance running on your machine by following these steps:

Prerequisites
Make sure Python 3.8 or higher is installed.

Step 1: Clone the Repository
```Bash
git clone https://github.com/your-username/ChurnLens.git
cd ChurnLens
```

Step 2: Install Dependencies
Create a virtual environment (recommended) and install the required modules:
```Bash
# Optional: Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
## 🖥️ Running the App
```

Run the Gradio server to launch the interface locally:
```Bash
python app.py
```

Upon running, a local web server URL will be provided (usually http://127.0.0.1:7860). Open this address in your web browser to test and query individual profiles.

## 🤗 Deployment on Hugging Face Spaces
This project is optimized for deployment as a Hugging Face Space. To host your application:

* Create a new Space on Hugging Face with the Gradio SDK.

* Push your project files (app.py, requirements.txt, churn_model.pkl, scaler.pkl) directly to the Space repository.

* Hugging Face will automatically handle the environment setup and host your web dashboard permanently for public access.

## 📄 License
Distributed under the MIT License. See the MIT License guidelines for more details.
