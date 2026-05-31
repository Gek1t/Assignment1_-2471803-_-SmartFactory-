

# Smart Factory Predictive Maintenance System (NASA C-MAPSS FD002)

**Module**: IOT106TC Big Data Analytics 

**Assignment**: Assignment 2: Predictive Modeling & Interactive Dashboard 

**Student ID**: 2471803 

**Institution**: School of Internet of Things, XJTLU Entrepreneur College (Taicang) 

---

## 1. Project Architecture & Pipeline

This production-ready system delivers an end-to-end Prognostic and Health Management (PHM) solution for predicting the Remaining Useful Life (RUL) of turbofan engines operating under **highly complex, multi-regime conditions (NASA FD002 dataset)**. 

### Core Machine Learning Pipeline

1. **Multi-Regime Normalization**: To combat the 6 distinct operational settings in FD002, the pipeline applies **K-Means Clustering** on operational settings to identify regimes, followed by **Regime-Specific Z-Score Scaling** to decouple environmental noise from degradation signals.
2. 
**Feature Engineering**: Dynamic rolling statistics (moving average, standard deviation) computed per engine instance, supplemented by sensor degradation trend analysis. 


3. 
**Weighted Ensemble Model**: A calibrated hybrid architecture blending **XGBoost Regressor** and **Random Forest Regressor** to optimize RUL predictions under high-variance constraints. 



---

## 2. Directory Structure

The project strictly follows the standard submission taxonomy required by the `IOT106TC` specification:

```text
IOT106TC_Assignment2_2471803/
├── README.md                  # Project documentation & execution guide
├── requirements.txt           # Verified Python dependency manifest
├── data/
│   └── SmartFactory_FD002_features.csv  # Preprocessed feature matrix from Assignment 1
├── models/                    # Serialized pipeline and model artifacts
│   ├── final_model.pkl        # Trained Weighted Ensemble model
│   ├── scaler.pkl             # Global/Regime scaler object
│   └── selected_features.pkl  # Serialized list of optimized features
├── notebooks/                 # Step-by-step development sequence
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
└── dashboard/                 # Mandatory UI directory
    └── app.py                 # Streamlit SCADA Dashboard source code

```

---

## 3. Local Installation & Environment Setup

### Step 1: Open Terminal and Navigate to Project Root

You must execute all subsequent commands from the **project root directory** (`IOT106TC_Assignment2_2471803`). Choose **ONE** of the following methods based on your setup to open the terminal at the correct location:

* **Method A: Windows (File Explorer Address Bar - Easiest)**
1. Open Windows File Explorer and double-click into your unzipped project folder `IOT106TC_Assignment2_2471803` (where you can see the `data/` and `models/` folders).
2. Click on the **empty space of the address bar** at the top of the window. The path text will be highlighted.
3. Type **`cmd`** and press **Enter**. A black command prompt window will pop up, *automatically located in the root directory*.


* **Method B: VS Code (Integrated Terminal)**
1. Open the project folder `IOT106TC_Assignment2_2471803` directly using VS Code.
2. Click the top menu bar: **Terminal** -> **New Terminal** (终端 -> 新建终端).



---

### Step 2: Install Core Dependencies (Global Environment)

Directly install all required data science and dashboard environment packages into your global Python environment using the system launcher:

```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

```

---

### Step 3: ⚠️ CRITICAL PRE-REQUISITE (Pipeline Generation)

> **IMPORTANT**: The dashboard relies on serialized model artifacts and engineered datasets. You **MUST** open your Jupyter environment and run the three pipeline notebooks in the `notebooks/` directory sequentially **BEFORE** attempting to launch the dashboard.
> Running these files will populate the required files in `data/` and `models/`:
> 1. Run `01_feature_engineering.ipynb` to output the engineered feature set.
> 2. Run `02_model_training.ipynb` to train and serialize the models and scalers.
> 3. Run `03_model_evaluation.ipynb` to evaluate and generate error metrics.
> 
> 

---

## 4. Running the Dashboard (Fixed Execution Path)

> ⚠️ **CRITICAL EXECUTION NOTE**: Running `streamlit run app.py` from the root will fail because the course syllabus strictly requires the application code to reside inside the `dashboard/` directory. You must invoke the script via its relative path from the project root directory using the system launcher so that it can correctly discover and read the dataset and model artifacts (`../data/` or `../models/`).

Execute this command exactly from the **project root directory** (which you opened in Step 1):

```bash
py -m streamlit run dashboard/app.py

```

### Accessing the Web UI

Once the terminal outputs the network status, open your web browser and navigate to:

* **Local URL**: `http://localhost:8501`

---

## 5. Dashboard Key Features & SCADA Interface

The interactive Streamlit interface acts as a full-scale Supervisory Control and Data Acquisition (SCADA) twin tailored for Fleet Managers and Financial Directors:

* **Fleet Alert Priority Queue (APQ)**: An automated risk-triage engine that instantly bubbles up high-risk assets (Engines with RUL < 30 cycles) into a red-flagged critical maintenance queue.
* **Real-Time ROI Quantification**: A dynamic financial module converting predictive accuracy into business value, computing net downtime cost avoidance ($) based on early scheduling vs. catastrophic failure mitigation.
* **Dynamic Explainable AI (XAI)**: Integrated **SHAP (Shapley Additive exPlanations)** visualization panels, translating dense ensemble tree decisions into human-readable sensor drill-downs for non-technical stakeholders.

---

## 6. Engineering Dependencies Manifest

The pipeline utilizes verified versions of the following analytics stack (specified in `requirements.txt`):

* **Core Analytics**: `pandas >= 2.0.0`, `numpy >= 1.20.0`
* **Machine Learning**: `scikit-learn >= 1.2.0`, `xgboost >= 1.7.0`
* **Explainability & Storage**: `shap >= 0.41.0`, `joblib >= 1.2.0`
* **Visualization Interface**: `streamlit >= 1.22.0`, `plotly >= 5.14.0`, `matplotlib`, `seaborn`

---

## 7. Known Boundary Conditions & Limitations

* **Regime Invariance**: The pipeline's preprocessing block is strictly calibrated via K-Means to the 6 operational settings of NASA's C-MAPSS FD002 sub-dataset. Introducing different datasets (e.g., single-regime FD001) requires disabling the clustering loop.
* **Memory & Caching Constraint**: To load extensive fleet histories swiftly under 10 seconds, the application leverages `st.cache_resource`. If running on resource-constrained local environments, clear the browser cache if multi-engine logs lag.

---

*Developed strictly in accordance with LMO Academic Integrity Guidelines for IOT106TC, XJTLU Entrepreneur College (Taicang).*