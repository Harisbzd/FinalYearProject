# Diabetes Prediction System - Final Year Project

## Professor's Technical Overview

**Student:** Haris Behzad  
**Project Type:** Final Year Project (FYP)  
**Domain:** Healthcare Informatics & Machine Learning  
**Technology Stack:** React.js, Flask, Python ML Libraries, MongoDB Atlas

---

## 🎯 Project Overview & Academic Context

This project implements a comprehensive **web-based diabetes prediction system** that demonstrates the practical application of machine learning algorithms in healthcare informatics. The system serves as a proof-of-concept for how modern web technologies can be integrated with machine learning models to create accessible healthcare tools.

### **Academic Objectives Achieved:**
- Implementation of multiple machine learning algorithms for comparative analysis
- Development of a full-stack web application with modern technologies
- Integration of cloud-based database systems for scalability
- Application of healthcare data science principles using real-world datasets
- Implementation of proper software engineering practices and API design

## 🏗️ System Architecture & Technical Implementation

### **High-Level Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React.js      │    │   Flask API     │    │   MongoDB       │
│   Frontend      │◄──►│   Backend       │◄──►│   Atlas Cloud   │
│   (Port 3000)   │    │   (Port 5001)   │    │   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │   ML Models     │
│   - Dashboard   │    │   - XGBoost     │
│   - Forms       │    │   - Random Forest│
│   - Analytics   │    │   - KNN         │
│   - Auth        │    │   - Logistic Reg│
└─────────────────┘    └─────────────────┘
```

### **Machine Learning Implementation**

#### **1. Data Source & Preprocessing**
- **Dataset**: BRFSS (Behavioral Risk Factor Surveillance System) 2015
- **Features**: 21 health indicators including BMI, blood pressure, cholesterol, lifestyle factors
- **Data Engineering**: Feature scaling, handling missing values, categorical encoding
- **Class Imbalance**: Addressed using SMOTE (Synthetic Minority Oversampling Technique)

#### **2. Model Selection & Rationale**
- **Logistic Regression**: Baseline model for interpretability and statistical significance
- **XGBoost**: Advanced gradient boosting for high accuracy and feature importance
- **Random Forest**: Ensemble method for robustness and overfitting prevention
- **K-Nearest Neighbors**: Instance-based learning for non-parametric approach

#### **3. Model Training & Validation**
- **Cross-validation**: 5-fold cross-validation for robust performance estimation
- **Hyperparameter Tuning**: Grid search optimization for each algorithm
- **Model Persistence**: All models saved using joblib for production deployment
- **Performance Tracking**: Comprehensive metrics stored for comparison

### **Technical Features**

#### **Backend (Flask API)**
- **RESTful API Design**: Clean separation of concerns with dedicated endpoints
- **Authentication**: JWT-based secure user authentication
- **Model Serving**: Real-time prediction endpoints for all ML models
- **Data Validation**: Input validation and error handling
- **Cloud Integration**: MongoDB Atlas for scalable data storage

#### **Frontend (React.js)**
- **Modern UI/UX**: Responsive design with Tailwind CSS
- **State Management**: React Context for authentication and data flow
- **Real-time Predictions**: Dynamic form handling with instant results
- **Visual Analytics**: Interactive charts and model performance metrics
- **Progressive Web App**: Build optimization for production deployment

## 🛠️ Technology Stack

### Frontend
- **React 18**: Modern UI framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Router**: Client-side routing

### Backend
- **Flask**: Python web framework
- **MongoDB**: NoSQL database for user data
- **JWT**: JSON Web Tokens for authentication

### Machine Learning
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting framework
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Joblib**: Model serialization

## 📊 Model Performance & Evaluation Methodology

### **Comprehensive Model Comparison**

| Model | Accuracy | AUC Score | Sensitivity | Specificity | F1-Score |
|-------|----------|-----------|-------------|-------------|----------|
| **XGBoost** | 75.3% | 82.7% | 79.4% | 71.2% | 0.76 |
| **Random Forest** | 74.8% | 82.1% | 78.1% | 71.5% | 0.75 |
| **Logistic Regression** | 74.6% | 82.3% | 76.8% | 72.5% | 0.74 |
| **K-Nearest Neighbors** | 73.2% | 80.9% | 75.3% | 71.1% | 0.73 |

### **Evaluation Methodology**
- **Dataset Split**: 80% training, 20% testing with stratified sampling
- **Cross-Validation**: 5-fold cross-validation for robust performance estimation
- **Metrics Selection**: 
  - **Accuracy**: Overall correctness for general performance
  - **AUC-ROC**: Area under curve for binary classification performance
  - **Sensitivity**: True positive rate (crucial for medical diagnosis)
  - **Specificity**: True negative rate (reducing false alarms)
- **Statistical Significance**: Paired t-tests for model comparison
- **Feature Importance**: SHAP values for model interpretability

### **Key Findings**
1. **XGBoost** emerged as the best-performing model with highest accuracy and AUC
2. **Logistic Regression** provides excellent interpretability with competitive performance
3. **Random Forest** offers good balance between accuracy and robustness
4. **KNN** serves as a baseline non-parametric approach

### **Clinical Relevance**
- **Sensitivity > 75%**: Ensures most diabetes cases are detected
- **Specificity > 70%**: Minimizes false positive diagnoses
- **AUC > 80%**: Indicates strong discriminative ability
- **Overall Performance**: Suitable for screening and risk assessment

## 🏗️ Project Structure & Code Organization

```
FYP2-main/
├── client/                          # React.js Frontend Application
│   ├── src/
│   │   ├── components/              # Reusable UI components
│   │   │   ├── Dashboard.tsx        # Main prediction interface
│   │   │   ├── PredictionForm.tsx   # Health data input form
│   │   │   ├── ModelComparison.tsx  # Side-by-side model results
│   │   │   └── Analytics.tsx        # Performance visualizations
│   │   ├── context/                 # React Context for state management
│   │   │   └── AuthContext.tsx      # Authentication state
│   │   ├── routes/                  # Application routing
│   │   └── data/                    # Static data and configurations
│   ├── public/                      # Static assets and ML visualizations
│   │   ├── *.png                    # Model performance charts
│   │   └── *.txt                    # Classification reports
│   └── build/                       # Production build output
├── flask-server/                    # Flask Backend API
│   ├── ml_models/                   # Machine Learning Implementation
│   │   ├── xgboost_model.py         # XGBoost implementation
│   │   ├── randomforest_model.py    # Random Forest implementation
│   │   ├── knn_model.py             # K-Nearest Neighbors
│   │   ├── logistic_regression.py   # Logistic Regression
│   │   ├── model_utils.py           # Shared ML utilities
│   │   └── feature_engineering.py   # Data preprocessing
│   ├── auth/                        # Authentication System
│   │   ├── routes.py                # Auth API endpoints
│   │   ├── jwt_utils.py             # JWT token management
│   │   └── utils.py                 # Password hashing utilities
│   ├── *.joblib                     # Trained ML models (serialized)
│   ├── *.txt                        # Model performance metrics
│   ├── *.npy                        # Feature column definitions
│   ├── server.py                    # Main Flask application
│   ├── requirements.txt             # Python dependencies
│   └── .env                         # Environment configuration
└── README.md                        # Project documentation
```

### **Code Architecture Principles**
- **Separation of Concerns**: Clear separation between frontend, backend, and ML components
- **Modular Design**: Each ML model is independently implemented and testable
- **API-First Approach**: RESTful API design for scalability and maintainability
- **Configuration Management**: Environment-based configuration for different deployment stages
- **Error Handling**: Comprehensive error handling and logging throughout the application

## 🚀 System Deployment & Usage Instructions

### **Prerequisites for Professor/Evaluator**
- **Node.js** (v16 or higher) - For React frontend
- **Python** (v3.8 or higher) - For Flask backend and ML models
- **Git** - For repository cloning
- **Internet Connection** - For MongoDB Atlas cloud database

### **Quick Setup (5 Minutes)**

1. **Clone Repository**
   ```bash
   git clone https://github.com/Harisbzd/FinalYearProject.git
   cd FinalYearProject
   ```

2. **Backend Setup**
   ```bash
   cd flask-server
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd ../client
   npm install
   ```

### **Running the Application**

**Terminal 1 - Start Backend Server:**
```bash
cd flask-server
source venv/bin/activate
python server.py
```
*Server runs on: http://localhost:5001*

**Terminal 2 - Start Frontend:**
```bash
cd client
npm start
```
*Application opens at: http://localhost:3000*

### **System Access**
- **Web Application**: http://localhost:3000
- **API Documentation**: http://localhost:5001 (Flask endpoints)
- **Database**: MongoDB Atlas (cloud-hosted, no local setup required)

### **Testing the System**
1. **User Registration**: Create a new account
2. **Health Data Input**: Fill the prediction form with health indicators
3. **Model Comparison**: Test all 4 ML models simultaneously
4. **Analytics Dashboard**: View performance metrics and visualizations

## 📈 Model Training & Development Process

### **Model Training Pipeline**
The system includes a complete ML pipeline for model development and evaluation:

```bash
cd flask-server
source venv/bin/activate

# Individual model training
python ml_models/logistic_regression.py    # Baseline model
python ml_models/xgboost_model.py          # Best performing model
python ml_models/randomforest_model.py     # Ensemble method
python ml_models/knn_model.py              # Instance-based learning

# Feature engineering and preprocessing
python ml_models/feature_engineering.py    # Data preprocessing pipeline
```

### **Development Methodology**
1. **Data Exploration**: Comprehensive EDA using Jupyter notebooks
2. **Feature Engineering**: Automated preprocessing pipeline
3. **Model Selection**: Systematic comparison of algorithms
4. **Hyperparameter Tuning**: Grid search optimization
5. **Model Validation**: Cross-validation and performance metrics
6. **Production Deployment**: Model serialization and API integration

## 🔧 System Configuration & Technical Details

### **Environment Configuration**
The system uses environment-based configuration for different deployment stages:

**Production Configuration (`.env` file):**
```
MONGO_URI=mongodb+srv://harisbehzad00:Harisk12239%40@diabeties.24eh2gm.mongodb.net/diabeties?retryWrites=true&w=majority&appName=Diabeties
JWT_SECRET_KEY=your_secret_key_here
FLASK_ENV=production
```

### **API Endpoints Documentation**
The Flask backend provides comprehensive REST API endpoints:

- `POST /api/predict-diabetes` - Main prediction endpoint (XGBoost)
- `POST /api/predict-diabetes-xgboost` - XGBoost-specific predictions
- `POST /api/predict-diabetes-randomforest` - Random Forest predictions
- `POST /api/predict-diabetes-knn` - KNN predictions
- `POST /api/predict-diabetes-logreg` - Logistic Regression predictions
- `GET /api/predict-diabetes` - Retrieve user's prediction history
- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication

### **Model Persistence & Loading**
- **Model Files**: All trained models saved as `.joblib` files
- **Performance Metrics**: Stored as `.txt` files for quick loading
- **Feature Columns**: Saved as `.npy` files for consistent preprocessing
- **Scalers**: Preprocessing objects saved for production use

## 📊 Dataset & Data Science Methodology

### **Dataset Information**
- **Source**: BRFSS (Behavioral Risk Factor Surveillance System) 2015
- **Size**: 253,680 records with 21 features
- **Target Variable**: Binary diabetes classification (0: No Diabetes, 1: Diabetes)
- **Data Quality**: Preprocessed and cleaned for machine learning

### **Feature Engineering**
The system includes 21 health indicators organized into categories:

**Demographic Features:**
- Age, Sex, Education, Income

**Medical History:**
- High Blood Pressure, High Cholesterol, Cholesterol Check
- Heart Disease/Attack, Stroke

**Lifestyle Factors:**
- BMI (Body Mass Index), Smoking Status
- Physical Activity, Heavy Alcohol Consumption
- Fruit and Vegetable Consumption

**Health Status:**
- General Health, Mental Health, Physical Health
- Difficulty Walking

**Healthcare Access:**
- Any Healthcare, No Doctor Due to Cost

### **Data Preprocessing Pipeline**
1. **Missing Value Handling**: Imputation strategies for incomplete records
2. **Feature Scaling**: Standardization for algorithms requiring normalized inputs
3. **Categorical Encoding**: One-hot encoding for categorical variables
4. **Class Imbalance**: SMOTE (Synthetic Minority Oversampling) for balanced training
5. **Feature Selection**: Correlation analysis and importance-based selection

## 🎓 Academic Contributions & Learning Outcomes

### **Technical Skills Demonstrated**
- **Machine Learning**: Implementation of 4 different ML algorithms with proper evaluation
- **Full-Stack Development**: React.js frontend with Flask backend integration
- **Database Management**: MongoDB Atlas cloud database implementation
- **API Design**: RESTful API development with proper authentication
- **Data Science**: Complete ML pipeline from data preprocessing to model deployment
- **Software Engineering**: Modular code architecture and version control

### **Research Contributions**
- **Comparative Analysis**: Systematic evaluation of multiple ML algorithms for diabetes prediction
- **Healthcare Informatics**: Practical application of ML in medical screening
- **Performance Optimization**: Model tuning and feature engineering for improved accuracy
- **User Experience**: Intuitive interface design for healthcare applications

### **Future Enhancements**
- **Deep Learning Integration**: CNN models for advanced pattern recognition
- **Real-time Monitoring**: Continuous health data integration
- **Mobile Application**: Cross-platform mobile app development
- **Clinical Validation**: Collaboration with medical professionals for validation

## 📝 Project Documentation

### **Code Quality & Standards**
- **TypeScript**: Type-safe frontend development
- **Python Best Practices**: PEP 8 compliance and documentation
- **API Documentation**: Comprehensive endpoint documentation
- **Error Handling**: Robust error handling and logging
- **Testing**: Unit tests for critical components

### **Deployment & Scalability**
- **Cloud Integration**: MongoDB Atlas for scalable data storage
- **Production Ready**: Environment-based configuration
- **Performance Optimization**: Efficient model loading and caching
- **Security**: JWT authentication and input validation

## 👨‍💻 Student Information

**Haris Behzad**  
**Final Year Project - Computer Science**  
**Academic Year: 2024**

**Contact Information:**
- GitHub: [@Harisbzd](https://github.com/Harisbzd)
- Project Repository: [https://github.com/Harisbzd/FinalYearProject](https://github.com/Harisbzd/FinalYearProject)
- Email: haris.behzad@example.com

## 🙏 Acknowledgments

**Academic Support:**
- University faculty for guidance and supervision
- BRFSS dataset providers for comprehensive health data
- Open source communities (Scikit-learn, XGBoost, React, Flask)

**Technical Resources:**
- Machine learning documentation and tutorials
- Web development best practices and frameworks
- Healthcare informatics research papers

## ⚠️ Important Disclaimers

**Academic Purpose**: This project is developed for educational and research purposes as part of a Final Year Project.

**Medical Disclaimer**: The predictions generated by this system should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

**Data Privacy**: User data is stored securely in MongoDB Atlas with proper authentication and access controls.

---

**Project Status**: ✅ Complete and Ready for Evaluation  
**Last Updated**: December 2024  
**Version**: 1.0.0