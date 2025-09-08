# Diabetes Prediction System - Final Year Project

A comprehensive machine learning-based diabetes prediction system built with React frontend and Flask backend, featuring multiple ML models with advanced evaluation metrics.

## 🎯 Project Overview

This project implements a web-based diabetes prediction system that uses machine learning algorithms to predict diabetes risk based on health indicators. The system provides a user-friendly interface for both patients and healthcare professionals to assess diabetes risk.

## 🚀 Features

### Machine Learning Models
- **Logistic Regression**: Traditional statistical approach with excellent interpretability
- **XGBoost**: Advanced gradient boosting with high accuracy
- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
- **Random Forest**: Ensemble method for robust predictions

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Sensitivity**: True positive rate (diabetes detection rate)
- **Specificity**: True negative rate (non-diabetes identification rate)

### User Interface
- **Patient Dashboard**: Input health data and view predictions
- **Model Comparison**: Compare different ML models side-by-side
- **Visual Analytics**: Feature importance, confusion matrices, ROC curves
- **Authentication**: Secure user registration and login system

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

## 📊 Model Performance

### XGBoost (Best Performing)
- **Accuracy**: 75.3%
- **AUC Score**: 82.7%
- **Sensitivity**: 79.4%
- **Specificity**: 71.2%

### Logistic Regression
- **Accuracy**: 74.6%
- **AUC Score**: 82.3%
- **Sensitivity**: 76.8%
- **Specificity**: 72.5%

## 🏗️ Project Structure

```
FYP2-main/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── context/        # Authentication context
│   │   ├── routes/         # Routing configuration
│   │   └── data/          # Static data files
│   └── public/            # Static assets and ML visualizations
├── flask-server/          # Flask backend
│   ├── ml_models/         # Machine learning models
│   ├── auth/             # Authentication modules
│   └── server.py         # Main Flask application
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- MongoDB

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harisbzd/FinalYearProject.git
   cd FinalYearProject
   ```

2. **Backend Setup**
   ```bash
   cd flask-server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd client
   npm install
   ```

4. **Database Setup**
   - Install and start MongoDB
   - Update database connection in `flask-server/server.py`

### Running the Application

1. **Start the Flask server**
   ```bash
   cd flask-server
   source venv/bin/activate
   python server.py
   ```

2. **Start the React development server**
   ```bash
   cd client
   npm start
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## 📈 Model Training

To retrain the models with updated data:

```bash
cd flask-server
source venv/bin/activate

# Train Logistic Regression
python ml_models/logistic_regression.py

# Train XGBoost
python ml_models/xgboost_model.py

# Train other models
python ml_models/knn_model.py
python ml_models/randomforest_model.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the flask-server directory:
```
MONGODB_URI=mongodb://localhost:27017/diabetes_prediction
JWT_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

### Model Configuration
- Models are automatically saved after training
- Configuration files are in `flask-server/ml_models/`
- Visualization outputs are saved to `client/public/`

## 📊 Dataset

The system uses the BRFSS (Behavioral Risk Factor Surveillance System) 2015 dataset with the following features:
- High Blood Pressure
- High Cholesterol
- BMI
- Smoking Status
- Physical Activity
- Mental Health
- Physical Health
- Age, Sex, Education, Income
- And more health indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Haris Behzad**
- GitHub: [@Harisbzd](https://github.com/Harisbzd)
- Project Link: [https://github.com/Harisbzd/FinalYearProject](https://github.com/Harisbzd/FinalYearProject)

## 🙏 Acknowledgments

- BRFSS dataset providers
- Scikit-learn and XGBoost communities
- React and Flask documentation teams
- Open source contributors

## 📞 Support

For support, email haris.behzad@example.com or create an issue in the repository.

---

**Note**: This is a Final Year Project for academic purposes. The predictions should not be used as a substitute for professional medical advice.