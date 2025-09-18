# Complete Setup Guide - Diabetes Prediction System

## Professor's Setup Instructions

**Student:** Haris Behzad  
**Project:** Final Year Project - Diabetes Prediction System  
**Total Size:** ~464MB (Optimized for submission)

---

## 🎯 Quick Start (5 Minutes Setup)

### **Prerequisites**
- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download here](https://python.org/)
- **Git** - [Download here](https://git-scm.com/)
- **Internet Connection** (for MongoDB Atlas cloud database)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/Harisbzd/FinalYearProject.git
cd FinalYearProject
```

### **Step 2: Backend Setup**
```bash
# Navigate to backend directory
cd flask-server

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### **Step 3: Frontend Setup**
```bash
# Navigate to frontend directory (from project root)
cd client

# Install Node.js dependencies
npm install --legacy-peer-deps
```

### **Step 4: Run the Application**

**Terminal 1 - Start Backend Server:**
```bash
cd flask-server
source venv/bin/activate  # Windows: venv\Scripts\activate
python server.py
```
*Server will start on: http://localhost:5001*

**Terminal 2 - Start Frontend:**
```bash
cd client
npm start
```
*Application will open at: http://localhost:3000*

---

## 🔧 Detailed Setup Instructions

### **Backend (Flask) Setup**

#### **1. Python Environment Setup**
```bash
cd flask-server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify Python version (should be 3.8+)
python --version
```

#### **2. Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

#### **3. Environment Configuration**
The `.env` file is already configured with MongoDB Atlas connection:
```bash
# Check if .env file exists
ls -la .env

# View connection string (optional)
cat .env
```

#### **4. Test Backend Connection**
```bash
# Test database connection
python -c "
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client.diabeties
print('✅ Database connection successful!')
print('📊 Collections:', db.list_collection_names())
"
```

#### **5. Start Backend Server**
```bash
python server.py
```

**Expected Output:**
```
* Running on http://127.0.0.1:5001
* Debug mode: on
```

---

### **Frontend (React) Setup**

#### **1. Node.js Environment Setup**
```bash
cd client

# Verify Node.js version (should be 16+)
node --version

# Verify npm version
npm --version
```

#### **2. Install Dependencies**
```bash
# Install all required packages
npm install

# This will create node_modules/ directory
# Install time: 2-3 minutes
```

#### **3. Verify Installation**
```bash
# Check if dependencies are installed
ls -la node_modules/

# Verify package.json
cat package.json
```

#### **4. Start Frontend Development Server**
```bash
npm start
```

**Expected Output:**
```
Compiled successfully!

You can now view client in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.xxx:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

---

## 🧪 Testing the System

### **1. User Registration**
1. Open http://localhost:3000
2. Click "Register" or "Sign Up"
3. Create a new account with:
   - Username: `test_user`
   - Email: `test@example.com`
   - Password: `password123`

### **2. Make a Prediction**
1. Login with your credentials
2. Fill out the health form with sample data:
   - High Blood Pressure: Yes
   - High Cholesterol: No
   - BMI: 25.5
   - Age: 35
   - (Fill other fields as needed)
3. Click "Predict Diabetes"
4. View results from all 4 ML models

### **3. Test All Models**
- **XGBoost**: Best performing model (75.3% accuracy)
- **Random Forest**: Ensemble method (74.8% accuracy)
- **KNN**: Instance-based learning (73.2% accuracy)
- **Logistic Regression**: Interpretable model (74.6% accuracy)

---

## 📊 System Architecture Verification

### **Backend API Endpoints**
Test these endpoints to verify functionality:

```bash
# Test main prediction endpoint
curl -X POST http://localhost:5001/api/predict-diabetes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"HighBP":1,"HighChol":0,"BMI":25.5,"Age":35,...}'

# Test authentication
curl -X POST http://localhost:5001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"password123"}'
```

### **Database Connection**
- **MongoDB Atlas**: Cloud-hosted database
- **Collections**: users, predictions, sessions
- **Connection**: Automatic via .env configuration

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### **1. Python Virtual Environment Issues**
```bash
# If venv activation fails:
python -m venv venv --clear
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### **2. Node.js Dependencies Issues**
```bash
# Clear npm cache and reinstall:
cd client
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### **3. Port Already in Use**
```bash
# If port 3000 is busy:
npm start -- --port 3001

# If port 5001 is busy:
# Edit flask-server/server.py and change port number
```

#### **4. Database Connection Issues**
```bash
# Test internet connection:
ping mongodb.net

# Verify .env file:
cat flask-server/.env
```

#### **5. Module Import Errors**
```bash
# Reinstall Python packages:
cd flask-server
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

---

## 📁 Project Structure After Setup

```
FinalYearProject/
├── client/                          # React Frontend
│   ├── node_modules/                # ✅ Created after npm install
│   ├── src/                         # Source code
│   ├── public/                      # Static assets
│   ├── package.json                 # Dependencies
│   └── package-lock.json            # Lock file
├── flask-server/                    # Flask Backend
│   ├── venv/                        # ✅ Created after python -m venv
│   ├── ml_models/                   # ML model files
│   ├── auth/                        # Authentication
│   ├── *.joblib                     # Trained models
│   ├── *.csv                        # Datasets
│   ├── server.py                    # Main application
│   ├── requirements.txt             # Python dependencies
│   └── .env                         # Environment config
├── README.md                        # Main documentation
├── MONGODB_SETUP.md                 # Database documentation
└── SETUP_GUIDE.md                   # This file
```

---

## 🎓 Academic Evaluation Checklist

### **For Professor/Evaluator:**

#### **✅ System Functionality**
- [ ] User registration and login works
- [ ] All 4 ML models make predictions
- [ ] Results are displayed correctly
- [ ] Database stores user data and predictions
- [ ] Frontend and backend communicate properly

#### **✅ Technical Implementation**
- [ ] Python virtual environment setup
- [ ] Node.js dependencies installation
- [ ] MongoDB Atlas cloud database connection
- [ ] JWT authentication system
- [ ] RESTful API endpoints
- [ ] React frontend with TypeScript

#### **✅ Machine Learning Models**
- [ ] XGBoost model (best performing)
- [ ] Random Forest model
- [ ] K-Nearest Neighbors model
- [ ] Logistic Regression model
- [ ] Model performance metrics available
- [ ] Feature importance analysis

#### **✅ Code Quality**
- [ ] Clean, documented code
- [ ] Proper error handling
- [ ] Security best practices
- [ ] Modular architecture
- [ ] Professional documentation

---

## 📞 Support & Contact

**Student:** Haris Behzad  
**Email:** haris.behzad@example.com  
**GitHub:** [@Harisbzd](https://github.com/Harisbzd)  
**Repository:** [https://github.com/Harisbzd/FinalYearProject](https://github.com/Harisbzd/FinalYearProject)

---

## ⚠️ Important Notes

1. **Internet Required**: MongoDB Atlas cloud database requires internet connection
2. **No Local Database**: No need to install MongoDB locally
3. **All Models Included**: All 4 ML models are pre-trained and ready to use
4. **Production Ready**: System is configured for immediate use
5. **Academic Purpose**: This is a Final Year Project for educational evaluation

---

**Setup Time**: ~5 minutes  
**Total Project Size**: ~464MB  
**Status**: ✅ Ready for Evaluation  
**Last Updated**: December 2024
