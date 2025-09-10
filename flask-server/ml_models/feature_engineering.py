"""
Shared Feature Engineering Module for Diabetes Prediction Models

This module contains the common feature engineering functions used across
all machine learning models to ensure consistency and reduce code duplication.
"""

import pandas as pd
import numpy as np


def apply_feature_engineering(X):
    """
    Apply comprehensive feature engineering to the diabetes dataset.
    
    This function creates clinically relevant features including:
    - Interaction terms between important variables
    - Polynomial features for BMI
    - Categorical groupings
    - Risk factor combinations
    - Socioeconomic indicators
    
    Args:
        X (pd.DataFrame): Input features dataframe
        
    Returns:
        pd.DataFrame: Enhanced features dataframe with engineered features
    """
    # Create a copy to avoid modifying the original
    X_enhanced = X.copy()
    
    # Add feature interactions that are clinically relevant
    X_enhanced['BMI_Age_interaction'] = X_enhanced['BMI'] * X_enhanced['Age']
    X_enhanced['HighBP_HighChol_interaction'] = X_enhanced['HighBP'] * X_enhanced['HighChol']
    X_enhanced['GenHlth_BMI_interaction'] = X_enhanced['GenHlth'] * X_enhanced['BMI']
    X_enhanced['HeartDisease_BMI_interaction'] = X_enhanced['HeartDiseaseorAttack'] * X_enhanced['BMI']
    X_enhanced['Stroke_HeartDisease_interaction'] = X_enhanced['Stroke'] * X_enhanced['HeartDiseaseorAttack']
    
    # Add polynomial features for BMI (important for diabetes)
    X_enhanced['BMI_squared'] = X_enhanced['BMI'] ** 2
    X_enhanced['BMI_sqrt'] = np.sqrt(X_enhanced['BMI'])
    
    # Add age groups (clinically relevant)
    X_enhanced['Age_group'] = pd.cut(X_enhanced['Age'], bins=[0, 4, 8, 12, 15], labels=[0, 1, 2, 3]).astype(float)
    
    # Enhanced feature engineering for better accuracy
    # Risk factor combinations
    X_enhanced['Cardiovascular_Risk'] = X_enhanced['HighBP'] + X_enhanced['HighChol'] + X_enhanced['HeartDiseaseorAttack'] + X_enhanced['Stroke']
    X_enhanced['Lifestyle_Risk'] = X_enhanced['Smoker'] + X_enhanced['HvyAlcoholConsump'] + (1 - X_enhanced['PhysActivity'])
    X_enhanced['Health_Access_Risk'] = (1 - X_enhanced['AnyHealthcare']) + X_enhanced['NoDocbcCost']
    
    # BMI categories (clinically relevant)
    X_enhanced['BMI_category'] = pd.cut(X_enhanced['BMI'], bins=[0, 18.5, 25, 30, 50], labels=[0, 1, 2, 3]).astype(float)
    X_enhanced['BMI_obese'] = (X_enhanced['BMI'] >= 30).astype(float)
    X_enhanced['BMI_overweight'] = ((X_enhanced['BMI'] >= 25) & (X_enhanced['BMI'] < 30)).astype(float)
    
    # Age-BMI risk interaction
    X_enhanced['Age_BMI_Risk'] = X_enhanced['Age'] * X_enhanced['BMI'] / 10  # Normalized interaction
    
    # Health status combinations
    X_enhanced['Mental_Physical_Health'] = X_enhanced['MentHlth'] + X_enhanced['PhysHlth']
    X_enhanced['Health_Problems'] = X_enhanced['GenHlth'] + X_enhanced['DiffWalk']
    
    # Income-Education interaction (socioeconomic factors)
    X_enhanced['Socioeconomic_Status'] = X_enhanced['Income'] * X_enhanced['Education']
    
    # Fruit and vegetable consumption patterns
    X_enhanced['Healthy_Diet'] = X_enhanced['Fruits'] + X_enhanced['Veggies']
    X_enhanced['Diet_Quality'] = X_enhanced['Fruits'] * X_enhanced['Veggies']
    
    # Handle NaN values created by feature engineering
    X_enhanced = X_enhanced.fillna(0)
    
    return X_enhanced


def get_feature_engineering_info():
    """
    Get information about the feature engineering process.
    
    Returns:
        dict: Information about original and enhanced feature counts
    """
    return {
        'original_features': 21,
        'engineered_features': 20,
        'total_features': 41,
        'description': 'Enhanced with clinically relevant interactions, polynomial features, and risk combinations'
    }
