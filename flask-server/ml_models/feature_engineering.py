
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')


def winsorize_outliers(series, limits=(0.01, 0.01)):
    """Apply winsorizing to handle outliers."""
    return pd.Series(mstats.winsorize(series, limits=limits), index=series.index)


def apply_feature_engineering(X, handle_outliers=True, use_one_hot=True, feature_selection=None, target=None, remove_duplicates=True):
    X_enhanced = X.copy()
    
    # Convert string values to numeric for mathematical operations
    numeric_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                      'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    for col in numeric_columns:
        if col in X_enhanced.columns:
            X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce')
    
    # Remove duplicate rows
    if remove_duplicates:
        original_count = len(X_enhanced)
        X_enhanced = X_enhanced.drop_duplicates()
        duplicates_removed = original_count - len(X_enhanced)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        
        # If target is provided, also remove corresponding target values
        if target is not None:
            target = target.iloc[X_enhanced.index]
    
    # Handle outliers
    if handle_outliers:
        continuous_features = ['BMI', 'Age', 'MentHlth', 'PhysHlth']
        for feature in continuous_features:
            if feature in X_enhanced.columns:
                X_enhanced[feature] = winsorize_outliers(X_enhanced[feature])
    
    # Create interaction terms
    X_enhanced['BMI_Age_interaction'] = X_enhanced['BMI'] * X_enhanced['Age']
    X_enhanced['HighBP_HighChol_interaction'] = X_enhanced['HighBP'] * X_enhanced['HighChol']
    X_enhanced['GenHlth_BMI_interaction'] = X_enhanced['GenHlth'] * X_enhanced['BMI']
    
    # Add polynomial features
    X_enhanced['BMI_squared'] = X_enhanced['BMI'] ** 2
    X_enhanced['Age_squared'] = X_enhanced['Age'] ** 2
    
    # Create categorical features
    if use_one_hot:
        # Age groups
        X_enhanced['Age_group'] = pd.cut(X_enhanced['Age'], bins=[0, 4, 8, 12, 15], 
                                        labels=['Young', 'Middle', 'Senior', 'Elderly'])
        age_dummies = pd.get_dummies(X_enhanced['Age_group'], prefix='Age_group')
        X_enhanced = pd.concat([X_enhanced, age_dummies], axis=1)
        X_enhanced.drop('Age_group', axis=1, inplace=True)
        
        # BMI categories
        X_enhanced['BMI_category'] = pd.cut(X_enhanced['BMI'], bins=[0, 18.5, 25, 30, 50], 
                                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Merge rare categories
        bmi_counts = X_enhanced['BMI_category'].value_counts()
        min_samples = max(50, len(X_enhanced) * 0.01)
        
        if bmi_counts.get('Underweight', 0) < min_samples:
            X_enhanced['BMI_category'] = X_enhanced['BMI_category'].replace('Underweight', 'Normal')
        
        bmi_dummies = pd.get_dummies(X_enhanced['BMI_category'], prefix='BMI_category')
        X_enhanced = pd.concat([X_enhanced, bmi_dummies], axis=1)
        X_enhanced.drop('BMI_category', axis=1, inplace=True)
    else:
        # Numeric encoding
        X_enhanced['Age_group'] = pd.cut(X_enhanced['Age'], bins=[0, 4, 8, 12, 15], 
                                        labels=[0, 1, 2, 3]).astype(float)
        X_enhanced['BMI_category'] = pd.cut(X_enhanced['BMI'], bins=[0, 18.5, 25, 30, 50], 
                                           labels=[0, 1, 2, 3]).astype(float)
    
    # Risk factor combinations
    X_enhanced['Cardiovascular_Risk'] = (X_enhanced['HighBP'] + X_enhanced['HighChol'] + 
                                        X_enhanced['HeartDiseaseorAttack'] + X_enhanced['Stroke']) / 4
    X_enhanced['Lifestyle_Risk'] = (X_enhanced['Smoker'] + X_enhanced['HvyAlcoholConsump'] + 
                                   (1 - X_enhanced['PhysActivity'])) / 3
    X_enhanced['Health_Access_Risk'] = ((1 - X_enhanced['AnyHealthcare']) + 
                                       X_enhanced['NoDocbcCost']) / 2
    
    # BMI indicators
    X_enhanced['BMI_obese'] = (X_enhanced['BMI'] >= 30).astype(float)
    X_enhanced['BMI_overweight'] = ((X_enhanced['BMI'] >= 25) & (X_enhanced['BMI'] < 30)).astype(float)
    
    # Health status combinations
    X_enhanced['Health_Status_Score'] = (X_enhanced['GenHlth'] + X_enhanced['MentHlth'] + 
                                        X_enhanced['PhysHlth']) / 3
    X_enhanced['Health_Problems'] = X_enhanced['GenHlth'] + X_enhanced['DiffWalk']
    
    # Socioeconomic factors
    X_enhanced['Socioeconomic_Status'] = (X_enhanced['Income'] * X_enhanced['Education']) / 8
    
    # Diet quality indicators
    X_enhanced['Diet_Score'] = (X_enhanced['Fruits'] + X_enhanced['Veggies']) / 2
    
    # Additional clinical features
    X_enhanced['Metabolic_Risk'] = (X_enhanced['HighBP'] + X_enhanced['HighChol'] + 
                                   X_enhanced['BMI_obese']) / 3
    X_enhanced['Activity_Level'] = X_enhanced['PhysActivity'] * (1 - X_enhanced['DiffWalk'])
    
    # Handle NaN values
    X_enhanced = X_enhanced.fillna(0)
    
    # Remove redundant features
    X_enhanced = detect_and_remove_redundant_features(X_enhanced, correlation_threshold=0.95, target=target)
    
    # Feature selection
    if feature_selection is not None and target is not None and feature_selection < len(X_enhanced.columns):
        selected_features, _ = select_features(X_enhanced, target, method='mutual_info', k=feature_selection)
        X_enhanced = X_enhanced[selected_features]
    
    # Return both X and target if target was provided
    if target is not None:
        return X_enhanced, target
    else:
        return X_enhanced


def select_features(X, y, method='mutual_info', k=30):
    """Select top k features using various feature selection methods."""
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_
        
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_
        
    else:
        # Placeholder for permutation importance
        selected_features = X.columns.tolist()[:k]
        feature_scores = None
    
    return selected_features, feature_scores


def apply_scaling(X, method='standard', fit_scaler=None):
    """Apply scaling to features with different methods."""
    if method == 'standard':
        scaler = StandardScaler() if fit_scaler is None else fit_scaler
    elif method == 'robust':
        scaler = RobustScaler() if fit_scaler is None else fit_scaler
    else:
        raise ValueError("Method must be 'standard' or 'robust'")
    
    if fit_scaler is None:
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler
    else:
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler


def get_feature_importance_analysis(X, y, model=None):
    """Analyze feature importance using multiple methods."""
    results = {}
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    results['mutual_info'] = dict(zip(X.columns, mi_scores))
    
    f_scores, _ = f_classif(X, y)
    results['f_statistic'] = dict(zip(X.columns, f_scores))
    
    if model is not None:
        perm_importance = permutation_importance(model, X, y, random_state=42, n_repeats=10)
        results['permutation'] = dict(zip(X.columns, perm_importance.importances_mean))
    
    return results


def detect_and_remove_redundant_features(X, correlation_threshold=0.95, target=None):
    """Detect and remove redundant features based on correlation analysis."""
    X_clean = X.copy()
    
    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    if target is not None and len(to_drop) > 0:
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_clean, target)
        feature_importance = dict(zip(X_clean.columns, rf.feature_importances_))
        
        final_to_drop = []
        for feature in to_drop:
            correlations = corr_matrix[feature].sort_values(ascending=False)
            most_correlated = correlations.index[1]
            
            if feature_importance.get(feature, 0) < feature_importance.get(most_correlated, 0):
                final_to_drop.append(feature)
            else:
                final_to_drop.append(most_correlated)
        
        to_drop = final_to_drop
    
    X_clean = X_clean.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} redundant features: {to_drop}")
    return X_clean



