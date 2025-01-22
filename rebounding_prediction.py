import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create directory structure for results"""
    base_dir = 'basketball_analysis'
    categories = ['Playmaking', 'Shooter', 'Rebounding', 'Defense']  
    model_types = ['rf', 'hist_gb', 'xgb']
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each category and model type
    for category in categories:
        category_dir = os.path.join(base_dir, category.lower())
        os.makedirs(category_dir, exist_ok=True)
        
        # Create model-specific subdirectories
        for model_type in model_types:
            model_dir = os.path.join(category_dir, model_type)
            # Create subdirectories for different types of output
            for subdir in ['plots', 'models', 'results']:
                os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)
    
    return base_dir

category_features = {  

    'Playmaking': [
        'AST', 'AST_per_40', 'AST%',
        'TOV', 'TOV%', 'USG%',
        'OWS', 'WS', 'WS/40'
    ],  



    'Shooter': ['FGA', 'FG%', '3PA', '3P%','FTA', 'FT%',
        'TS%', 'eFG%', 'OWS'
    ],

        'Rebounding': [
        'TRB', 'TRB_per_40', 'TRB%', 'Minutes_Per_Game',
        'DWS', 'WS/40'
    ],
    

    

    
    'Defense': [
        'STL', 'STL_per_40', 'STL%',
        'BLK', 'BLK_per_40', 'BLK%',
        'DWS', 'WS', 'WS/40'
    ]
}

def engineer_features(df):
    df = df.copy()
    
    # Shooter features
    df['Three_Point_Rate'] = df['3PA'] / (df['FGA'] + 1e-6)
    df['Shot_Selection'] = df['eFG%'] / (df['USG%'] + 1e-6)
    df['Pure_Shooting'] = (df['3P%'] + df['FT%']) / 2
    
    # Rebounder features
    df['Rebound_per_Minute'] = df['TRB'] / (df['Minutes_Per_Game'] + 1e-6)
    df['Rebound_Efficiency'] = df['TRB%'] * df['Minutes_Per_Game'] / 40
    df['Impact_Rebounding'] = df['TRB_per_40'] * df['WS/40']
    
    # Playmaker features
    df['AST_to_TOV'] = df['AST'] / (df['TOV'] + 1e-6)
    df['Playmaking_Impact'] = df['AST%'] / (df['TOV%'] + 1e-6)
    df['Ball_Control'] = (df['AST_per_40'] * df['WS/40']) / (df['TOV'] + 1e-6)
    
    # Defender features
    df['Stocks'] = df['STL'] + df['BLK']
    df['Stocks_per_40'] = df['STL_per_40'] + df['BLK_per_40']
    df['Defense_Impact'] = (df['STL%'] + df['BLK%']) * df['DWS']
    
    return df

# Add engineered features to categories
category_features['Shooter'].extend(['Three_Point_Rate', 'Shot_Selection', 'Pure_Shooting'])
category_features['Rebounding'].extend(['Rebound_per_Minute', 'Rebound_Efficiency', 'Impact_Rebounding'])
category_features['Playmaking'].extend(['AST_to_TOV', 'Playmaking_Impact', 'Ball_Control'])
category_features['Defense'].extend(['Stocks', 'Stocks_per_40', 'Defense_Impact'])
def add_advanced_features(df):
    df = df.copy()
    if 'Minutes_Per_Game' in df.columns:
        minutes_factor = 40 / df['Minutes_Per_Game'].replace(0, 40)
        for stat in ['BLK', 'STL', 'TRB', 'AST']:
            if stat in df.columns:
                df[f'{stat}_per_40'] = df[stat] * minutes_factor
    return df

def generate_valid_weights(n_features, sorted_correlations, n_steps=6):
    """Generate weight combinations prioritizing features with higher correlations"""
    weights_list = []
    step = 1.0 / (n_steps - 1)
    
    # Create mapping from correlation-sorted order to original feature order
    feature_order = [feat for feat, _ in sorted_correlations]
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_order)}
    
    def generate_combinations(current_weights, feature_idx, remaining_sum):
        if feature_idx == n_features - 1:
            current_weights[feature_idx] = remaining_sum
            # Reorder weights according to correlation strength
            ordered_weights = [0] * n_features
            for feat, weight in zip(feature_order, current_weights):
                ordered_weights[feature_to_idx[feat]] = weight
            weights_list.append(ordered_weights)
            return
            
        # For highly correlated features, try higher weights first
        if feature_idx < n_features // 3:  # Top third of features
            weights = [i * step for i in range(n_steps-1, -1, -1)]  # Higher to lower
        elif feature_idx < 2 * n_features // 3:  # Middle third
            weights = [i * step for i in range(n_steps)]  # Lower to higher
        else:  # Bottom third
            weights = [i * step for i in range(n_steps)]  # Lower to higher
            
        for weight in weights:
            if weight <= remaining_sum:
                current_weights[feature_idx] = weight
                generate_combinations(current_weights, 
                                   feature_idx + 1, 
                                   remaining_sum - weight)
    
    # Initialize with equal weights
    weights_list.append([1.0/n_features] * n_features)
    
    # Generate combinations
    current_weights = [0] * n_features
    generate_combinations(current_weights, 0, 1.0)
    
    # Filter valid combinations
    weights_list = [weights for weights in weights_list 
                   if abs(sum(weights) - 1.0) < 1e-10]
    
    return weights_list

def print_detailed_metrics(y_true, y_pred, model_name):
    """Print detailed classification metrics for each class"""
    print(f"\nDetailed Metrics for {model_name}")
    print("-" * 50)
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%'])
    print(report)
    
    # Calculate additional metrics per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    print("\nPer-Class Statistics:")
    print("-" * 50)
    for i, class_name in enumerate(['Bottom 25%', 'Middle 50%', 'Top 25%']):
        print(f"\n{class_name}:")
        print(f"Precision: {precision[i]:.3f}")
        print(f"Recall: {recall[i]:.3f}")
        print(f"F1-Score: {f1[i]:.3f}")
        print(f"Support: {support[i]}")

def optimize_category_weights(df, category_features, target_column, n_steps=6):
    features = category_features
    
    # Calculate correlations
    correlations = [(feat, abs(df[feat].corr(df[target_column]))) 
                   for feat in features]
    sorted_correlations = sorted(correlations, key=lambda x: x[1], reverse=True)
    
    results = {
        'hist_gb': {'best_accuracy': 0, 'best_weights': None, 'best_model': None, 'best_predictions': None},
        'xgb': {'best_accuracy': 0, 'best_weights': None, 'best_model': None, 'best_predictions': None},
        'rf': {'best_accuracy': 0, 'best_weights': None, 'best_model': None, 'best_predictions': None}
    }
    
    weight_combinations = generate_valid_weights(len(features), sorted_correlations, n_steps)
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    hist_gb_model = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.1,
        max_depth=12,
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=12,
        subsample=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=12,
        random_state=42
    )
    
    models = {
        'hist_gb': hist_gb_model,
        'xgb': xgb_model,
        'rf': rf_model
    }
    
    X = df[features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    iteration = 0
    
    try:
        for i, weights in enumerate(weight_combinations, 1):
            if i % 5 == 0:
                print(f"Progress: {i}/{len(weight_combinations)} combinations tested")
                
            X_train_weighted = X_train.copy()
            X_test_weighted = X_test.copy()
            for feat, weight in zip(features, weights):
                X_train_weighted[feat] *= weight
                X_test_weighted[feat] *= weight
            
            for model_name, model in models.items():
                model.fit(X_train_weighted, y_train)
                y_pred = model.predict(X_test_weighted)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > results[model_name]['best_accuracy']:
                    iteration += 1
                    results[model_name]['best_accuracy'] = accuracy
                    results[model_name]['best_weights'] = weights
                    results[model_name]['best_model'] = model
                    results[model_name]['best_predictions'] = y_pred
                    
                    print(f"\nNew best {model_name.upper()} accuracy: {accuracy:.3f}")
                    print_detailed_metrics(y_test, y_pred, model_name)
                    
                    # Save weight combinations
                    save_dir = f'basketball_analysis/{target_column.split("_")[0].lower()}/{model_name}'
                    
                    # Save weights to text file
                    with open(f'{save_dir}/results/weights_iter_{iteration}.txt', 'w') as f:
                        f.write(f"Weight Combination - Iteration {iteration}\n")
                        f.write("-" * 50 + "\n")
                        for feat, weight in zip(features, weights):
                            f.write(f"{feat}: {weight:.3f} ({weight*100:.1f}%)\n")
                        f.write(f"\nAccuracy: {accuracy:.3f}\n")
                        
                    # Save model
                    model_data = {
                        'model': model,
                        'weights': dict(zip(features, weights)),
                        'accuracy': accuracy,
                        'features': features
                    }
                    joblib.dump(model_data, f'{save_dir}/models/model_iter_{iteration}.joblib')
                    
                    # Create and save plots
                    # 1. Confusion Matrix
                    plt.figure(figsize=(8, 6))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.3f}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.savefig(f'{save_dir}/plots/confusion_matrix_iter_{iteration}.png')
                    plt.close()
                    
                    # 2. Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        plt.figure(figsize=(10, 6))
                        importance = model.feature_importances_
                        features_imp = pd.Series(importance, index=features).sort_values(ascending=True)
                        features_imp.plot(kind='barh')
                        plt.title(f'Feature Importance - {model_name}\nAccuracy: {accuracy:.3f}')
                        plt.tight_layout()
                        plt.savefig(f'{save_dir}/plots/feature_importance_iter_{iteration}.png')
                        plt.close()
                    
                    # 3. Weight Distribution
                    plt.figure(figsize=(10, 6))
                    weight_series = pd.Series(weights, index=features).sort_values(ascending=True)
                    weight_series.plot(kind='barh')
                    plt.title(f'Weight Distribution - {model_name}\nAccuracy: {accuracy:.3f}')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/plots/weight_distribution_iter_{iteration}.png')
                    plt.close()
                    
                    # Save classification report
                    with open(f'{save_dir}/results/classification_report_iter_{iteration}.txt', 'w') as f:
                        f.write(classification_report(y_test, y_pred, 
                                                   target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
    
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        
    return results, y_test

def train_category_model(category_name, features, data):
    print(f"\nTraining models for {category_name}...")
    print("=" * 80)
    
    data = data.copy()
    target_col = f'{category_name}_Category'
    
    # Calculate correlations
    correlations = {feature: abs(data[feature].corr(data[target_col])) 
                   for feature in features}
    
    # Print feature correlations sorted by strength
    print("\nFeature correlations with target (sorted):")
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, corr in sorted_correlations:
        print(f"{feature}: {corr:.3f}")
    
    # Get the highest correlated feature and create binary features
    top_feature = sorted_correlations[0][0]
    print(f"\nCreating binary features for top correlated feature: {top_feature}")
    
    # Create binary features for the highest correlated feature
    data[f'{top_feature}_bottom_25'] = (
        data[top_feature] <= data[top_feature].quantile(0.25)
    ).astype(int)
    data[f'{top_feature}_top_25'] = (
        data[top_feature] >= data[top_feature].quantile(0.75)
    ).astype(int)
    
    # Add these new binary features to the feature list
    features = features + [f'{top_feature}_bottom_25', f'{top_feature}_top_25']
    print(f"Added binary features: {top_feature}_bottom_25, {top_feature}_top_25")
    
    # Train models and get results
    results, y_test = optimize_category_weights(data, features, target_col)
    
    # Find best model and print final detailed results
    best_model_type = max(results.items(), key=lambda x: x[1]['best_accuracy'])[0]
    best_result = results[best_model_type]
    
    print(f"\nFinal Best Model for {category_name} ({best_model_type}):")
    print("=" * 80)
    print(f"Overall Accuracy: {best_result['best_accuracy']:.3f}")
    print("\nFeature Importance Weights:")
    for feat, weight in zip(features, best_result['best_weights']):
        print(f"{feat}: {weight:.3f}")
    
    print("\nFinal Classification Report:")
    print_detailed_metrics(y_test, best_result['best_predictions'], best_model_type)
    
    return results

def prepare_data():
    print("Loading and preparing data...")
    nba_data = pd.read_csv('nba_player_ratings_25_99v2.csv')
    college_stats = pd.read_csv('college_stats_final.csv')
    
    # Add per-40 stats
    college_stats = add_advanced_features(college_stats)
    
    # Add engineered features
    college_stats = engineer_features(college_stats)
    
    # Merge datasets
    merged_data = pd.merge(college_stats, nba_data, on='Player', how='inner')
    
    # Create categories for all aspects
    aspects = ['Playmaking', 'Shooter',  'Rebounding', 'Defense']
    for aspect in aspects:
        merged_data[f'{aspect}_Category'] = pd.qcut(
            merged_data[f'{aspect}_Score'], 
            q=[0, 0.25, 0.75, 1], 
            labels=['Bottom 25%', 'Middle 50%', 'Top 25%']
        ).cat.codes
    
    return merged_data

def process_all_categories(data):
    scaler = StandardScaler()
    results = {}
    
    for category, features in category_features.items():
        print(f"\nProcessing {category}...")
        print("=" * 80)
        
        # Standardize features
        data_scaled = data.copy()
        data_scaled[features] = scaler.fit_transform(data[features])
        
        # Train models
        category_results = train_category_model(category, features, data_scaled)
        results[category] = category_results
        
        # Save best model and metadata
        best_model_type = max(category_results.items(), 
                            key=lambda x: x[1]['best_accuracy'])[0]
        best_result = category_results[best_model_type]
        
        # Save model and metadata
        model_data = {
            'model': best_result['best_model'],
            'weights': dict(zip(features, best_result['best_weights'])),
            'accuracy': best_result['best_accuracy'],
            'scaler': scaler,
            'features': features
        }
        
        model_filename = f'{category.lower()}_best_model.joblib'
        joblib.dump(model_data, model_filename)
        print(f"\nModel saved as {model_filename}")
    
    return results
def print_final_summary(all_results):
    print("\nFinal Results Summary")
    print("=" * 80)
    
    for category, results in all_results.items():
        print(f"\n{category} Results:")
        print("-" * 40)
        
        for model_type, result in results.items():
            print(f"{model_type}:")
            print(f"Accuracy: {result['best_accuracy']:.3f}")
            
        best_model = max(results.items(), key=lambda x: x[1]['best_accuracy'])
        print(f"\nBest Model: {best_model[0]} ({best_model[1]['best_accuracy']:.3f})")

if __name__ == "__main__":
    print("Starting basketball player analysis...")
    print("=" * 80)
    
    try:
        # Create directories
        base_dir = create_directories()
        print("Created directory structure in 'basketball_analysis'")
        
        # Prepare data
        merged_data = prepare_data()
        
        # Process all categories
        all_results = process_all_categories(merged_data)
        
        # Print final summary
        print_final_summary(all_results)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")