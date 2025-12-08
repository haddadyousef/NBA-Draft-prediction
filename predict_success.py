import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import APPNP
import umap
from skimage.draw import disk
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras import layers, models, utils
import warnings
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from tensorflow.keras.callbacks import EarlyStopping
import os

import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        return tf.reduce_mean(alpha * (1-pt)**gamma * ce)
    return loss

# Define a global output folder
OUTPUT_FOLDER = 'final_run_results_mid_age_smote'
STAT_TEST_FOLDER = os.path.join(OUTPUT_FOLDER, 'statistical_tests')
os.makedirs(STAT_TEST_FOLDER, exist_ok=True)


category_features = { 
    'Rebounding': [
        'TRB', 'TRB_per_40', 'TRB%',
        'Games', 'Minutes_Per_Game',
        'DWS', 'WS', 'WS/40'
    ],

    'Playmaking': [
        'AST', 'AST_per_40', 'AST%',
        'TOV', 'TOV%', 'USG%',
        'OWS', 'WS', 'WS/40'
    ],



    'Defense': [
        'STL', 'STL_per_40', 'STL%',
        'BLK', 'BLK_per_40', 'BLK%',
        'DWS', 'WS', 'WS/40'
    ], 

    'Shooter': [
        'FG', 'FGA', 'FG%', 
        '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%',
        'TS%', 'eFG%', 'OWS'
    ]





}

def create_binary_top_bottom_features(df, feature, feature_name):
    """Create Top 25 and Bottom 25 binary features for a specified feature."""
    top_25 = df[feature] >= df[feature].quantile(0.75)
    bot_25 = df[feature] <= df[feature].quantile(0.25)
    df[f"{feature_name}_Top_25"] = top_25.astype(int)
    df[f"{feature_name}_Bottom_25"] = bot_25.astype(int)
    return df

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
    
    # Interaction terms for defensive features
    df['STL_BLK_Interaction'] = df['STL_per_40'] * df['BLK_per_40']
    df['STL_BLK_Impact'] = df['STL%'] * df['BLK%']
    
    return df

# Add engineered features to categories
category_features['Shooter'].extend(['Three_Point_Rate', 'Shot_Selection', 'Pure_Shooting'])
category_features['Rebounding'].extend(['Rebound_per_Minute', 'Rebound_Efficiency', 'Impact_Rebounding'])
category_features['Playmaking'].extend(['AST_to_TOV', 'Playmaking_Impact', 'Ball_Control'])
category_features['Defense'].extend(['Stocks', 'Stocks_per_40', 'Defense_Impact', 'STL_BLK_Interaction', 'STL_BLK_Impact'])

def add_advanced_features(df):
    df = df.copy()
    if 'Minutes_Per_Game' in df.columns:
        minutes_factor = 40 / df['Minutes_Per_Game'].replace(0, 40)
        for stat in ['BLK', 'STL', 'TRB', 'AST']:
            if stat in df.columns:
                df[f'{stat}_per_40'] = df[stat] * minutes_factor
    return df

def save_skill_statistical_tables(
    feat_results, 
    category_features,
    output_folder='output_files_binary/statistical_tests/'
):
    os.makedirs(output_folder, exist_ok=True)
    
    skills = ["Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Scorer": "Scorer Statistical Analysis",
        "Rebounding": "Rebounding Statistical Analysis",
        "Shooter": "Shooting Statistical Analysis",
        "Playmaking": "Playmaking Statistical Analysis"
    }

    for skill in skills:
        if skill not in feat_results:
            continue
        corr = feat_results[skill]['corr']
        info = feat_results[skill]['info_gain']

        # Collect features: category, binary, cluster
        skill_feats = set(category_features[skill])
        corr_feats = set(corr['feature'])
        info_feats = set(info['feature'])
        all_feats = corr_feats | info_feats | skill_feats
        # Add cluster features
        cluster_feats = {f for f in all_feats if f.startswith('cluster_') or f.startswith('dist_to_cluster_')}
        # Add binary features present
        binary_feats = {f for f in all_feats if '_Top_25' in f or '_Bottom_25' in f}
        feats = list(skill_feats | cluster_feats | binary_feats | corr_feats | info_feats)

        # Build table (sorted by information gain, descending)
        info_dict = info.set_index('feature')['info_gain'].to_dict()
        corr_dict = corr.set_index('feature')['abs_corr'].to_dict()
        feats_sorted = sorted(feats, key=lambda f: info_dict.get(f, 0), reverse=True)

        # Identify top correlation feature and its binaries
        if not corr.empty:
            top_corr_feat = corr.sort_values('abs_corr', ascending=False).iloc[0]['feature']
            highlight_set = {top_corr_feat, f"{top_corr_feat}_Top_25", f"{top_corr_feat}_Bottom_25"}
        else:
            highlight_set = set()

        table_rows = []
        for feat in feats_sorted:
            corr_val = corr_dict.get(feat, np.nan)
            info_val = info_dict.get(feat, np.nan)
            table_rows.append([feat, corr_val, info_val])

        table_df = pd.DataFrame(table_rows, columns=['Feature', 'Correlation', 'Information Gain'])

        # Plot as table
        fig, ax = plt.subplots(figsize=(7, 0.4*len(table_df)+2))
        ax.axis('off')
        mpl_table = ax.table(
            cellText=[
                [
                    row[0], 
                    f"{row[1]:.3f}" if not pd.isnull(row[1]) else "", 
                    f"{row[2]:.3f}" if not pd.isnull(row[2]) else ""
                ] 
                for row in table_rows
            ],
            colLabels=table_df.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.scale(1.1, 1.1)

        # Style: highlight the top corr feat and its binaries
        for key, cell in mpl_table.get_celld().items():
            if key[0] == 0:
                cell.set_fontsize(12)
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            elif key[0] > 0:
                feat_name = table_rows[key[0]-1][0]
                if feat_name in highlight_set:
                    cell.set_facecolor('#ffeb99')  # highlight color
                elif key[1] == 0:  # Feature name column
                    cell.set_text_props(weight='bold')
        
        plt.title(pretty_names[skill], fontsize=14, pad=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{skill.lower()}_statistical_table.png"), bbox_inches='tight', dpi=220)
        plt.close()
        print(f"Saved {skill} summary to {os.path.join(output_folder, f'{skill.lower()}_statistical_table.png')}")

def load_or_train_trees(data, cluster_descriptions, binary_features_by_skill, force_retrain=False):
    """
    Load saved decision tree results if available, otherwise train and save them.
    """
    cache_file = os.path.join(OUTPUT_FOLDER, "all_results.joblib")

    if os.path.exists(cache_file) and not force_retrain:
        print(f"\nLoading cached tree results from {cache_file}")
        cached = joblib.load(cache_file)
        return cached["all_results"], cached["all_test_results"]

    # Otherwise, train from scratch
    print("\nNo cache found (or retraining forced). Training trees...")
    all_results, all_test_results = process_all_categories(data, cluster_descriptions, binary_features_by_skill)

    # Save to disk
    joblib.dump({"all_results": all_results, "all_test_results": all_test_results}, cache_file)
    print(f"\nSaved tree results cache at {cache_file}")

    return all_results, all_test_results        

def save_overall_statistical_summary_table(
        feat_results, 
        output_path='output_files_binary/statistical_tests/overall_statistical_summary_table.png'
    ):
    # Skills and labels
    skills = ["Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Rebounding": "Rebounding",
        "Shooter": "Shooting",
        "Playmaking": "Playmaking"
    }

    # Build a DataFrame for each skill, then align them by all features
    all_features = set()
    skill_tables = {}
    for skill in skills:
        if skill not in feat_results:
            continue
        corr = feat_results[skill]['corr'][['feature', 'abs_corr']].set_index('feature')
        info = feat_results[skill]['info_gain'][['feature', 'info_gain']].set_index('feature')
        df = corr.join(info, how='outer')
        skill_tables[skill] = df
        all_features.update(df.index.tolist())
    
    all_features = sorted(list(all_features))
    
    # Build a multi-index DataFrame: rows=features, columns=(Skill, Metric)
    arrays = []
    for skill in skills:
        arrays += [(pretty_names[skill], "Correlation"), (pretty_names[skill], "Info Gain")]
    columns = pd.MultiIndex.from_tuples(arrays)
    data = []
    for feat in all_features:
        row = []
        for skill in skills:
            df = skill_tables.get(skill)
            if df is not None and feat in df.index:
                row.append(df.loc[feat, 'abs_corr'])
                row.append(df.loc[feat, 'info_gain'])
            else:
                row += [np.nan, np.nan]
        data.append(row)
    table = pd.DataFrame(data, columns=columns, index=all_features)
    
    # Identify binary features for highlighting (optional: bold them)
    highlight_features = []
    for skill in skills:
        # Find highest correlated feature
        corr = feat_results[skill]['corr']
        if not corr.empty:
            top_feat = corr.iloc[0]['feature']
            highlight_features += [f"{top_feat}_Top_25", f"{top_feat}_Bottom_25"]
    
    # Plot table as image (matplotlib)
    fig, ax = plt.subplots(figsize=(min(2+2*len(skills), 20), min(0.5*len(all_features)+1, 28)))
    ax.axis('off')
    
    # Optionally bold the binary features
    def _bold(val, feat):
        if feat in highlight_features and not pd.isnull(val):
            return f"\033[1m{val:.3f}\033[0m"
        elif not pd.isnull(val):
            return f"{val:.3f}"
        else:
            return ""
    
    # Prepare display table with bold formatting
    display_table = []
    for i, feat in enumerate(all_features):
        row = [feat]
        for j in range(len(skills)):
            for k in range(2):
                val = table.iloc[i, j*2 + k]
                row.append(_bold(val, feat))
        display_table.append(row)
    
    col_labels = ["Feature"]
    for skill in skills:
        col_labels += [f"{pretty_names[skill]} Corr.", f"{pretty_names[skill]} InfoGain"]
    
    # Draw the table
    mpl_table = ax.table(
        cellText=display_table,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.scale(1.2, 1.2)
    
    # Highlight header
    for key, cell in mpl_table.get_celld().items():
        if key[0] == 0:  # header
            cell.set_fontsize(11)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif key[1] == 0:  # feature column
            cell.set_fontsize(10)
            cell.set_text_props(weight='bold')
        # Optionally: highlight the binary features for each skill
        elif display_table[key[0]-1][0] in highlight_features:
            cell.set_facecolor('#ffe599')
    
    plt.title("Statistical Feature Summary (Correlation and Information Gain)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=220)
    plt.close()
    print(f"Overall statistical summary table saved as {output_path}")

def save_overall_statistical_summary(feat_results, output_path='output_files_binary/statistical_tests/overall_statistical_summary.png'):
    # Prepare skill order and pretty names
    skills = ["Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Rebounding": "Rebounding Statistical Analysis",
        "Shooter": "Shooter Statistical Analysis",
        "Playmaking": "Playmaking Statistical Analysis"
    }
    
    summaries = []
    for skill in skills:
        if skill not in feat_results:
            continue
        corr = feat_results[skill]['corr']
        info = feat_results[skill]['info_gain']
        chi = feat_results[skill]['chi']
        
        s = []
        s.append(f"{pretty_names[skill]}\n" + "-"*40)
        
        s.append("Feature correlations with target (sorted):")
        for i, row in corr.iterrows():
            s.append(f"{row['feature']}: {row['abs_corr']:.3f}")
        s.append("")
        
        s.append("Information Gain Results:")
        for i, row in info.iterrows():
            s.append(f"{row['feature']}: {row['info_gain']:.3f}")
        s.append("")
        
        s.append("Chi-square Test Results:")
        for i, row in chi.iterrows():
            s.append(f"{row['feature']}: chi2 = {row['chi2_stat']:.1f}, p-value = {row['p_value']:.2e}")
        summaries.append("\n".join(s))

    # Plot as columns
    fig, axs = plt.subplots(1, len(summaries), figsize=(7*len(summaries), 14))
    if len(summaries) == 1:
        axs = [axs]
    for i, summary in enumerate(summaries):
        axs[i].axis('off')
        axs[i].text(0, 1, summary, va='top', ha='left', fontsize=10, family='monospace')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Overall statistical summary saved as {output_path}")

def add_clustering_features(df, n_clusters=5, random_state=42):
    """
    Clusters players based only on selected college stats archetypes.
    """
    print("\nPerforming player clustering analysis (college stats only)...")
    result = df.copy()

    # Define clustering features strictly from college stats
    cluster_features = [
        'PPG',
        'DWS',
        'AST',   # Playmaking
        'BLK',   # Rim protection
        '3P%',   # Shooting
        'TRB',   # Rebounding
        'TOV',   # Ball security
        'FT%',   # Free throw shooting
        'USG%',  # Usage
        'STL',  
        'FTA', 
        'WS' # Steals (perimeter defense)
    ]

    # Use only features present in the dataframe
    valid_features = [f for f in cluster_features if f in result.columns]
    if len(valid_features) < 3:
        print("Not enough valid features for clustering. Using available features instead.")
        valid_features = [col for col in result.columns if col not in 
                         ['Player', 'Team', 'Conference', 'Season']][:10]

    print(f"Using {len(valid_features)} features for clustering: {', '.join(valid_features)}")

    clustering_data = result[valid_features].fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    n_components = min(5, len(valid_features))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_data = pca.fit_transform(scaled_data)

    # Elbow method plot
    distortions = []
    K = range(1, min(10, len(result)))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(pca_data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'cluster_elbow_method.png'))
    plt.close()
    print(f"Elbow method plot saved as 'cluster_elbow_method.png'")

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(pca_data)
    result['player_cluster'] = clusters

    # One-hot encode clusters
    for i in range(n_clusters):
        result[f'cluster_{i}'] = (result['player_cluster'] == i).astype(int)

    # Distance to centers
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        center_pca = center.reshape(1, -1)
        result[f'dist_to_cluster_{i}'] = np.linalg.norm(pca_data - center_pca, axis=1)

    # Cluster stats/descriptions
    cluster_stats = {}
    cluster_descriptions = {}
    for cluster in range(n_clusters):
        cluster_players = result[result['player_cluster'] == cluster]
        cluster_means = cluster_players[valid_features].mean()
        cluster_stats[cluster] = cluster_means
        distinctive_stats = []
        for feat in valid_features:
            overall_mean = result[feat].mean()
            cluster_mean = cluster_means[feat]
            std_dev = result[feat].std()
            if overall_mean != 0 and std_dev > 0:
                z_score = (cluster_mean - overall_mean) / std_dev
                distinctive_stats.append((feat, z_score))
        distinctive_stats.sort(key=lambda x: abs(x[1]), reverse=True)
        top_stats = distinctive_stats[:3]
        description = []
        for stat, z_score in top_stats:
            if z_score > 1:
                description.append(f"high {stat}")
            elif z_score < -1:
                description.append(f"low {stat}")
        if description:
            cluster_descriptions[cluster] = "Players with " + ", ".join(description)
        else:
            cluster_descriptions[cluster] = "Balanced players"

    # Print cluster info
    print("\nPlayer Archetypes (Clusters):")
    for cluster, description in cluster_descriptions.items():
        player_count = sum(result['player_cluster'] == cluster)
        print(f"Cluster {cluster} ({player_count} players): {description}")
        examples = result[result['player_cluster'] == cluster]['Player'].head(3).tolist()
        if examples:
            print(f"  Examples: {', '.join(examples)}")

    # Visualization
    try:
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(
            pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', s=40, alpha=0.6, label='Players'
        )
        plt.xlabel('Principal Component 1 (captures most variance)')
        plt.ylabel('Principal Component 2 (captures next most variance)')
        plt.title('Player Clusters (PCA-Reduced Space)')
        centers = kmeans.cluster_centers_
        for i, center in enumerate(centers):
            distances = np.linalg.norm(pca_data - center, axis=1)
            closest_idx = np.argmin(distances)
            player_name = result.iloc[closest_idx]['Player']
            plt.scatter(center[0], center[1], c='red', s=180, marker='X', edgecolors='black', linewidths=2, zorder=5)
            plt.annotate(
                f"Cluster {i} center\n({player_name})",
                (center[0], center[1]),
                textcoords="offset points",
                xytext=(0, -45),
                ha='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
            )
        for cluster in range(n_clusters):
            cluster_mask = result['player_cluster'] == cluster
            examples = result[cluster_mask].head(2)
            for idx, player in examples.iterrows():
                player_idx = result.index.get_loc(idx)
                x, y = pca_data[player_idx, 0], pca_data[player_idx, 1]
                plt.annotate(player['Player'], (x, y), fontsize=8, alpha=0.7,
                             xytext=(5, 5), textcoords='offset points', color='black')
        cbar = plt.colorbar(scatter, label='Cluster Index')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'player_clusters.png'), dpi=200)
        plt.close()
        print("Player cluster visualization saved as 'player_clusters.png'")
    except Exception as e:
        print(f"Couldn't create cluster visualization: {str(e)}")

    return result, cluster_descriptions

def run_and_save_stat_tests(df, cluster_descriptions, category_features, engineered_feats):
    categories = ['Rebounding', 'Shooter', 'Playmaking']
    n_clusters = len(cluster_descriptions)
    feature_results = {}
    binary_features_by_skill = {}

    for skill in categories:
        print(f"\nRunning statistical tests for {skill}...")

        # Start with classic + engineered + cluster features
        base_feats = category_features[skill] + engineered_feats[skill]
        cluster_feats = [col for col in df.columns if col.startswith('cluster_') or col.startswith('dist_to_cluster_')]
        all_feats = list(set(base_feats + cluster_feats))  # Avoid duplicates

        # Find most correlated feature for this skill
        skill_target = f"{skill}_Category"
        corrs = {}
        for feat in all_feats:
            try:
                corrs[feat] = abs(df[feat].corr(df[skill_target]))
            except:
                corrs[feat] = 0

        # Pick feature with highest |correlation|
        top_corr_feat = max(corrs, key=corrs.get)
        print(f"  Most correlated for {skill}: {top_corr_feat} (r={corrs[top_corr_feat]:.3f})")

        # Add binary top/bottom 25% features for this
        df = create_binary_top_bottom_features(df, top_corr_feat, top_corr_feat)
        new_bins = [f"{top_corr_feat}_Top_25", f"{top_corr_feat}_Bottom_25"]
        binary_features_by_skill[skill] = new_bins

        # Add them to the feature pool
        all_feats += [f"{top_corr_feat}_Top_25", f"{top_corr_feat}_Bottom_25"]

        # --- 1. Save Correlations ---
        corr_df = pd.DataFrame({'feature': list(corrs.keys()), 'abs_corr': list(corrs.values())})
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
        csv_corr = os.path.join(STAT_TEST_FOLDER, f"{skill}_correlations.csv")
        corr_df.to_csv(csv_corr, index=False)
        # PNG
        plt.figure(figsize=(8, min(0.4*len(corr_df), 15)))
        sns.barplot(x='abs_corr', y='feature', data=corr_df.head(20))
        plt.title(f"{skill} - Top 20 Feature Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(STAT_TEST_FOLDER, f"{skill}_correlations.png"))
        plt.close()

        # --- 2. ANOVA ---
        anova_res = []
        for feat in all_feats:
            try:
                groups = [df[df[skill_target]==i][feat] for i in range(3)]
                fval, pval = f_oneway(*groups)
                anova_res.append((feat, fval, pval))
            except:
                anova_res.append((feat, np.nan, np.nan))
        anova_df = pd.DataFrame(anova_res, columns=['feature','f_stat','p_value']).sort_values('f_stat', ascending=False)
        anova_df.to_csv(os.path.join(STAT_TEST_FOLDER, f"{skill}_anova.csv"), index=False)

        # --- 3. Information Gain ---
        x = df[all_feats].fillna(0)
        y = df[skill_target]
        info_gains = mutual_info_classif(x, y)
        info_gain_df = pd.DataFrame({'feature': all_feats, 'info_gain': info_gains}).sort_values('info_gain', ascending=False)
        info_gain_df.to_csv(os.path.join(STAT_TEST_FOLDER, f"{skill}_information_gain.csv"), index=False)

        # --- 4. Chi-square ---
        chi_sq_res = []
        # Only do for binary/categorical features
        for feat in all_feats:
            try:
                if len(df[feat].unique()) <= 10 or 'Top_25' in feat or 'Bottom_25' in feat or feat.startswith('cluster_'):
                    cont = pd.crosstab(df[feat], df[skill_target])
                    chi2, pval, dof, ex = chi2_contingency(cont)
                    chi_sq_res.append((feat, chi2, pval))
            except:
                chi_sq_res.append((feat, np.nan, np.nan))
        chi_df = pd.DataFrame(chi_sq_res, columns=['feature','chi2_stat','p_value']).sort_values('chi2_stat', ascending=False)
        chi_df.to_csv(os.path.join(STAT_TEST_FOLDER, f"{skill}_chisquare.csv"), index=False)

        # Save top rows as images for quick review
        for name, df_to_save in [
            ('anova', anova_df), 
            ('information_gain', info_gain_df), 
            ('chisquare', chi_df)
        ]:
            plt.figure(figsize=(10, min(0.4*len(df_to_save), 15)))
            sns.barplot(x=df_to_save.columns[1], y='feature', data=df_to_save.head(20))
            plt.title(f"{skill} - Top 20 {name.replace('_', ' ').capitalize()}")
            plt.tight_layout()
            plt.savefig(os.path.join(STAT_TEST_FOLDER, f"{skill}_{name}.png"))
            plt.close()

        feature_results[skill] = {
            'corr': corr_df, 'anova': anova_df, 'info_gain': info_gain_df, 'chi': chi_df
        }

    print(f"\nAll statistical test results saved under: {STAT_TEST_FOLDER}")
    return df, feature_results, binary_features_by_skill

def run_statistical_tests(data, features, target_col):
    """Run all necessary statistical tests for features"""
    from scipy.stats import chi2_contingency, f_oneway
    from sklearn.feature_selection import mutual_info_classif
    
    # Ensure no NaN values
    data = data.copy()
    data[features] = data[features].fillna(0)
    
    test_results = {}
    
    # 1. Information Gain
    X = data[features]
    y = data[target_col]
    info_gains = mutual_info_classif(X, y)
    test_results['information_gain'] = dict(zip(features, info_gains))
    
    # 2. ANOVA Test
    anova_results = {}
    for feature in features:
        try:
            groups = [group[feature].values for name, group in data.groupby(target_col)]
            f_stat, p_value = f_oneway(*groups)
            anova_results[feature] = {'f_stat': f_stat, 'p_value': p_value}
        except:
            anova_results[feature] = {'f_stat': 0, 'p_value': 1}
    test_results['anova'] = anova_results
    
    # 3. Chi-squared Test for feature independence
    chi_squared_results = {}
    for i, feat1 in enumerate(features):
        for feat2 in features[i+1:]:
            try:
                # Bin continuous data for chi-square test
                feat1_binned = pd.qcut(data[feat1], q=10, duplicates='drop')
                feat2_binned = pd.qcut(data[feat2], q=10, duplicates='drop')
                contingency = pd.crosstab(feat1_binned, feat2_binned)
                chi2, p_value, _, _ = chi2_contingency(contingency)
                chi_squared_results[f'{feat1}_vs_{feat2}'] = {
                    'chi2': chi2,
                    'p_value': p_value
                }
            except:
                chi_squared_results[f'{feat1}_vs_{feat2}'] = {
                    'chi2': 0,
                    'p_value': 1
                }
    test_results['chi_squared'] = chi_squared_results
    
    return test_results

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

def save_results(category, model_name, y_true, y_pred, feature_importances, accuracy, folder_path, features_list=None):
    """Save results including confusion matrix, classification report, and feature importances as visualizations."""
    from sklearn.metrics import confusion_matrix, classification_report
    import json
    import seaborn as sns
    
    folder_path = os.path.join(OUTPUT_FOLDER, folder_path)  # Ensure folder is under OUTPUT_FOLDER
    os.makedirs(folder_path, exist_ok=True)
    
    # Save confusion matrix as PNG
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bottom 25%', 'Middle 50%', 'Top 25%'],
                yticklabels=['Bottom 25%', 'Middle 50%', 'Top 25%'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} ({category})\nAccuracy: {accuracy:.3f}')
    plt.tight_layout()
    cm_path = os.path.join(folder_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Save classification report as a styled table image
    report = classification_report(y_true, y_pred, 
                                  target_names=['Bottom 25%', 'Middle 50%', 'Top 25%'], 
                                  output_dict=True)
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Add overall accuracy to the report
    report_df.loc['accuracy', 'precision'] = accuracy
    report_df.loc['accuracy', 'recall'] = accuracy
    report_df.loc['accuracy', 'f1-score'] = accuracy
    report_df.loc['accuracy', 'support'] = len(y_true)
    
    # Create a styled table
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    
    # Round to 3 decimal places
    report_df = report_df.round(3)
    
    # Create table
    table = plt.table(
        cellText=report_df.values,
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(f'Classification Report - {model_name} ({category})', fontsize=14, pad=20)
    plt.tight_layout()
    report_path = os.path.join(folder_path, 'classification_report.png')
    plt.savefig(report_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save JSON version as well for programmatic access
    report_json_path = os.path.join(folder_path, 'classification_report.json')
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save feature importances as PNG if available
    if feature_importances is not None and features_list is not None:
        # Create a DataFrame for better visualization
        feat_imp = pd.DataFrame({
            'Feature': features_list,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features only to keep it readable
        top_n = min(20, len(feat_imp))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp.head(top_n))
        plt.title(f'Top {top_n} Feature Importances - {model_name} ({category})')
        plt.tight_layout()
        fi_path = os.path.join(folder_path, 'feature_importances.png')
        plt.savefig(fi_path)
        plt.close()
        
        # Also save the raw data
        feat_imp.to_csv(os.path.join(folder_path, 'feature_importances.csv'), index=False)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, category, class_weight_dict):
    """Train and evaluate models with voting ensemble"""
    # Define the base models with hyperparameter tuning
    param_grid = {
        'hist_gb': {
            'max_iter': [500, 1000],
            'learning_rate': [0.05, 0.1],
            'max_depth': [8, 12]
        },
        'xgb': {
            'n_estimators': [500, 1000],
            'learning_rate': [0.05, 0.1],
            'max_depth': [8, 12],
            'subsample': [0.8, 1.0]
        },
        'rf': {
            'n_estimators': [500, 1000],
            'max_depth': [8, 12]
        }
    }
    
    base_models = [
        ('hist_gb', HistGradientBoostingClassifier(random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(class_weight=class_weight_dict, random_state=42))
    ]
    
    results = {}
    
    for model_name, model in base_models:
        print(f"\nTraining {model_name.upper()} model...")

        # Precompute sample weights for this fold
        sample_weights = y_train.map(class_weight_dict) if hasattr(y_train, 'map') else np.array([class_weight_dict[yy] for yy in y_train])

        if model_name in ['hist_gb', 'xgb']:
            grid_search = GridSearchCV(model, param_grid[model_name], cv=3, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search = GridSearchCV(model, param_grid[model_name], cv=3, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Adjust predictions to favor category 1 (Middle 50%)
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test)
            close_to_cat1 = np.abs(y_proba[:, 1] - np.max(y_proba, axis=1)) < 0.2
            y_pred = np.argmax(y_proba, axis=1)
            y_pred[close_to_cat1] = 1
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        
        # Print detailed metrics
        print_detailed_metrics(y_test, y_pred, model_name)
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
        else:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
            feature_importances = perm_importance.importances_mean
        
        # Store results
        results[model_name] = {
            'model': best_model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'feature_importances': feature_importances
        }
        
        # Save results
        folder_path = os.path.join('new_methods_predictions', category, model_name)
        save_results(category, model_name, y_test, y_pred, feature_importances, accuracy, folder_path, X_train.columns.tolist())
    
    # Train and evaluate the voting ensemble
    print("\nTraining VOTING model...")
    voting_model = VotingClassifier(
        estimators=[(name, result['model']) for name, result in results.items()],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = voting_model.predict(X_test)
    
    # Adjust predictions to favor category 1 (Middle 50%)
    if hasattr(voting_model, 'predict_proba'):
        y_proba = voting_model.predict_proba(X_test)
        close_to_cat1 = np.abs(y_proba[:, 1] - np.max(y_proba, axis=1)) < 0.2
        y_pred = np.argmax(y_proba, axis=1)
        y_pred[close_to_cat1] = 1
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Print detailed metrics
    print_detailed_metrics(y_test, y_pred, 'voting')
    
    # Store results for voting model
    results['voting'] = {
        'model': voting_model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'feature_importances': None  # Voting model does not have feature importances
    }
    
    # Save results for voting model
    folder_path = os.path.join('new_methods_predictions', category, 'voting')
    save_results(category, 'voting', y_test, y_pred, None, accuracy, folder_path, X_train.columns.tolist())
    
    return results

def train_category_model(category_name, features, data, cluster_descriptions):
    print(f"\nTraining models for {category_name}...")
    print("=" * 80)
    
    data = data.copy()
    target_col = f'{category_name}_Category'
    
    # Add cluster information to features
    cluster_features = [col for col in data.columns if col.startswith('cluster_') or col.startswith('dist_to_cluster_')]
    all_features = features.copy() + cluster_features
    
    print(f"Using {len(features)} base features + {len(cluster_features)} cluster features")
    
    print("\nRunning Statistical Tests...")
    test_results = run_statistical_tests(data, all_features, target_col)
    
    print("\nFeature Information Gain:")
    sorted_info_gains = sorted(test_results['information_gain'].items(), 
                             key=lambda x: x[1], reverse=True)
    for feature, gain in sorted_info_gains[:10]:  # Show top 10 features
        print(f"{feature}: {gain:.3f}")
    
    print("\nCluster Feature Information Gain:")
    cluster_info_gains = [(feat, gain) for feat, gain in sorted_info_gains 
                        if feat.startswith('cluster_') or feat.startswith('dist_to_cluster_')]
    for feature, gain in cluster_info_gains:
        if feature.startswith('cluster_'):
            cluster_num = int(feature.split('_')[1])
            description = cluster_descriptions.get(cluster_num, "Unknown cluster type")
            print(f"{feature}: {gain:.3f} - {description}")
        else:
            cluster_num = int(feature.split('_')[-1])
            description = cluster_descriptions.get(cluster_num, "Unknown cluster type")
            print(f"{feature}: {gain:.3f} - Distance to {description}")
    
    correlations = {feature: abs(data[feature].corr(data[target_col])) 
                   for feature in all_features}
    
    print("\nFeature correlations with target (top 10):")
    sorted_correlations = sorted(correlations.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:10]
    for feature, corr in sorted_correlations:
        print(f"{feature}: {corr:.3f}")
    
    X = data[all_features]
    y = data[target_col]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track best model results (per model name)
    best_fold_results = {}
    best_fold_accuracies = {}
    best_fold_extras = {}
    
    for train_index, test_index in skf.split(X, y):
        # ----- Split -----
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # ----- Fit scaler only on training data -----
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # ----- Targeted synthetic augmentation via hardness -----
        hardness_scores = compute_knn_shapley_scores(X_train_scaled.values, y_train.values, K=5)
        X_train_res, y_train_res = targeted_synthetic_augmentation(
            X_train_scaled.values, y_train.values, hardness_scores,
            tau=0.1,  # augment 10% hardest, can tune
            synth_multiplier=2.0,  # 1x synthetic data for hard points
            method='tvae',  # or 'ctgan'
            random_state=42
        )
        # Convert back to DataFrame for downstream code
        X_train_res = pd.DataFrame(X_train_res, columns=X_train_scaled.columns)
        y_train_res = pd.Series(y_train_res)

        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_res)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
        class_weight_dict = dict(zip(classes, class_weights))
        # Manually boost the Top 25% class weight (class 2)
        class_weight_dict[2] = class_weight_dict[2] * 1.5
        class_weight_dict[0] = class_weight_dict[0] * 1.25    # Increase by 50%

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train_res)
        
        # Get results for this fold
        model_results = train_and_evaluate_models(X_train_res, X_test_scaled, y_train_res, y_test, category_name, class_weight_dict)
        
        # Track best results for each model type
        for model_name, result in model_results.items():
            acc = result['accuracy']
            if (model_name not in best_fold_accuracies) or (acc > best_fold_accuracies[model_name]):
                best_fold_accuracies[model_name] = acc
                best_fold_results[model_name] = result
                # Save the data needed for save_results later
                best_fold_extras[model_name] = {
                    "y_true": y_test,
                    "y_pred": result['predictions'],
                    "feature_importances": result['feature_importances'],
                    "features_list": X_train.columns.tolist()
                }
    
    # After all folds, save only the best fold for each model
    for model_name, result in best_fold_results.items():
        folder_path = os.path.join('new_methods_predictions', category_name, model_name)
        extras = best_fold_extras[model_name]
        save_results(
            category_name, model_name, 
            extras["y_true"], extras["y_pred"], 
            extras["feature_importances"], result['accuracy'], 
            folder_path, extras["features_list"]
        )
    
    print("\nCluster Distribution across Categories:")
    for cluster_num in range(len(cluster_descriptions)):
        cluster_col = f'cluster_{cluster_num}'
        if cluster_col in data.columns:
            cat_counts = data[data[cluster_col] == 1][target_col].value_counts().sort_index()
            
            description = cluster_descriptions.get(cluster_num, "Unknown cluster type")
            
            print(f"\nCluster {cluster_num} ({description}):")
            total = cat_counts.sum()
            for cat, count in cat_counts.items():
                cat_name = ['Bottom 25%', 'Middle 50%', 'Top 25%'][cat]
                print(f"  {cat_name}: {count} players ({count/total*100:.1f}%)")
    
    # Pick best model overall (highest accuracy)
    best_model_name = max(best_fold_results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model_result = best_fold_results[best_model_name]

    print(f"\nFeature Importance for Best Model ({best_model_name.upper()}):")
    if best_model_result['feature_importances'] is not None:
        feature_importance = dict(zip(all_features, best_model_result['feature_importances']))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_importance:
            if importance > 0.01:
                if feature.startswith('cluster_'):
                    cluster_num = int(feature.split('_')[1])
                    description = cluster_descriptions.get(cluster_num, "Unknown cluster type")
                    print(f"{feature}: {importance:.3f} - {description}")
                else:
                    print(f"{feature}: {importance:.3f}")
    else:
        print("Feature importances are not available for the voting model.")
    
    return best_fold_results, test_results

def prepare_data():
    print("Loading and preparing data...")
    nba_data = pd.read_csv('nba_player_ratings_25_99_newv2.csv')
    college_stats = pd.read_csv('college_stats_with_seasons.csv')
    
    # ---------------------------------------------------------
    # NEW: drop / ignore players whose last college season is 2018
    # ---------------------------------------------------------
    # season column is assumed to be an integer or string like "2018"
    # Keep only rows where season != 2018
    before_rows = len(college_stats)
    college_stats = college_stats[college_stats['season'] != 2018]
    # If the season column is a string, the above still works, but you could
    # also do: college_stats = college_stats[college_stats['season'] != '2018']
    after_rows = len(college_stats)
    print(f"Filtered out {before_rows - after_rows} college rows with season == 2018")
    # ---------------------------------------------------------

    college_stats = college_stats.fillna(0)

    # --- Add: One-hot encode LastConf, keep YearsPlayed as numeric ---
    if 'LastConf' in college_stats.columns:
        conf_dummies = pd.get_dummies(college_stats['LastConf'], prefix='Conf')
        college_stats = pd.concat([college_stats, conf_dummies], axis=1)
    else:
        print("Warning: 'LastConf' column not found in college_stats_final.csv")
        conf_dummies = pd.DataFrame()

    college_stats = add_advanced_features(college_stats)
    college_stats = engineer_features(college_stats)
    
    merged_data = pd.merge(college_stats, nba_data, on='Player', how='inner')
    merged_data = merged_data.fillna(0)

    merged_data, cluster_descriptions = add_clustering_features(merged_data, n_clusters=5)

    aspects = ['Playmaking', 'Rebounding', 'Defense', 'Shooter']
    for aspect in aspects:
        merged_data[f'{aspect}_Category'] = pd.qcut(
            merged_data[f'{aspect}_Score'], 
            q=[0, 0.25, 0.75, 1], 
            labels=['Bottom 25%', 'Middle 50%', 'Top 25%'],
            duplicates='drop'
        ).cat.codes
    merged_data['Scorer_Category'] = pd.qcut(
        merged_data['Scorer_Score'], 
        q=[0, 0.25, 0.75, 1], 
        labels=['Bottom 25%', 'Middle 50%', 'Top 25%'],
        duplicates='drop'
    ).cat.codes

    print("\nData Shape:", merged_data.shape)
    print("\nChecking for remaining NaN values:")
    null_counts = merged_data.isnull().sum()
    print(null_counts[null_counts > 0])
    
    # --- Add: save the list of conference dummy columns for later use ---
    conf_cols = [col for col in merged_data.columns if col.startswith('Conf_')]
    return merged_data, cluster_descriptions, conf_cols

def analyze_cluster_predictions(data, cluster_descriptions):
    print("\nAnalyzing Cluster Developmental Patterns")
    print("=" * 80)
    
    categories = ['Rebounding', 'Shooter', 'Playmaking']
    n_clusters = len(cluster_descriptions)
    
    success_rates = pd.DataFrame(index=range(n_clusters), columns=categories)
    
    for cluster in range(n_clusters):
        cluster_players = data[data['player_cluster'] == cluster]
        
        print(f"\nCluster {cluster} ({cluster_descriptions.get(cluster, 'Unknown')}):")
        
        for category in categories:
            cat_col = f'{category}_Category'
            top_category_count = sum(cluster_players[cat_col] == 2)
            total_count = len(cluster_players)
            success_rate = (top_category_count / total_count) * 100 if total_count > 0 else 0
            
            success_rates.loc[cluster, category] = success_rate
            
            print(f"  {category}: {success_rate:.1f}% become top 25% players")
    
    plt.figure(figsize=(12, 8))
    success_rates.plot(kind='bar', figsize=(12, 6))
    plt.title('Cluster Success Rates by Category')
    plt.xlabel('Cluster')
    plt.ylabel('% of Players Reaching Top 25%')
    plt.xticks(rotation=0)
    plt.legend(title='Category')
    plt.grid(axis='y', alpha=0.3)
    
    ax = plt.gca()
    ax.set_xticklabels([f"{i}: {cluster_descriptions.get(i, 'Unknown')[:30]}..." 
                        for i in range(n_clusters)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'cluster_success_rates.png'))
    plt.close()
    
    print("\nCluster success rate visualization saved as 'cluster_success_rates.png'")
    
    print("\nBest Development Path for Each Cluster:")
    for cluster in range(n_clusters):
        best_category = success_rates.loc[cluster].idxmax()
        best_rate = success_rates.loc[cluster, best_category]
        
        print(f"Cluster {cluster} ({cluster_descriptions.get(cluster, 'Unknown')}):")
        print(f"  Best category: {best_category} ({best_rate:.1f}% reach top 25%)")
        
    return success_rates

def row_to_image(row, feature_map, img_size=16):
    """Convert a 1D feature vector into a 2D image using a feature_map."""
    img = np.zeros((img_size, img_size), dtype=np.float32)
    for i, val in enumerate(row):
        x, y = feature_map[i]
        img[x, y] = val
    return img

def create_feature_map(features, method='umap', img_size=32, random_state=42):
    """Generate a pixel location for each feature using UMAP or t-SNE."""
    n_features = len(features)
    # Fake data: each feature as a one-hot row
    X = np.eye(n_features)
    
    if method == 'tsne':
        perplexity = max(2, min(5, n_features // 2))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)

    else:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    coords = reducer.fit_transform(X)
    
    # Normalize coordinates to [0, img_size-1]
    coords -= coords.min(axis=0)
    coords /= coords.max(axis=0)
    coords *= (img_size - 1)
    coords = np.round(coords).astype(int)
    # Remove duplicates by jittering
    for i in range(len(coords)):
        while tuple(coords[i]) in [tuple(c) for j, c in enumerate(coords) if j < i]:
            coords[i] += np.random.randint(-1, 2, size=2)
            coords[i] = np.clip(coords[i], 0, img_size-1)
    return coords

def tabular_to_images(df, features, method='umap', img_size=32):
    feature_map = create_feature_map(features, method=method, img_size=img_size)
    images = []
    for idx, row in df[features].iterrows():
        img = row_to_image(row.values, feature_map, img_size=img_size)
        images.append(img)
    images = np.stack(images)
    return images, feature_map

def train_cnn(images, labels, num_classes, epochs=20):
    images = images[..., np.newaxis]
    labels_cat = utils.to_categorical(labels, num_classes)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=images.shape[1:]),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    model.fit(
    images, labels_cat,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[es]
    )

    return model

def process_all_categories(data, cluster_descriptions, binary_features_by_skill):
    scaler = StandardScaler()
    results = {}
    all_test_results = {}

    for category, features in category_features.items():
        print(f"\nProcessing {category}...")
        print("=" * 80)
        
        data_scaled = data.copy()
        data_scaled[features] = scaler.fit_transform(data[features])

        # --- Add binary features here ---
        extra_binaries = binary_features_by_skill.get(category, [])
        all_feats = features + extra_binaries

        category_results, test_results = train_category_model(
            category, all_feats, data_scaled, cluster_descriptions
        )
        
        results[category] = category_results
        all_test_results[category] = test_results
        
        best_model_name = max(category_results.items(), 
                           key=lambda x: x[1]['accuracy'])[0]
        best_model_result = category_results[best_model_name]
        
        model_data = {
            'model': best_model_result['model'],
            'accuracy': best_model_result['accuracy'],
            'scaler': scaler,
            'features': all_feats,
            'cluster_features': [col for col in data.columns if col.startswith('cluster_') or col.startswith('dist_to_cluster_')],
            'test_results': test_results,
            'cluster_descriptions': cluster_descriptions
        }
        
        model_filename = os.path.join(OUTPUT_FOLDER, f'{category.lower()}_best_model_with_clusters.joblib')
        joblib.dump(model_data, model_filename)
        print(f"\nModel saved as {model_filename}")
    
    return results, all_test_results

def print_final_summary(all_results):
    print("\nFinal Results Summary")
    print("=" * 80)
    
    for category, results in all_results.items():
        print(f"\n{category} Results:")
        print("-" * 40)
        
        for model_type, result in results.items():
            print(f"{model_type}:")
            print(f"Accuracy: {result['accuracy']:.3f}")
            
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]} ({best_model[1]['accuracy']:.3f})")

def compute_knn_shapley_scores(X, y, K=5):
    # KNN distances for each sample with respect to the rest of the training data
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # Remove self-neighbor (distance=0)
    indices = indices[:, 1:]
    # For each point, how many of its neighbors have different label?
    hardness = []
    for i, neighbors in enumerate(indices):
        neighbor_labels = y[neighbors]
        diff = np.sum(neighbor_labels != y[i])
        hardness.append(diff / K)
    return np.array(hardness)

def targeted_synthetic_augmentation(X, y, hardness_scores, tau=0.1, synth_multiplier=1.0, method='tvae', random_state=42):
    # Select the hardest tau% of samples
    n_hard = int(len(X) * tau)
    hard_indices = np.argsort(hardness_scores)[-n_hard:]  # Select hardest
    X_hard = X[hard_indices]
    y_hard = y[hard_indices]
    # Prepare DataFrame for SDV
    df_hard = pd.DataFrame(X_hard, columns=[f'f{i}' for i in range(X_hard.shape[1])])
    df_hard['label'] = y_hard
    n_synth = int(n_hard * synth_multiplier)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_hard)
    # Train generator
    if method == 'ctgan':
        generator = CTGANSynthesizer(metadata, epochs=300)
    else:
        generator = TVAESynthesizer(metadata, epochs=300)
    generator.fit(df_hard)
    synth = generator.sample(n_synth)
    # Concatenate synthetic data with original training data
    X_aug = np.vstack([X, synth.drop('label', axis=1).values])
    y_aug = np.concatenate([y, synth['label'].values])
    return X_aug, y_aug

def build_player_graph(df, all_features, target_col, k_neighbors=5):
    """
    Build a graph where nodes=players, edges=clusters + KNN + conferences.
    """
    # Node features
    x = torch.tensor(df[all_features].values, dtype=torch.float)

    # Labels
    y = torch.tensor(df[target_col].values, dtype=torch.long)

    # Build edges
    edge_index = []

    # 1. Connect players in the same cluster
    for c in df['player_cluster'].unique():
        idx = df.index[df['player_cluster'] == c].tolist()
        if len(idx) > 1:
            # Instead of fully connecting, sample a few neighbors
            for i in idx:
                sampled = np.random.choice([j for j in idx if j != i],
                                           size=min(5, len(idx)-1),
                                           replace=False)
                for j in sampled:
                    edge_index.append([i, j])
    # 2. Add KNN edges (feature similarity)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine').fit(df[all_features].values)
    _, indices = nbrs.kneighbors(df[all_features].values)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip itself
            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected

    # 3. Add conference edges
    conf_cols = [c for c in df.columns if c.startswith("Conf_")]
    for conf in conf_cols:
        idx = df.index[df[conf] == 1].tolist()
        for i in idx:
            for j in idx:
                if i != j:
                    edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)


class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3, K=10, alpha=0.1):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 2-layer MLP with residual connection
        h = F.relu(self.bn1(self.lin1(x)))
        h = F.dropout(h, p=0.5, training=self.training)
        h2 = F.relu(self.bn2(self.lin2(h)))
        h = h + h2  # residual
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin3(h)
        x = self.prop(h, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)    


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.5)
        self.conv3 = GATConv(hidden_channels*heads, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def generate_synthetic_nodes(df, all_features, target_col, class_label, n_synth=100, method='tvae'):
    """Generate synthetic nodes for a specified class label."""
    class_df = df[df[target_col] == class_label].copy()
    synth_df = class_df[all_features + [target_col]]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=synth_df)

    if method == 'ctgan':
        generator = CTGANSynthesizer(metadata, epochs=300)
    else:
        generator = TVAESynthesizer(metadata, epochs=300)

    generator.fit(synth_df)
    synth = generator.sample(n_synth)

    X_synth = synth[all_features].values
    y_synth = synth[target_col].values
    return X_synth, y_synth    
    
def generate_synthetic_top_nodes(df, all_features, target_col, n_synth=100, method='tvae'):
    """Generate synthetic Top 25% nodes for GNN."""
    # Filter only Top 25% players
    top_df = df[df[target_col] == 2].copy()
    synth_df = top_df[all_features + [target_col]]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=synth_df)

    if method == 'ctgan':
        generator = CTGANSynthesizer(metadata, epochs=300)
    else:
        generator = TVAESynthesizer(metadata, epochs=300)

    generator.fit(synth_df)
    synth = generator.sample(n_synth)

    # Features and labels
    X_synth = synth[all_features].values
    y_synth = synth[target_col].values
    return X_synth, y_synth

def add_synthetic_to_graph(graph_data, X_synth, y_synth, k_neighbors=5, restrict_to_class=None):
    """Append synthetic nodes to an existing graph and connect them via KNN.
       If restrict_to_class is set, connect only to real nodes of that class.
    """
    x_synth = torch.tensor(X_synth, dtype=torch.float)
    y_synth = torch.tensor(y_synth, dtype=torch.long)

    num_existing = graph_data.num_nodes
    graph_data.x = torch.cat([graph_data.x, x_synth], dim=0)
    graph_data.y = torch.cat([graph_data.y, y_synth], dim=0)

    # Choose candidate pool for KNN
    if restrict_to_class is not None:
        mask = (graph_data.y[:num_existing] == restrict_to_class).cpu().numpy()
        pool_x = graph_data.x[:num_existing][mask].cpu().numpy()
        pool_idx = np.where(mask)[0]
    else:
        pool_x = graph_data.x[:num_existing].cpu().numpy()
        pool_idx = np.arange(num_existing)

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(pool_idx)), metric='cosine').fit(pool_x)
    _, indices = nbrs.kneighbors(x_synth.cpu().numpy())

    new_edges = []
    for i, neighbors in enumerate(indices):
        synth_idx = num_existing + i
        for j in neighbors:
            real_idx = pool_idx[j]
            new_edges.append([synth_idx, real_idx])
            new_edges.append([real_idx, synth_idx])

    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    graph_data.edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)

    return graph_data

def train_gnn(data, num_epochs=1000, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = APPNPNet(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=data.y.max().item()+1
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    num_real = len(merged_data)
    num_nodes = data.num_nodes

    perm = torch.randperm(num_real)
    train_end = int(0.8 * num_real)
    train_idx, test_idx = perm[:train_end], perm[train_end:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    data.train_mask[num_real:] = True  # synthetic only in training

    classes = np.unique(data.y.cpu().numpy())
    weights = compute_class_weight('balanced', classes=classes, y=data.y.cpu().numpy())
    weights = weights.astype(float)
    weights[2] *= 3.0  # extra boost for Top class
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = pytorch_focal_loss(out[data.train_mask], data.y[data.train_mask],
                                  alpha=class_weights, gamma=2.0)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            pred = out[data.test_mask].argmax(dim=1)
            acc = (pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.3f}")

    return model

def pytorch_focal_loss(logits, targets, alpha=None, gamma=2.0):
    """Focal loss for PyTorch GNN."""
    ce_loss = F.nll_loss(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss
    return focal.mean()

def ensemble_predictions(gnn_probs, tree_probs, alpha=0.5):
    """Weighted average ensemble: alpha*GNN + (1-alpha)*Trees"""
    return alpha * gnn_probs + (1 - alpha) * tree_probs


if __name__ == "__main__":
    print("Starting basketball player analysis with clustering and early statistical tests...")
    print("=" * 80)
    
    
    try:
        merged_data, cluster_descriptions, conf_cols = prepare_data()

        # Add 'YearsPlayed' and conference dummies to every skill's features
        for cat in category_features:
            if 'YearsPlayed' not in category_features[cat]:
                category_features[cat].append('YearsPlayed')
            for conf_col in conf_cols:
                if conf_col not in category_features[cat]:
                    category_features[cat].append(conf_col)

                    # DEEPINSIGHT / MREP-DEEPINSIGHT: CNN ON TABULAR-TO-IMAGE (READY TO RUN BLOCK)
        ################################################################################
        engineered_feats = {
    'Playmaking': ['AST_to_TOV', 'Playmaking_Impact', 'Ball_Control'],
    'Rebounding': ['Rebound_per_Minute', 'Rebound_Efficiency', 'Impact_Rebounding'],
    'Defense': ['Stocks', 'Stocks_per_40', 'Defense_Impact', 'STL_BLK_Interaction', 'STL_BLK_Impact'],
    'Shooter': ['Three_Point_Rate', 'Shot_Selection', 'Pure_Shooting'],
    'Scorer': ['Three_Point_Rate', 'Shot_Selection', 'Pure_Shooting']
        }

        merged_data, feat_results, binary_features_by_skill = run_and_save_stat_tests(
            merged_data, cluster_descriptions, category_features, engineered_feats
        )
        print("\n" + "="*80)
        print("GRAPH NEURAL NETWORK EXPERIMENTS (Shooter skill example)")
        print("="*80)

        category = "Shooter"

        # Gather full feature set
        base_feats = category_features[category]
        cluster_feats = [c for c in merged_data.columns if c.startswith("cluster_") or c.startswith("dist_to_cluster_")]
        binary_feats = binary_features_by_skill.get(category, [])
        all_features = base_feats + cluster_feats + binary_feats

        # Scale features
        scaler = StandardScaler()
        merged_data[all_features] = scaler.fit_transform(merged_data[all_features])

        # Build graph
        # Build graph
        target_col = f"{category}_Category"
        graph_data = build_player_graph(merged_data, all_features, target_col, k_neighbors=5)

        X_bot, y_bot = generate_synthetic_nodes(
            merged_data, all_features, target_col, class_label=1, n_synth=500, method='tvae'
        )
        graph_data = add_synthetic_to_graph(graph_data, X_bot, y_bot, k_neighbors=3, restrict_to_class=2)

        # --- Augment Middle 50% (class 1) ---
        X_mid, y_mid = generate_synthetic_nodes(
            merged_data, all_features, target_col, class_label=1, n_synth=200, method='tvae'
        )
        graph_data = add_synthetic_to_graph(graph_data, X_mid, y_mid, k_neighbors=3, restrict_to_class=2)

        # --- Augment Top 25% (class 2) [increase to 500] ---
        # --- Augment Top 25% (class 2) with ~200 nodes ---
        X_top, y_top = generate_synthetic_top_nodes(
            merged_data, all_features, target_col, n_synth=1200, method='tvae'
        )
        graph_data = add_synthetic_to_graph(graph_data, X_top, y_top, k_neighbors=5, restrict_to_class=2)

        print(f"Graph after augmentation: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
                # Train GNN
        gnn_model = train_gnn(graph_data, num_epochs=1000)

        # Evaluate
        gnn_model.eval()
        out = gnn_model(graph_data)
        pred = out[graph_data.test_mask].argmax(dim=1).cpu().numpy()
        true = graph_data.y[graph_data.test_mask].cpu().numpy()

        print("\nGNN Results:")
        print(classification_report(true, pred, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
        print("Confusion matrix:\n", confusion_matrix(true, pred))
        ################################################################################
        # -------------------------------------------------------------------------
        # ENSEMBLE: Combine GNN with best tabular VotingClassifier
        # -------------------------------------------------------------------------
        # Get softmax probabilities from GNN
        gnn_probs = out[graph_data.test_mask].exp().detach().cpu().numpy() # convert log-softmax to probs

        # Get tabular ensemble (VotingClassifier) from process_all_categories
        # NOTE: We need to wait until all_results is created later in the script
        # So we temporarily store true labels and gnn_probs here
        gnn_true = true
        gnn_probs_to_use = gnn_probs

        print("\n" + "="*80)
        print("DEEPINSIGHT/MREP-DEEPINSIGHT CNN EXPERIMENTS (Shooter skill example)")
        print("="*80)

        # --- 1. Choose skill/category to run DeepInsight on ---
        category = "Shooter"
        base_feats = category_features[category]

        # Add cluster features
        cluster_feats = [col for col in merged_data.columns if col.startswith("cluster_") or col.startswith("dist_to_cluster_")]

        # Add binary features for this category
        binary_feats = binary_features_by_skill.get(category, [])

        all_features = base_feats + cluster_feats + binary_feats
        print(f"Using {len(all_features)} features for CNN (Shooter).")
        labels = merged_data[f'{category}_Category'].values
        num_classes = 3
        img_size = 32



        # Scale
        scaler = StandardScaler()
        merged_data[all_features] = scaler.fit_transform(merged_data[all_features])

        # --- 3. Train/Test Split (same split for tabular + CNN) ---
        split_random_state = 42
        X_tabular = merged_data[all_features].values
        y_tabular = labels

        X_train_tab, X_test_tab, y_train, y_test = train_test_split(
            X_tabular, y_tabular, test_size=0.2, random_state=split_random_state, stratify=y_tabular
        )

        # Targeted synthetic augmentation
        hardness_scores = compute_knn_shapley_scores(X_train_tab, y_train, K=5)
        X_train_tab_aug, y_train_aug = targeted_synthetic_augmentation(
            X_train_tab, y_train, hardness_scores,
            tau=0.1, synth_multiplier=1.0, method='tvae', random_state=42
        )
        print("Original training class distribution:", np.bincount(y_train))
        print("After targeted synthetic augmentation:", np.bincount(y_train_aug))

        # --- 4. Create fixed feature maps once ---
        fmap_umap = create_feature_map(all_features, method='umap', img_size=img_size)
        fmap_tsne = create_feature_map(all_features, method='tsne', img_size=img_size)

        # Convert augmented training + test to images
        X_train_df_aug = pd.DataFrame(X_train_tab_aug, columns=all_features)
        X_test_df = pd.DataFrame(X_test_tab, columns=all_features)

        imgs_train_umap = np.stack([row_to_image(row.values, fmap_umap, img_size) for _, row in X_train_df_aug.iterrows()])
        imgs_test_umap = np.stack([row_to_image(row.values, fmap_umap, img_size) for _, row in X_test_df.iterrows()])

        imgs_train_tsne = np.stack([row_to_image(row.values, fmap_tsne, img_size) for _, row in X_train_df_aug.iterrows()])
        imgs_test_tsne = np.stack([row_to_image(row.values, fmap_tsne, img_size) for _, row in X_test_df.iterrows()])



        # --- 4. Train CNNs on Both Representations ---
        print("\nTraining CNN on UMAP images...")
        cnn_model_umap = train_cnn(imgs_train_umap, y_train_aug, num_classes, epochs=80)
        print("\nTraining CNN on t-SNE images...")
        cnn_model_tsne = train_cnn(imgs_train_tsne, y_train_aug, num_classes, epochs=80)




        # --- 5. Predict on Test Set ---
        print("\nPredicting (UMAP)...")
        probs_umap = cnn_model_umap.predict(imgs_test_umap)
        print("Predicting (t-SNE)...")
        probs_tsne = cnn_model_tsne.predict(imgs_test_tsne)

        # --- 6. Ensemble (MRep-DeepInsight): Average the Softmax Probabilities ---
        probs_ensemble = (probs_umap + probs_tsne) / 2.0
        y_pred_umap = np.argmax(probs_umap, axis=1)
        y_pred_tsne = np.argmax(probs_tsne, axis=1)
        y_pred_ensemble = np.argmax(probs_ensemble, axis=1)

        # --- 7. Print Classification Reports ---



        print("\nCNN (UMAP) Results:")
        print(classification_report(y_test, y_pred_umap, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_umap))

        print("\nCNN (t-SNE) Results:")
        print(classification_report(y_test, y_pred_tsne, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tsne))

        print("\nMRep-DeepInsight CNN (UMAP + t-SNE ensemble) Results:")
        print(classification_report(y_test, y_pred_ensemble, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_ensemble))

        # --- 8. Save Reports as PNG ---
        def save_cnn_report(report, cm, filename_prefix):
            # Save classification report as image
            df = pd.DataFrame(report).transpose().round(3)
            plt.figure(figsize=(8, 4))
            ax = plt.subplot(111, frame_on=False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            table = plt.table(
                cellText=df.values,
                rowLabels=df.index,
                colLabels=df.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title(filename_prefix + " Classification Report")
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_classification_report.png", dpi=220, bbox_inches='tight')
            plt.close()

            # Save confusion matrix as image
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Bottom 25%', 'Middle 50%', 'Top 25%'],
                        yticklabels=['Bottom 25%', 'Middle 50%', 'Top 25%'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(filename_prefix + " Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_confusion_matrix.png", dpi=220, bbox_inches='tight')
            plt.close()

        save_cnn_report(
            classification_report(y_test, y_pred_umap, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%'], output_dict=True),
            confusion_matrix(y_test, y_pred_umap),
            f"{OUTPUT_FOLDER}/deepinsight_cnn_umap"
        )
        save_cnn_report(
            classification_report(y_test, y_pred_tsne, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%'], output_dict=True),
            confusion_matrix(y_test, y_pred_tsne),
            f"{OUTPUT_FOLDER}/deepinsight_cnn_tsne"
        )
        save_cnn_report(
            classification_report(y_test, y_pred_ensemble, target_names=['Bottom 25%', 'Middle 50%', 'Top 25%'], output_dict=True),
            confusion_matrix(y_test, y_pred_ensemble),
            f"{OUTPUT_FOLDER}/deepinsight_cnn_ensemble"
        )

        print("\nDeepInsight/MRep-DeepInsight CNN results and reports saved in output_files_binary/")
        ################################################################################
        # END OF BLOCK
        ################################################################################    

        # 1. Prepare raw data
        #merged_data, cluster_descriptions = prepare_data()
        
        # 2. Run and save all statistical tests (correlations, ANOVA, info gain, chi-square)
        #    This will add binary features for each category as needed


        # save_overall_statistical_summary(feat_results)
        save_skill_statistical_tables(feat_results, category_features)
        
        # 3. Analyze cluster developmental patterns
        analyze_cluster_predictions(merged_data, cluster_descriptions)
        
        # 4. Model training and evaluation
        all_results, all_test_results = load_or_train_trees(
    merged_data, cluster_descriptions, binary_features_by_skill, force_retrain=False
)
        
        # 5. Print summary of best results
        print_final_summary(all_results)
        # -------------------------------------------------------------------------
        # Now we can do the ensemble (need tabular model from all_results)
        # -------------------------------------------------------------------------
        best_tabular = all_results["Shooter"]["voting"]["model"]
        num_real = len(merged_data)
        test_mask_real = graph_data.test_mask[:num_real].cpu().numpy()
        # Build test set for tabular model (same players as GNN test set)
        X_test_tab = merged_data.loc[test_mask_real, all_features].values
        tree_probs = best_tabular.predict_proba(X_test_tab)

        # Combine GNN + Trees
        probs_ens = ensemble_predictions(gnn_probs_to_use, tree_probs, alpha=0.5)
        ens_pred = np.argmax(probs_ens, axis=1)

        print("\nEnsemble Results (GNN + Trees):")
        print(classification_report(gnn_true, ens_pred,
                                    target_names=['Bottom 25%', 'Middle 50%', 'Top 25%']))
        print("Confusion matrix:\n", confusion_matrix(gnn_true, ens_pred))
        
        # 6. Print summary of top features by information gain for each skill
        print("\nStatistical Test Summary")
        print("=" * 80)
        for category, tests in all_test_results.items():
            print(f"\n{category} Statistical Significance:")
            print("-" * 40)
            print("Top 3 Features by Information Gain:")
            sorted_gains = sorted(tests['information_gain'].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            for feature, gain in sorted_gains:
                print(f"{feature}: {gain:.3f}")
        
        print("\nAnalysis complete!")



    
        
    except Exception as e:
        import traceback
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()
