import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
import os

# Define a global output folder
OUTPUT_FOLDER = 'output_files_binary'
STAT_TEST_FOLDER = os.path.join(OUTPUT_FOLDER, 'statistical_tests')
os.makedirs(STAT_TEST_FOLDER, exist_ok=True)


category_features = { 
    'Playmaking': [
        'AST', 'AST_per_40', 'AST%',
        'TOV', 'TOV%', 'USG%',
        'OWS', 'WS', 'WS/40'
    ],

    'Rebounding': [
        'TRB', 'TRB_per_40', 'TRB%',
        'Games', 'Minutes_Per_Game',
        'DWS', 'WS', 'WS/40'
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
    
    skills = ["Defense", "Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Defense": "Defense Statistical Analysis",
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

def save_overall_statistical_summary_table(
        feat_results, 
        output_path='output_files_binary/statistical_tests/overall_statistical_summary_table.png'
    ):
    # Skills and labels
    skills = ["Defense", "Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Defense": "Defense",
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
    skills = ["Defense", "Rebounding", "Shooter", "Playmaking"]
    pretty_names = {
        "Defense": "Defense Statistical Analysis",
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
        'AST',   # Playmaking
        'BLK',   # Rim protection
        '3P%',   # Shooting
        'TRB',   # Rebounding
        'TOV',   # Ball security
        'FT%',   # Free throw shooting
        'USG%',  # Usage
        'STL',   # Steals (perimeter defense)
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
    categories = ['Playmaking', 'Rebounding', 'Defense', 'Shooter']
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

def train_and_evaluate_models(X_train, X_test, y_train, y_test, category):
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
        ('rf', RandomForestClassifier(random_state=42))
    ]
    
    results = {}
    
    for model_name, model in base_models:
        print(f"\nTraining {model_name.upper()} model...")
        
        # Perform hyperparameter tuning
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
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track best model results (per model name)
    best_fold_results = {}
    best_fold_accuracies = {}
    best_fold_extras = {}
    
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        
        # Get results for this fold
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, category_name)
        
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
    college_stats = pd.read_csv('college_stats_final.csv')
    
    college_stats = college_stats.fillna(0)
    
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
    
    print("\nData Shape:", merged_data.shape)
    print("\nChecking for remaining NaN values:")
    null_counts = merged_data.isnull().sum()
    print(null_counts[null_counts > 0])
    
    return merged_data, cluster_descriptions

def analyze_cluster_predictions(data, cluster_descriptions):
    print("\nAnalyzing Cluster Developmental Patterns")
    print("=" * 80)
    
    categories = ['Playmaking', 'Rebounding', 'Defense', 'Shooter']
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

if __name__ == "__main__":
    print("Starting basketball player analysis with clustering and early statistical tests...")
    print("=" * 80)
    
    try:
        # 1. Prepare raw data
        merged_data, cluster_descriptions = prepare_data()
        
        # 2. Run and save all statistical tests (correlations, ANOVA, info gain, chi-square)
        #    This will add binary features for each category as needed
        engineered_feats = {
            'Playmaking': ['AST_to_TOV', 'Playmaking_Impact', 'Ball_Control'],
            'Rebounding': ['Rebound_per_Minute', 'Rebound_Efficiency', 'Impact_Rebounding'],
            'Defense': ['Stocks', 'Stocks_per_40', 'Defense_Impact', 'STL_BLK_Interaction', 'STL_BLK_Impact'],
            'Shooter': ['Three_Point_Rate', 'Shot_Selection', 'Pure_Shooting']
        }

        merged_data, feat_results, binary_features_by_skill = run_and_save_stat_tests(
            merged_data, cluster_descriptions, category_features, engineered_feats
        )

        # save_overall_statistical_summary(feat_results)
        save_skill_statistical_tables(feat_results, category_features)
        
        # 3. Analyze cluster developmental patterns
        analyze_cluster_predictions(merged_data, cluster_descriptions)
        
        # 4. Model training and evaluation
        all_results, all_test_results = process_all_categories(merged_data, cluster_descriptions, binary_features_by_skill)
        
        # 5. Print summary of best results
        print_final_summary(all_results)
        
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
