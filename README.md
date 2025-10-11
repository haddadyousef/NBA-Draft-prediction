NCAA-to-NBA Player Projection

This project analyzes how NCAA player performance translates into NBA outcomes. The workflow combines feature engineering, clustering, and synthetic data augmentation to train ensemble machine learning models that predict whether players will land in the Top 25%, Middle 50%, or Bottom 25% of NBA performance outcomes.

Repository Structure:

NBA-Draft-prediction/
   README.md – Project description and usage instructions
   requirements.txt – Python dependencies
   data/ – Input data (CSV files)
      college_stats_final.csv
      nba_player_ratings_25_99_newv2.csv
   predict_success.py – Main analysis pipeline

   
   final_runthru_SSAC - Output Data
      new_methods_predictions/ - model results
      
      statistical_tests/ - ANOVA, Chi-sq, correlation, and information gain tests for each skill
      
      final_runthru_SSAC/ - SVI probabilities
      
         new_methods_predictions/(skill)/(model)
         
         sti_probabilities.csv - SVI prediction probabilities for each class
         
      accuracy_summary_barplot.png
      
      cluster_elbow_method.png
      
      cluster_success_rates.png
      
      player_clusters.png
   
   future runs will be saved to OUTPUT_FOLDER  



Setup

This project requires Python 3.12  

It is recommended to use a virtual environment for isolation.

1. Create a virtual environment


From the project root:

brew install python@3.12 for Mac or winget install Python.Python.3.12 for Windows

python3.12 -m venv venv312

2. Activate the environment

macOS/Linux:

source venv312/bin/activate

venv312\Scripts\activate

3. Install dependencies

With the environment activated, run:


pip install --upgrade pip


pip install -r requirements.txt
Once the environment is set up, you can run the pipeline with:

python predict_success.py

Clone the repository and install the dependencies:

> git clone https://github.com/haddadyousef/NBA-Draft-prediction

cd NBA-Draft-prediction

pip install -r requirements.txt

Python version: 3.12 or later

Core libraries installed in requirements.txt: scikit-learn, tensorflow, torch, torch-geometric, sdv, xgboost, umap-learn, seaborn, matplotlib

Running the Pipeline

To execute the full workflow, run:

predict_success.py

The pipeline performs the following steps:

Data preparation:

-Loads and merges NCAA and NBA datasets

-Applies feature engineering (per-40 stats, ratios, interaction effects)

-Adds YearsPlayed and conference dummy variables

-Player clustering

P-erforms PCA + KMeans clustering on NCAA features

-Generates cluster visualizations and elbow method plots

Statistical analysis:

-Runs correlation, ANOVA, information gain, and chi-square tests

-Saves results as CSVs and annotated plots inside statistical_tests folder


Skills: Rebounding, Shooting, Playmaking, Defense

Models: Random Forest, HistGradientBoosting, XGBoost, and a Voting Ensemble

Uses stratified 5-fold cross-validation

Includes SMOTE oversampling and synthetic generation with TVAE/CTGAN

Adds fold-specific cluster features


Evaluation:

-Saves confusion matrices, classification reports, and feature importances

Saves Skill Translation Index (STI) probabilities

Produces summary barplots of model accuracies

All outputs are written to the results folder.

Outputs:


Cluster analyses

   cluster_elbow_method.png: optimal number of clusters
   
   player_clusters.png: NCAA archetype visualization with PCA

Statistical tests

   CSVs and barplots for correlations, ANOVA, information gain, chi-square
   Statistical summary tables per skill

Model evaluation

   Confusion matrices (PNG)
   Classification reports (PNG and JSON)
   Feature importance plots and CSV tables
   SVI probabilities CSVs
   Overall results
   
   accuracy_summary_barplot.png: grouped bar chart comparing model accuracies


Key Findings

   NCAA archetype clusters reveal meaningful developmental patterns such as “high DWS / high WS” defenders or “high AST / high STL / high TOV” playmakers.
   
   Certain clusters, such as strong rebounders or defenders, translate into NBA Top 25% success more frequently.
   
   Synthetic augmentation helps address class imbalance and improve predictions for rare Top 25% outcomes.
   

