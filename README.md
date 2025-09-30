NCAA-to-NBA Player Projection

This project analyzes how NCAA player performance translates into NBA outcomes. The workflow combines feature engineering, clustering, and synthetic data augmentation to train ensemble machine learning models that predict whether players will land in the Top 25%, Middle 50%, or Bottom 25% of NBA performance outcomes.

Repository Structure:

NBA-Draft-prediction/
   README.md – Project description and usage instructions
   requirements.txt – Python dependencies
   data/ – Input data (CSV files)
      college_stats_final.csv
      nba_player_ratings_25_99_newv2.csv
   src/ – Source code
      predict_success.py – Main analysis pipeline
   
   cluster_elbow_method.png
   player_clusters.png
   statistical_tests/...
   new_methods_predictions/...
   accuracy_summary_barplot.png


Setup
Clone the repository and install the dependencies:

> git clone https://github.com/haddadyousef/NBA-Draft-prediction
cd NBA-Draft-prediction
pip install -r requirements.txt

Python version: 3.12 or later
Core libraries installed in requirements.txt: scikit-learn, tensorflow, torch, torch-geometric, sdv, xgboost, umap-learn, seaborn, matplotlib

Running the Pipeline
To execute the full workflow, run:

python src/predict_success.py

The pipeline performs the following steps:

Data preparation

Load and merge NCAA and NBA datasets
Apply feature engineering (per-40 stats, ratios, interaction effects)
Add YearsPlayed and conference dummy variables
Player clustering

Perform PCA + KMeans clustering on NCAA features
Generate cluster visualizations and elbow method plots
Statistical analysis

Run correlation, ANOVA, information gain, and chi-square tests
Save results as CSVs and annotated plots under results/statistical_tests
Model training per skill category

Skills: Rebounding, Shooting, Playmaking, Defense
Models: Random Forest, HistGradientBoosting, XGBoost, and a Voting Ensemble
Uses stratified 5-fold cross-validation
Includes SMOTE oversampling and synthetic generation with TVAE/CTGAN
Adds fold-specific cluster features
Evaluation

Save confusion matrices, classification reports, and feature importances
Save Skill Translation Index (STI) probabilities
Produce summary barplots of model accuracies
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
   STI probabilities CSVs
   Overall results
   
   accuracy_summary_barplot.png: grouped bar chart comparing model accuracies


Key Findings

   NCAA archetype clusters reveal meaningful developmental patterns such as “high DWS / high WS” defenders or “high AST / high STL / high TOV” playmakers.
   
   Certain clusters, such as strong rebounders or defenders, translate into NBA Top 25% success more frequently.
   
   Synthetic augmentation helps address class imbalance and improve predictions for rare Top 25% outcomes.
   
   Voting ensembles consistently outperform single models across skill categories.
