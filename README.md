download all contents - make sure to save them in the same folder

install required packages for code to compile by running this line in terminal:


pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn

UPDATED COMMANDS FOR GPU INTEGRATION:
conda create -n basketball_gpu python=3.9
conda activate basketball_gpu
conda install -c rapidsai -c nvidia -c conda-forge cuml=23.12 python=3.9 cuda-version=11.8
conda install -c conda-forge xgboost
conda install -c conda-forge catboost

then run

while running, double check if folders are being created for Shooting, Rebouding, etc. 

  Each of those should have 3 subfolders: curf,xgb,catboost.
  
    Each of those should have 3 subfolders inside them: plots, models, results.
    
  Make sure those folders are being updated continuously while running.
