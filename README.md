This repository contains the code for the paper **TITLE**.  The purpose of this project is to build predictive models to identify key features that predict face mask usage and other, general health behaviours in Australia during the COVID-19 pandemic.  Particularly, we investigate the common and distinct predictive features that predict face masks and general health behaviours before mandates came into place and after.

The repository folders are as follows:

1. `analysis` - Miscellaneous exploratory analysis of data and results
2. `archive` - Older versions of data
3. `code` - Python code to reproduce results
4. `data` - All data generated from the files in `code`
5. `figures` - Folder where figures are saved
6. `help_functions` - Misc. python scripts
7. `HPC_Versions` - Versions of the python code set up to run on an HPC SLURM system.
8. `R` - R code to reproduce figures
9. `raw_data` - YouGov and OxCGRT data.
10. `references` - Misc references
11. `results` - Results from python code

The python code has the following functions:

1. `00_mask_mandates.py`-`3_split_data.py` generate the testing and training data.
2. `4_logistic_regression.py` cross validates a logistic regression model.
3. `(5-7)* * _ *.py` fit the binary tree, xgboost, and random forest models respectively
4. `* * (a-d)_ *.py` investigate hyper-parameter importance, tuning, optimal model selection, and model cross validation respectively.
5. `* (a-f) * _ *.py` fit the models for the following (respective) datasets: face masks (FM) all time, general protective health behaviours (GP) all time, FM before mandates, GP before mandates, FM after mandates, GP after mandates.
6. `8 * _ *.py` collects all model results for each data set for comparison
7. `9_permute_feature_importance.py` calculates the sensitivity of upsampling on feature importance.
8. `10_fit_models.py`-`11_test_metrics.py` fits and tests the optimal models.