import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

def objective(trial):
    x = pd.read_csv("data/X_train.csv", keep_default_na = False)
    y = pd.read_csv("data/y_train.csv", keep_default_na = False).values.ravel()

    rf_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    n_estimators_max = trial.suggest_int("n_estimators", 50, 1000)
    criterion_list = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # Additional parameters
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.5, step=0.01)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.01)
    oob_score = trial.suggest_categorical('oob_score', [True, False])
    warm_start = trial.suggest_categorical('warm_start', [True, False])
    random_state = trial.suggest_int('random_state', 1, 42)

    rf = RandomForestClassifier(
        n_estimators=n_estimators_max,
        max_depth=rf_max_depth,
        criterion=criterion_list,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,  # Set bootstrap to True
        oob_score=oob_score,
        ccp_alpha=ccp_alpha,
        min_impurity_decrease=min_impurity_decrease,
        warm_start=warm_start
    )

    kf = KFold(n_splits=5)
    score = cross_val_score(rf, x, y, cv=kf, scoring='accuracy')
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=100, n_jobs=-1)
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    print(study.best_trial)
