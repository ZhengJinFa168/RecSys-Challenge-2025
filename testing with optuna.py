import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender, \
    ScaledPureSVDRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from helping_methods import toOutput, evaluate_algorithm
import optuna
import time
import optuna.visualization as vis


def objective(trial):
    URM_all_dataframe = pd.read_csv('data/data_train.csv')
    users_to_test = pd.read_csv('data/data_target_users_test.csv')
    user_id_list = users_to_test['user_id'].tolist()

    URM_all_dataframe.columns = ["UserID", "ItemID"]
    URM_all_dataframe['hasInteraction'] = 1
    URM_all = sps.csr_matrix((URM_all_dataframe['hasInteraction'].values,
                              (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20])

    recommender = ScaledPureSVDRecommender(URM_train)

    # 1.1. Suggest hyperparameters
    num_factors = trial.suggest_int('num_factors', 10, 1000)
    scaling_items = trial.suggest_float('scaling_items', 0.5, 1.5)
    scaling_users = trial.suggest_float('scaling_users', 0.5, 1.5)

    recommender.fit(num_factors = num_factors, random_seed = 1, scaling_items = scaling_items, scaling_users = scaling_users)

    score, _ = evaluator_test.evaluateRecommender(recommender)
    recall = score['RECALL']
    # 1.4. Return the score (Optuna minimizes by default, so we return 1 - accuracy)
    return recall # Or use direction='maximize' in create_study

def main():


    start_time = time.time()

    study = optuna.create_study(
        direction='maximize',
        study_name='ITEMKNNRecommender1',
        storage='sqlite:///OptunaStudies/ITEMKNNRecommender.db',  # This saves to a file
    )
    study.optimize(objective, n_trials=200)  # Run 50 trials
    # 3. Print the results
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (1 - Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # To get the best accuracy:
    best_accuracy = 1.0 - trial.value
    print(f"\nBest Cross-Validated Accuracy: {best_accuracy:.4f}")

    # Plot the optimization history
    vis.plot_optimization_history(study).show()

    # Plot the parameter importances
    vis.plot_param_importances(study).show()

    # Plot a slice of the parameters vs the objective value
    vis.plot_slice(study).show()

    # Plot the parallel coordinates
    vis.plot_parallel_coordinate(study).show()

    outputFile = "output.csv"

    end_time = time.time()

if __name__ == "__main__":
    main()