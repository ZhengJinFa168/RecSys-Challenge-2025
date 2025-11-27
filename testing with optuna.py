import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
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

    recommender = EASE_R_Recommender(URM_train)
    # 1.1. Suggest hyperparameters
    topK = trial.suggest_int('topK', 50, 1000)
    l2_norm = trial.suggest_float('l1_ratio', 0.000001, 1)

    recommender.fit(l2_norm= 1e3, topK = 200)

    score, _ = evaluator_test.evaluateRecommender(recommender)
    recall = score['RECALL']
    # 1.4. Return the score (Optuna minimizes by default, so we return 1 - accuracy)
    return recall # Or use direction='maximize' in create_study

def main():


    start_time = time.time()

    study = optuna.create_study(
        direction='maximize',
        study_name='SLIMElasticNetRecommender',
        storage='sqlite:///OptunaStudies/SLIME_optimization.db',  # This saves to a file
    )
    study.optimize(objective, n_trials=20)  # Run 50 trials
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

    print("ciao")
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