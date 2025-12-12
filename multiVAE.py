import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender, \
    ScaledPureSVDRecommender
from Recommenders.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from Recommenders.MatrixFactorization.SVDFeatureRecommender import SVDFeature
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch, \
    MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from helping_methods import toOutput, evaluate_algorithm

import time
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

    recommender = MultVAERecommender_PyTorch_OptimizerMask(URM_train, use_gpu=True)

    epochs = trial.suggest_int('epochs', 50, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.2, 0.7)
    total_anneal_steps = trial.suggest_int('total_anneal_steps', 50000, 300000)
    anneal_cap = trial.suggest_float('anneal_cap', 0.1, 0.5)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-4, 1e-2)
    sgd_mode = trial.suggest_categorical('sgd_mode', ['adam', 'sgd','rmsprop'])

    # Architecture (using OptimizerMask version):
    encoding_size = trial.suggest_int('encoding_size', 50, 200)
    next_layer_size_multiplier = trial.suggest_float('next_layer_size_multiplier', 1.5, 3.0)
    max_n_hidden_layers = trial.suggest_int('max_n_hidden_layers', 1, 3)

    recommender.fit(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, dropout=dropout,
                    total_anneal_steps=total_anneal_steps, anneal_cap=anneal_cap, l2_reg=l2_reg, sgd_mode=sgd_mode,
                    encoding_size=encoding_size, next_layer_size_multiplier=next_layer_size_multiplier,max_n_hidden_layers=max_n_hidden_layers)


    score, _ = evaluator_test.evaluateRecommender(recommender)
    recall = score['RECALL']
    # 1.4. Return the score (Optuna minimizes by default, so we return 1 - accuracy)
    return recall # Or use direction='maximize' in create_study

def main():


    start_time = time.time()

    study = optuna.create_study(
        direction='maximize',
        study_name='MultiVAE',
        storage='sqlite:///OptunaStudies/MultiVAERecommender.db',  # This saves to a file
    )
    study.optimize(objective, n_trials=10)  # Run 50 trials
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
