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

def main():
    URM_all_dataframe = pd.read_csv('data/data_train.csv')
    users_to_test=pd.read_csv('data/data_target_users_test.csv')
    user_id_list = users_to_test['user_id'].tolist()

    URM_all_dataframe.columns = ["UserID", "ItemID"]
    URM_all_dataframe['hasInteraction'] = 1
    URM_all = sps.csr_matrix((URM_all_dataframe['hasInteraction'].values,
                              (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))

    URM_train_complete, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_complete, train_percentage=0.80)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20])
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[20])

    start_time = time.time()

    recommender = IALSRecommender(URM_train_complete)
    recommender.fit(num_factors=86, alpha=3.17051677957448, epsilon=0.039873271755949916,reg=0.7743602221283774,init_mean=2.6486507999035966,init_std=2.3446108787910958,confidence_scaling='log')
    recommender.save_model(folder_path="best_models_train/", file_name="bestIALSRecommender")

    print(evaluator_validation.evaluateRecommender(recommender))

    #outputFile = "outputEASYRRecommender.csv"
    #toOutput(user_id_list, recommender, outputFile)

    end_time = time.time()

    n_users_to_test=len(users_to_test)
    print("Reasonable implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))


if __name__ == "__main__":
    main()