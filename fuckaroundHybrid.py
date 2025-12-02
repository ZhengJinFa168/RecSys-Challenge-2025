import numpy as np
import os
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from MergeRecommender import DifferentLossRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
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

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20])

    start_time = time.time()
    recommenders=[]

    slimElasticNetRecommender=SLIMElasticNetRecommender(URM_all)
    easyRecommender=EASE_R_Recommender(URM_all)

    test=True
    if not test:
        file_path= "best_models_train/"
        if(os.path.exists(file_path + "bestSLIMElasticNetRecommender.zip")):
            print("SLIMElasticNetRecommender is already trained")
            slimElasticNetRecommender.load_model(folder_path=file_path, file_name="bestSLIMElasticNetRecommender.zip")
        else:
            slimElasticNetRecommender.fit(topK=436, alpha = 0.001239600142319664, l1_ratio=0.001002639662685697)
            slimElasticNetRecommender.save_model(folder_path="best_models_train/", file_name="bestSLIMElasticNetRecommender")
        if (os.path.exists(file_path + "bestEASYR_Recommender.zip")):
            print("EASYR_Recommender is already trained")
            easyRecommender.load_model(folder_path=file_path, file_name="bestEASYR_Recommender.zip")
        else:
            easyRecommender.fit(topK=100, l2_norm=1e3, normalize_matrix=False)
            easyRecommender.save_model(folder_path="best_models_train/", file_name="bestEASYR_Recommender")
    else:
        slimElasticNetRecommender.fit(topK=436, alpha=0.001239600142319664, l1_ratio=0.001002639662685697)
        easyRecommender.fit(topK=100, l2_norm=1e3, normalize_matrix=False)
    #print(slimElasticNetRecommender._compute_item_score(users_to_test))
    #print(easyRecommender._compute_score_W_dense(users_to_test))
    recommenders.append(slimElasticNetRecommender)
    recommenders.append(easyRecommender)

    coefficients=[499.9522535401951,337.0469866084344]

    finalRecommender = DifferentLossRecommender(URM_all,recommenders)

    finalRecommender.fit(coefficients=coefficients)
    #finalRecommender.save_model(folder_path="./saved_models/" ,file_name="DifferentLossRecommender")

    print(evaluator_test.evaluateRecommender(finalRecommender))

    outputFile = "DifferentLossRecommender.csv"
    toOutput(user_id_list, finalRecommender, outputFile)

    end_time = time.time()

    n_users_to_test=len(users_to_test)
    print("Reasonable implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))


if __name__ == "__main__":
    main()