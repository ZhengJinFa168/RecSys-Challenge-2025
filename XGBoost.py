import os

import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from tqdm.asyncio import tqdm

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
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from helping_methods import toOutput, evaluate_algorithm
from xgboost import XGBRanker
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
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20])
    evaluator_val = EvaluatorHoldout(URM_validation, cutoff_list=[20])

    candidate_generator_recommender = ItemKNNCFRecommender(URM_train)
    candidate_generator_recommender.fit()
    n_users, n_items = URM_train.shape

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'
    print(training_dataframe)
    cutoff = 30

    for user_id in tqdm(range(n_users)):
        recommendations = candidate_generator_recommender.recommend(user_id, cutoff=cutoff)
        training_dataframe.loc[user_id, "ItemID"] = recommendations
    print(training_dataframe)

    training_dataframe = training_dataframe.explode("ItemID")
    print(training_dataframe)

    URM_validation_coo = sps.coo_matrix(URM_validation)

    correct_recommendations = pd.DataFrame({"UserID": URM_validation_coo.row,
                                            "ItemID": URM_validation_coo.col})
    print(correct_recommendations)

    training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                  indicator='Exist')
    print(training_dataframe)

    training_dataframe["Label"] = training_dataframe["Exist"] == "both"
    training_dataframe.drop(columns=['Exist'], inplace=True)
    print(training_dataframe)

    topPop = TopPop(URM_train)
    topPop.fit()

    p3alpha = P3alphaRecommender(URM_train)
    p3alpha.fit()

    slimElasticNetRecommender = SLIMElasticNetRecommender(URM_train)
    easyRecommender = EASE_R_Recommender(URM_train)

    test = False
    if not test:
        file_path = "best_models_train/"
        if (os.path.exists(file_path + "bestSLIMElasticNetRecommender.zip")):
            print("SLIMElasticNetRecommender is already trained")
            slimElasticNetRecommender.load_model(folder_path=file_path, file_name="bestSLIMElasticNetRecommender.zip")
        else:
            slimElasticNetRecommender.fit(topK=436, alpha=0.001239600142319664, l1_ratio=0.001002639662685697)
            slimElasticNetRecommender.save_model(folder_path="best_models_train/",
                                                 file_name="bestSLIMElasticNetRecommender")
        if (os.path.exists(file_path + "bestEASYR_Recommender.zip")):
            print("EASYR_Recommender is already trained")
            easyRecommender.load_model(folder_path=file_path, file_name="bestEASYR_Recommender.zip")
        else:
            easyRecommender.fit(topK=100, l2_norm=1e3, normalize_matrix=False)
            easyRecommender.save_model(folder_path="best_models_train/", file_name="bestEASYR_Recommender")
    else:
        slimElasticNetRecommender.fit(topK=436, alpha=0.001239600142319664, l1_ratio=0.001002639662685697)
        easyRecommender.fit(topK=100, l2_norm=1e3, normalize_matrix=False)

    other_algorithms = {
        "TopPop": topPop,
        "P3alpha": p3alpha,
        "EASYR_Recommender": easyRecommender,
        "SlimElasticNet": slimElasticNetRecommender,
    }

    training_dataframe = training_dataframe.set_index('UserID')

    for user_id in tqdm(range(n_users)):
        for rec_label, rec_instance in other_algorithms.items():
            item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()

            all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)

            training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns={"index": "UserID"})

    item_popularity = np.ediff1d(sps.csc_matrix(URM_train).indptr)

    training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

    user_popularity = np.ediff1d(sps.csr_matrix(URM_train).indptr)

    training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

    training_dataframe = training_dataframe.set_index('ItemID')
    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns={"index": "ItemID"})

    training_dataframe = training_dataframe.sort_values("UserID").reset_index()
    training_dataframe.drop(columns=['index'], inplace=True)

    groups = training_dataframe.groupby("UserID").size().values

    n_estimators = 50
    learning_rate = 1e-1
    reg_alpha = 1e-1
    reg_lambda = 1e-1
    max_depth = 5
    max_leaves = 0
    grow_policy = "depthwise"
    objective = "pairwise"
    booster = "gbtree"
    use_user_profile = False
    random_seed = None

    XGB_model = XGBRanker(objective='rank:{}'.format(objective),
                          n_estimators=int(n_estimators),
                          random_state=random_seed,
                          learning_rate=learning_rate,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          max_depth=int(max_depth),
                          max_leaves=int(max_leaves),
                          grow_policy=grow_policy,
                          verbosity=0,  # 2 if self.verbose else 0,
                          booster=booster,
                          )

    y_train = training_dataframe["Label"]
    X_train = training_dataframe.drop(columns=["Label"])

    XGB_model.fit(X_train,
                  y_train,
                  group=groups,
                  verbose=True)

    # Let's say I want to compute the prediction for a group of user-item pairs, for simplicity I will use a slice of the data used
    # for training because it already contains all the features
    X_to_predict = X_train[X_train["UserID"] == 10]

    XGB_model.predict(X_to_predict)

if __name__ == "__main__":
    main()