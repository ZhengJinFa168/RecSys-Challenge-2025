import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from helping_methods import toOutput, evaluate_algorithm
from SLIM_MSE_fastest import train_multiple_epochs
import os
import time



class ItemKNNCFRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, item_item_S):

        self.W_sparse = item_item_S

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

def main():
    URM_all_dataframe = pd.read_csv('data/data_train.csv')
    users_to_test=pd.read_csv('data/data_target_users_test.csv')
    user_id_list = users_to_test['user_id'].tolist()

    URM_all_dataframe.columns = ["UserID", "ItemID"]
    URM_all_dataframe['hasInteraction'] = 1.0
    URM_all = sps.csr_matrix((URM_all_dataframe['hasInteraction'].values,
                              (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    #URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    start_time = time.time()

    recommender = ItemKNNCFRecommender(URM_train)

    if os.path.exists('my_slim_model.npy'):
        # Load the model
        item_item_S = np.load('my_slim_model.npy')
    else:
        print("File doesn't exist, training model...")
        # Train the model
        item_item_S, final_loss, samples_per_second = train_multiple_epochs(URM_train, 0.0005, 20)
        # Save the model
        np.save('my_slim_model.npy', item_item_S)

    sparse_item_item = sps.csr_matrix(item_item_S)
    recommender.fit(sparse_item_item)

    outputFile = "outputMSE.csv"
    #toOutput(user_id_list, recommender, outputFile,testMode=False)

    evaluate_algorithm(URM_test, recommender)

    end_time = time.time()

    n_users_to_test=len(users_to_test)
    print("Reasonable implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))


if __name__ == "__main__":
    main()