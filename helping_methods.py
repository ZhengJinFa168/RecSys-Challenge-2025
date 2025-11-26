import numpy as np
import pandas as pd

def precision(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score

def AP(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return ap_score

def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0

    num_eval = 0

    for user_id in range(URM_test.shape[0]):

        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval

    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, MAP))

def toOutput(users_to_test, recommender, outputFile, cutoff_list=20):

    df = pd.DataFrame(columns=['user_id', 'item_list'])

    for user_id in users_to_test:
        #print(recommender.recommend(user_id, at=cutoff_list))
        # Convert item lists to space-separated strings
        
        recommendations = recommender.recommend(user_id, cutoff=cutoff_list, remove_seen_flag= True)

        item_string = ' '.join(map(str, recommendations))

        df.loc[user_id] = [user_id, item_string]

    df.to_csv(outputFile, index=False)
    print("Results saved to " + outputFile)