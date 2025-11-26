import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class HybridRecommender(BaseRecommender):
    def __init__(self, main_recommender, fallback_recommender, URM_train, min_ratings_threshold=5):
        super(HybridRecommender, self).__init__(URM_train, False)
        self.main_recommender = main_recommender
        self.fallback_recommender = fallback_recommender
        self.min_ratings_threshold = min_ratings_threshold
        self.user_ratings_count = np.array(URM_train.sum(axis=1)).flatten()

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        recommendations = []
        all_scores = []

        for user_id in user_id_array:
            if self.user_ratings_count[user_id] >= self.min_ratings_threshold:

                # Use main recommender
                if return_scores:
                    user_recs, user_scores = self.main_recommender.recommend(
                        [user_id], cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                        items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                        remove_custom_items_flag=remove_custom_items_flag, return_scores=True
                    )
                    all_scores.append(user_scores[0])  # Get scores for this single user
                else:
                    user_recs = self.main_recommender.recommend(
                        [user_id], cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                        items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                        remove_custom_items_flag=remove_custom_items_flag, return_scores=False
                    )
            else:
                print("CIAO")
                # Use fallback recommender
                if return_scores:
                    user_recs, user_scores = self.fallback_recommender.recommend(
                        [user_id], cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                        items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                        remove_custom_items_flag=remove_custom_items_flag, return_scores=True
                    )
                    all_scores.append(user_scores[0])  # Get scores for this single user
                else:
                    user_recs = self.fallback_recommender.recommend(
                        [user_id], cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                        items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                        remove_custom_items_flag=remove_custom_items_flag, return_scores=False
                    )

            recommendations.append(user_recs[0])  # Get recommendations for this single user

        # Return single list for one user, instead of list of lists
        if single_user:
            recommendations = recommendations[0]

        if return_scores:
            # Convert list of score arrays to a 2D array
            scores_batch = np.array(all_scores)
            return recommendations, scores_batch
        else:
            return recommendations