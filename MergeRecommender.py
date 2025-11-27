"""
Created on 23/12/2022

@author: Andrea Riboni
"""
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class MergeSimilarity(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarities, verbose = True):
        super(MergeSimilarity, self).__init__(URM_train, verbose = verbose)

        self.similarities = Similarities

        reference_shape = self.similarities[0].shape
        for similarity in self.similarities:
            if similarity.shape != reference_shape:
                raise ValueError("Similarities have different size")

        # CSR is faster during evaluation
        for i in range(len(self.similarities)):
            self.similarities[i] = check_matrix(self.similarities[i].copy(), 'csr')

    def fit(self, topK, coefficients):
        self.topK = topK
        self.coefficients = coefficients

        temp_W_sparse = self.similarities[0] - self.similarities[0]
        for i in range(len(self.coefficients)):
            temp_W_sparse += self.similarities[i] * self.coefficients[i]

        self.W_sparse = similarityMatrixTopK(temp_W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


"""
Created on 23/12/2022

@author: Andrea Riboni
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class DifferentLossRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Recommenders, verbose = True):
        super(DifferentLossRecommender, self).__init__(URM_train, verbose = verbose)
        self.recommenders = Recommenders

    def fit(self, coefficients):
        self.coefficients = coefficients

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        item_weights = self.recommenders[0]._compute_item_score(user_id_array) - self.recommenders[0]._compute_item_score(user_id_array)
        for i in range(len(self.recommenders)):
            item_weights += self.recommenders[1]._compute_item_score(user_id_array) * self.coefficients[i]

        return item_weights