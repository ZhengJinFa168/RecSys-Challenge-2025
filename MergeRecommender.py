from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import numpy as np

class DifferentLossRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"

    def __init__(self, URM_train, Recommenders, verbose=True, normalization_method="minmax"):
        """
        Args:
            URM_train: User-Rating Matrix
            Recommenders: List of recommender objects
            verbose: Verbosity flag
            normalization_method: Method for normalization:
                - "minmax": Scale each recommender's scores to [0, 1]
                - "zscore": Standardize to mean=0, std=1
                - "max": Divide by maximum value
                - "sum": Divide by sum of all scores
                - "l2": L2 normalization (divide by Euclidean norm)
                - None: No normalization
        """
        super(DifferentLossRecommender, self).__init__(URM_train, verbose=verbose)
        self.recommenders = Recommenders
        self.normalization_method = normalization_method
        self.normalization_stats = {}  # Store normalization parameters for each recommender
        # Compute normalization statistics for each recommender
        if self.normalization_method is not None:
            self._compute_normalization_stats()

    def fit(self, coefficients):
        """
        Fit the hybrid recommender.
        
        Args:
            coefficients: List of weights for each recommender
            fit_recommenders: If True, fit all base recommenders first
        """
        # Validate coefficients length matches number of recommenders
        if len(coefficients) != len(self.recommenders):
            raise ValueError(f"Number of coefficients ({len(coefficients)}) doesn't match number of recommenders ({len(self.recommenders)})")
        
        self.coefficients = coefficients

            
    def _compute_normalization_stats(self):
        """Compute normalization statistics for all recommenders."""
        if self.verbose:
            print(f"Computing normalization statistics using {self.normalization_method} method...")
        
        for i, recommender in enumerate(self.recommenders):
            # Compute scores for all users to get statistics
            all_scores = []
            
            # Sample users to compute statistics (for efficiency)
            n_users_sample = min(1000, self.URM_train.shape[0])
            user_sample = np.random.choice(self.URM_train.shape[0], n_users_sample, replace=False)
            
            for user_id in user_sample:
                scores = recommender._compute_item_score([user_id])
                all_scores.extend(scores.flatten())
            
            all_scores = np.array(all_scores)
            
            # Remove zeros to avoid division issues
            non_zero_scores = all_scores[all_scores != 0]
            
            if len(non_zero_scores) == 0:
                # Fallback to no normalization if all scores are zero
                self.normalization_stats[i] = {"method": None, "min": 0, "max": 1, "mean": 0, "std": 1}
                continue
                
            if self.normalization_method == "minmax":
                min_val = np.min(non_zero_scores)
                max_val = np.max(non_zero_scores)
                self.normalization_stats[i] = {"method": "minmax", "min": min_val, "max": max_val}
                
            elif self.normalization_method == "zscore":
                mean_val = np.mean(non_zero_scores)
                std_val = np.std(non_zero_scores)
                self.normalization_stats[i] = {"method": "zscore", "mean": mean_val, "std": std_val}
                
            elif self.normalization_method == "max":
                max_val = np.max(non_zero_scores)
                self.normalization_stats[i] = {"method": "max", "max": max_val}
                
            elif self.normalization_method == "sum":
                # Compute average sum per user
                sum_vals = []
                for user_id in user_sample:
                    scores = recommender._compute_item_score([user_id])
                    sum_vals.append(np.sum(scores))
                avg_sum = np.mean(sum_vals)
                self.normalization_stats[i] = {"method": "sum", "avg_sum": avg_sum}
                
            elif self.normalization_method == "l2":
                # Compute average L2 norm per user
                norm_vals = []
                for user_id in user_sample:
                    scores = recommender._compute_item_score([user_id])
                    norm_vals.append(np.linalg.norm(scores))
                avg_norm = np.mean(norm_vals)
                self.normalization_stats[i] = {"method": "l2", "avg_norm": avg_norm}
                
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def _normalize_scores(self, scores, recommender_idx):
        """Normalize scores based on pre-computed statistics."""
        if self.normalization_method is None or recommender_idx not in self.normalization_stats:
            return scores
            
        stats = self.normalization_stats[recommender_idx]
        
        if stats["method"] is None:
            return scores
            
        if stats["method"] == "minmax":
            min_val = stats["min"]
            max_val = stats["max"]
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            else:
                return scores
                
        elif stats["method"] == "zscore":
            mean_val = stats["mean"]
            std_val = stats["std"]
            if std_val > 0:
                return (scores - mean_val) / std_val
            else:
                return scores - mean_val
                
        elif stats["method"] == "max":
            max_val = stats["max"]
            if max_val > 0:
                return scores / max_val
            else:
                return scores
                
        elif stats["method"] == "sum":
            avg_sum = stats["avg_sum"]
            if avg_sum > 0:
                return scores / avg_sum
            else:
                return scores
                
        elif stats["method"] == "l2":
            avg_norm = stats["avg_norm"]
            if avg_norm > 0:
                return scores / avg_norm
            else:
                return scores
                
        return scores
    
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """Compute weighted combination of item scores from all recommenders."""
        # Initialize with zeros
        item_weights = np.zeros((len(user_id_array), self.n_items))
        
        for i, recommender in enumerate(self.recommenders):
            # Get raw scores from recommender
            recommender_scores = recommender._compute_item_score(user_id_array, items_to_compute)
            
            # Apply normalization if enabled
            if self.normalization_method is not None:
                recommender_scores = self._normalize_scores(recommender_scores, i)
            
            # Apply coefficient and add to combined scores
            weighted_scores = recommender_scores * self.coefficients[i]
            item_weights += weighted_scores
            
            if self.verbose and np.isnan(weighted_scores).any():
                print(f"Warning: NaN values in recommender {i} after weighting")
        
        # Apply final softmax if desired (optional)
        # item_weights = softmax(item_weights, axis=1)
        
        return item_weights
