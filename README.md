# RecSys Challenge 2025: Hybrid Recommender System

This project combines collaborative filtering, content-based techniques, matrix factorization, and tree-based ensembling to deliver high-accuracy, personalized recommendations under strict evaluation metrics.

- **Data Preprocessing & Feature Engineering:**
  - Cleaned user-item interaction matrices and categorical metadata.
  - Generated temporal, demographic, and implicit interaction signals.
- **Hybrid Model Architecture:**
  - **Collaborative Filtering:** Item-based and User-based k-NN heuristics.
  - **Matrix Factorization:** Latent factor extraction via Matrix Factorization / SVD.
  - **Content-Based Filtering:** TF-IDF and item feature embeddings.
  - **Gradient Boosting:** Final ensembling layer combining candidate scores using XGBoost.
- **Evaluation & Benchmarking:**
  - Custom evaluation pipeline computing **MAP@10**, **NDCG@10**, and **Precision@K**.
  - Hyperparameter optimization across model pipelines.
