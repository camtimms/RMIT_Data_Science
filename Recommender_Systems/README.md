# Movie Recommender Systems: Collaborative Filtering and Matrix Factorization

A comprehensive implementation and evaluation of two popular recommendation algorithms using the MovieLens 1M dataset.

## 📁 Project Structure

```
RMIT_Data_Science/Recommender_Systems/
├── ml-1m/                              # MovieLens 1M dataset
│   ├── README.md
│   ├── README.txt
│   ├── movies.dat                      # Movie information (ID, Title, Genres)
│   ├── ratings.dat                     # User ratings (UserID, MovieID, Rating, Timestamp)
│   └── users.dat                       # User demographics (UserID, Gender, Age, Occupation, Zip)
├── README.md                           # This file
├── calculate_user_ndcg.ipynb          # NDCG calculation utilities
├── movie_recomender_system.ipynb      # Main analysis notebook
├── movie_recomender_system_slides.pdf # Project presentation
└── progress_bar.ipynb                 # Progress tracking utilities
```

## 🎯 Project Overview

This project implements and compares three different recommender system approaches:

1. **k-Nearest Neighbors (kNN) Collaborative Filtering** - User-based recommendation using similarity metrics
2. **Matrix Factorization (SVD/SVD++)** - Latent factor models for rating prediction
3. **Improved Matrix Factorization with Features (IMFR)** - Enhanced SVD++ incorporating user demographics

## 📊 Dataset Information

**MovieLens 1M Dataset:**
- **Users:** 6,040 users with demographic information
- **Movies:** 3,706 movies with titles and genres
- **Ratings:** 1,000,209 ratings on a 1-5 scale
- **Sparsity:** Approximately 95.8% sparse user-item matrix

## 🔧 Implementation Details

### Task 1: k-Nearest Neighbors Collaborative Filtering

**Objective:** Implement user-based collaborative filtering to predict ratings for a randomly selected test user.

**Key Features:**
- Cosine similarity and Euclidean distance comparison
- Optimal k parameter selection (k=5, 10, 15, 20, 25)
- RMSE evaluation on user subsets
- Visualization of prediction errors and rating distributions

**Best Performance:** Euclidean distance with k=25

### Task 2: Matrix Factorization-based Recommendation

**Objective:** Implement and evaluate SVD-based matrix factorization techniques.

**Implementation:**
- **Base Model:** SVD using scikit-surprise library
- **Enhanced Model:** SVD++ with user demographic features (gender, age, occupation, zip code)
- Random selection of 5 movies for prediction
- RMSE evaluation and performance comparison

**Key Improvements:**
- Incorporation of implicit feedback from user demographics
- Feature engineering for categorical variables
- Enhanced prediction accuracy through user profiling

### Task 3: Ranking-based Evaluation and Comparison

**Objective:** Compare kNN and Matrix Factorization models using ranking metrics.

**Evaluation Setup:**
- 10 randomly selected users (with >100 ratings each)
- Top-20 movie recommendations per user
- Comprehensive evaluation using multiple metrics

**Evaluation Metrics:**
- **RMSE:** Root Mean Square Error for rating prediction accuracy
- **Average Precision (AP):** Binary relevance (rating ≥ 4 = relevant)
- **NDCG:** Normalized Discounted Cumulative Gain with graded relevance

## 📈 Results Summary

### Model Performance Comparison

| Model | RMSE | Average AP | Average NDCG |
|-------|------|------------|--------------|
| kNN (Euclidean, k=25) | [1.326] | [Value] | [Value] |
| SVD++ with Features | [Value] | [Value] | [Value] |

**Key Findings:**
- Euclidean distance outperformed cosine similarity for kNN
- Optimal k value was 25 for the user subset tested
- SVD++ with demographic features showed improved performance over basic SVD
- Matrix factorization models generally provided better ranking performance

## 🛠 Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scikit-surprise>=1.1.1
tqdm>=4.62.0
```

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd RMIT_Data_Science/Recommender_Systems/
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise tqdm
```

3. Download MovieLens 1M dataset (if not included):
```bash
# Dataset should be placed in ml-1m/ directory
```

### Running the Analysis

1. **Main Analysis:**
   ```bash
   jupyter notebook movie_recomender_system.ipynb
   ```

2. **NDCG Calculations:**
   ```bash
   jupyter notebook calculate_user_ndcg.ipynb
   ```

## 📋 Key Functions

### kNN Collaborative Filtering
```python
kNN_recommender(k, user_id, user_item_matrix, similarity_matrix)
evaluate_similarity_metrics(k, user_subset, user_item_matrix, ratings, cosine_sim_matrix, euclidean_sim_matrix)
```

### Matrix Factorization
```python
run_svdpp_with_user_features(ratings, users, movie_subset)
run_svdpp_with_all_movies(ratings, users, user_ids)
```

### Evaluation Metrics
```python
rmse(y_true, y_pred)
calculate_user_ndcg(recommendations, actual, k=20)
```

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Sparsity Analysis:** Item popularity distribution
- **Error Analysis:** Prediction error distributions
- **Performance Comparison:** Side-by-side model comparisons
- **Rating Distributions:** Actual vs. predicted rating patterns
- **Ranking Metrics:** AP and NDCG score comparisons

## 🔍 Technical Highlights

### Data Preprocessing
- User-item matrix creation with sparsity handling
- Similarity matrix computation (cosine and Euclidean)
- Feature engineering for demographic data

### Model Optimization
- Grid search for optimal k values
- Hyperparameter tuning for SVD models
- Cross-validation strategies for robust evaluation

### Evaluation Framework
- Multiple evaluation perspectives (prediction accuracy and ranking quality)
- Statistical significance testing
- Comprehensive error analysis

## 👨‍💻 Author

**Campbell Timms (s3720784)**  
RMIT University - Data Science

## 📄 License

This project is part of academic coursework and is intended for educational purposes.

---

*For detailed implementation and results, please refer to the Jupyter notebooks and presentation slides included in this repository.*
