# RMIT Data Science Projects

A portfolio of machine learning and data science projects completed as part of the RMIT University Data Science program, demonstrating various algorithms and analytical approaches across different domains.

## Repository Structure

```
RMIT_Data_Science/
├── README.md                     # This file
├── Data_Modeling/               # Wine quality analysis project
│   ├── README.md
│   ├── data.csv
│   ├── wine_quality.ipynb
│   └── wine_quality_data_modeling_report.pdf
└── Recommender_Systems/         # Movie recommendation algorithms
    ├── README.md
    ├── ml-1m/                   # MovieLens 1M dataset
    ├── movie_recomender_system.ipynb
    └── movie_recomender_system_slides.pdf
```

## Projects

### 🍷 [Data Modeling](./Data_Modeling/)
**Wine Quality Analysis & Prediction**

Machine learning analysis of wine quality using physicochemical properties. Implements regression, classification (kNN, Decision Trees), and clustering (K-Means, DBSCAN) on 4,781 wine samples.

**Key Features**: Modified kNN algorithms, hyperparameter tuning, comparative model evaluation

### 🎬 [Recommender Systems](./Recommender_Systems/)
**Movie Recommendation Algorithms**

Implementation and comparison of collaborative filtering approaches using the MovieLens 1M dataset. Features kNN collaborative filtering, SVD/SVD++ matrix factorization, and demographic-enhanced recommendations.

**Key Features**: User-based filtering, matrix factorization, ranking evaluation (NDCG, Average Precision)

## Technologies

- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn)
- **Jupyter Notebooks** for interactive analysis
- **Machine Learning**: Supervised/unsupervised learning, recommendation systems
- **Evaluation**: Cross-validation, multiple performance metrics

## Getting Started

Each project contains detailed setup instructions in its respective README. Common requirements:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Navigate to individual project directories for specific dependencies and usage instructions.

## Author

**Campbell Timms** (s3720784)  
RMIT University - Data Science Program  
Semester 2, 2024

---

## Quick Navigation

- 📊 **[Wine Quality Analysis](./Data_Modeling/)** - ML modeling with regression, classification & clustering
- 🔍 **[Movie Recommender Systems](./Recommender_Systems/)** - Collaborative filtering & matrix factorization
