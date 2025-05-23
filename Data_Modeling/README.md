# Wine Quality Data Modeling Project

This project applies machine learning techniques to analyze and predict wine quality based on physicochemical properties. The analysis includes regression, classification, and clustering approaches to understand the relationships between wine attributes and quality ratings.

## Project Overview

The project focuses on data modeling as a core step in the data science process, implementing various machine learning algorithms to solve different analytical problems using a wine quality dataset.

## Dataset

The dataset (`data.csv`) contains **4,781 wine samples** with **12 attributes** including:
- 11 physicochemical input variables (fixed acidity, volatile acidity, citric acid, etc.)
- 1 output variable (quality score from 0-10)

For detailed attribute descriptions, see [`Readme-data.txt`](Readme-data.txt).

## File Structure

```
Data_Modeling/
├── README.md                              # This file
├── Readme-data.txt                        # Dataset attribute descriptions
├── data.csv                              # Wine quality dataset
├── wine_quality.ipynb                    # Main analysis notebook
└── wine_quality_data_modeling_report.pdf # Comprehensive project report
```

## Tasks & Methodology

### Task 1: Regression Analysis
**Objective**: Analyze the relationship between alcohol content and wine density

- **Sample Size**: 200 instances (random subset)
- **Method**: Simple Linear Regression
- **Variables**: 
  - Dependent: Alcohol (% by volume)
  - Independent: Density (g/cm³)
- **Deliverables**:
  - Scatter plot visualization
  - Linear regression model with coefficient interpretation
  - Model evaluation metrics (MSE, R²)

### Task 2: Classification
**Objective**: Classify wine quality using multiple algorithms

- **Sample Size**: 500 instances (random subset)
- **Target Variable**: Wine quality (0-10 scale)
- **Algorithms Implemented**:

#### k-Nearest Neighbors (kNN)
- Optimal k selection using cross-validation
- Standard kNN implementation
- **Modified kNN**: Enhanced with distance weighting and feature scaling
- Performance comparison between standard and modified versions

#### Decision Tree Classifier
- Hyperparameter tuning using GridSearchCV
- Optimal parameters: `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=20`
- Tree visualization for interpretability
- Comparative analysis with kNN performance

### Task 3: Clustering Analysis
**Objective**: Discover natural groupings in wine data

- **Sample Size**: 600 instances (random subset)
- **Variables**: All physicochemical attributes (excluding quality)
- **Algorithms**:

#### K-Means Clustering
- Optimal k selection using elbow method
- k=6 clusters (matching quality score range)
- Performance evaluation using multiple metrics

#### DBSCAN Clustering
- Parameter selection using k-distance graph
- `eps=8`, `min_samples=5`
- Noise point identification and handling
- Comparative analysis with K-Means

## Key Features

- **Reproducible Results**: All random operations use fixed seeds
- **Comprehensive Evaluation**: Multiple metrics for each algorithm
- **Visual Analysis**: Scatter plots, confusion matrices, comparison charts
- **Data Quality**: Automatic handling of missing values
- **Model Comparison**: Side-by-side performance analysis

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `warnings` - Warning management

## Evaluation Metrics

### Classification Tasks
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)
- Confusion Matrix

### Clustering Tasks
- Rand Index
- Homogeneity Score
- Completeness Score
- V-Measure Score

### Regression Tasks
- Mean Squared Error (MSE)
- R² Score (Coefficient of Determination)

## Key Findings

1. **Regression**: Identified relationship patterns between alcohol content and wine density
2. **Classification**: Modified kNN with distance weighting and feature scaling showed improved performance
3. **Clustering**: Both K-Means and DBSCAN revealed distinct wine groupings, with varying effectiveness

## Usage

1. **Setup Environment**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run Analysis**:
   - Open `wine_quality.ipynb` in Jupyter Notebook/Lab
   - Execute cells sequentially
   - Generated CSV files will be saved with student ID prefix

3. **View Results**:
   - Interactive plots and metrics in notebook
   - Comprehensive analysis in `wine_quality_data_modeling_report.pdf`

## Generated Files

The notebook automatically creates sample datasets:
- `s3720784_A2SampleOne.csv` (200 instances for regression)
- `s3720784_A2SampleTwo.csv` (500 instances for classification)
- `s3720784_A2SampleThree.csv` (600 instances for clustering)

## Author

**Campbell Timms** (s3720784)  
RMIT University - Data Science Program  
Semester 2, 2024

## Academic Context

This project was completed as Assignment 2 for a Data Science course, demonstrating proficiency in:
- Data preprocessing and sampling
- Machine learning algorithm implementation
- Model evaluation and comparison
- Scientific reporting and visualization
- Code documentation and reproducibility

---

*For technical details and complete analysis, refer to the Jupyter notebook and accompanying report.*
