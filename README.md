Support Vector Machines (SVM) for Classification

Objective:

The primary objective of this project was to implement and evaluate Support Vector Machines (SVM) for both linear and non-linear binary classification. This involved comprehensive data preparation, model training, **visualization of the decision boundary** (using PCA), and rigorous **hyperparameter optimization** using Grid Search and cross-validation.

------------------------------------------------------------------------------------------------------------------------

Tools and Libraries:

* **Python 3**
* **Scikit-learn (sklearn):** Core library for SVM models (`SVC`), preprocessing, model selection (`GridSearchCV`, `train_test_split`), and evaluation.
* **NumPy:** Essential for numerical operations.
* **Pandas:** Used for data loading and manipulation.
* **Matplotlib:** Used for visualizing the decision boundary.
* **PCA:** Employed for dimensionality reduction to enable 2D plotting.

------------------------------------------------------------------------------------------------------------------------

Key Steps and Results:

### 1. Baseline Model Performance
The initial models (Linear and default RBF) were trained on the full 30 features. This section details the performance metrics of one of the baseline models.

 **[Click here to view the Baseline Classification Report](Screenshots/Classification_Report.png)**

### 2. Decision Boundary Visualization (Using PCA)
Since the dataset has 30 features, **Principal Component Analysis (PCA)** was used to reduce the data to 2 dimensions. This allowed us to train a 2D RBF SVM model and visualize its non-linear separation boundary.

 **[Click here to view the 2D RBF Decision Boundary Plot](Screenshots/RBF_Plot.png)**

 -----------------------------------------------------------------------------------------------------------------------
