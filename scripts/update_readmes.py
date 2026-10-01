import os
import textwrap
import subprocess

base_dir = r'C:\Users\ba650\Downloads\CV NLP RAG Project\GitHub_Temp'

readmes = {
    r'-Diabetes-Prediction-using-Machine-Learning': '''\
        # Diabetes Prediction using Machine Learning

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Enabled-orange)
        ![XGBoost](https://img.shields.io/badge/XGBoost-Powered-green)

        An advanced, end-to-end Machine Learning pipeline to predict the onset of diabetes based on diagnostic measures. 

        ## Features
        - **Comprehensive EDA:** Automatically generates feature correlation heatmaps to understand data distributions.
        - **Advanced Preprocessing:** Utilizes StandardScaler for normalization and SMOTE to handle class imbalances effectively.
        - **Hyperparameter Tuning:** Implements GridSearchCV to find the absolute best parameters for the Random Forest model.
        - **Model Comparison:** Evaluates both Random Forest and XGBoost classifiers using accuracy and ROC-AUC metrics.
        - **Production-Ready Export:** Saves the best performing model using joblib for seamless deployment.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn, Seaborn, Matplotlib

        ---
        *Built with by Bilal Ahmed.*
        ''',

    r'Customer-Churn-Prediction': '''\
        # Customer Churn Prediction

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-success)
        
        A highly robust, production-ready Machine Learning pipeline designed to predict customer churn in telecommunications and subscription-based services.

        ## Core Functionality
        - **Scikit-Learn Pipelines:** Utilizes ColumnTransformer and Pipeline for elegant, leak-proof data preprocessing.
        - **Categorical Handling:** Implements OneHotEncoder for flawless integration of categorical variables (Contract type, Internet Service).
        - **High-Performance Modeling:** Trained using the state-of-the-art XGBoost Classifier.
        - **Evaluation & Export:** Outputs detailed classification reports and exports the entire preprocessing+modeling pipeline as a .pkl file.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
        ''',

    r'Customer-Segmentation-using-K-Means-Clustering': '''\
        # Customer Segmentation using K-Means Clustering

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Unsupervised Learning](https://img.shields.io/badge/ML-Unsupervised-purple)

        An advanced unsupervised machine learning project that analyzes customer purchasing behavior and groups them into distinct segments for targeted marketing.

        ## Features
        - **Optimal Cluster Detection:** Programmatically determines the best number of clusters (k) using the Silhouette Score optimization loop.
        - **Dimensionality Reduction:** Employs Principal Component Analysis (PCA) to reduce complex multi-dimensional data down to 2 components.
        - **Data Visualization:** Automatically generates and saves a beautiful 2D scatter plot projection (customer_segments_pca.png) of the customer clusters using Seaborn.
        - **Model Export:** Exports the final KMeans model for future inferences.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Scikit-Learn (KMeans, PCA), Seaborn, Matplotlib, Pandas
        ''',

    r'titanic-ml-project': '''\
        # Titanic Survival Prediction (Advanced ML)

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Classification](https://img.shields.io/badge/Task-Classification-red)

        A sophisticated take on the classic Titanic dataset. This project goes beyond basic models by implementing advanced feature engineering and Gradient Boosting.

        ## Features
        - **Advanced Feature Engineering:** Extracts new, highly predictive features such as FamilySize and IsAlone from existing data.
        - **Gradient Boosting:** Utilizes GradientBoostingClassifier for superior accuracy over traditional decision trees.
        - **Robust Validation:** Uses k-fold cross-validation (cross_val_score) to ensure the model generalizes perfectly to unseen data.
        - **Export Ready:** Saves the trained model pipeline for immediate deployment.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
        ''',

    r'Case-Study-of-Netflix-Titles-Exploratory-Data-Analysis': '''\
        # Netflix Titles Exploratory Data Analysis (EDA)

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Data Analysis](https://img.shields.io/badge/Task-EDA-orange)

        A deep-dive analytical case study into Netflix's library of Movies and TV Shows. 

        ## Highlights
        - **Content Distribution Analysis:** Analyzes the ratio of Movies to TV Shows on the platform.
        - **Time-Series Trends:** Plots the release years using Seaborn KDE plots to observe historical content acquisition trends.
        - **Automated Visualizations:** Automatically generates and saves high-quality .png charts for reporting and presentations.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Pandas, Seaborn, Matplotlib, NumPy
        ''',

    r'Mobile-Phone-Price-Prediction-EDA': '''\
        # Mobile Phone Price Prediction

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Multi-Class Classification](https://img.shields.io/badge/ML-Multi--Class-brightgreen)

        An advanced multi-class classification project that predicts the price range of mobile phones based on their hardware specifications.

        ## Features
        - **Support Vector Machines (SVM):** Implements a linear SVC model to handle the complex, multi-dimensional feature space of mobile specs.
        - **Multi-Class Evaluation:** Outputs detailed metrics (Precision, Recall, F1-Score) for all 4 price brackets.
        - **Visual Analytics:** Generates a beautifully formatted Confusion Matrix heatmap using Seaborn to visualize true vs. predicted classifications.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Scikit-Learn, Seaborn, Matplotlib, Pandas
        ''',

    r'Pakistan-s-Largest-E-Commerce-Dataset-Analysis': '''\
        # Pakistan's Largest E-Commerce Dataset Analysis

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Data Science](https://img.shields.io/badge/Domain-Data%20Science-yellow)

        A comprehensive time-series analysis focusing on transaction behaviors, sales trends, and operational statuses of Pakistan's e-commerce landscape.

        ## Analysis Features
        - **Time-Series Processing:** Extracts and indexes time-based features (Year, Month, Day) using Pandas DateTime accessors.
        - **Sales Trend Forecasting:** Groups and analyzes completed transactions to generate a macroscopic view of monthly sales trends.
        - **Data Visualization:** Outputs a structured line-plot (monthly_sales_trend.png) highlighting revenue peaks and valleys over the dataset's lifespan.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Pandas, Seaborn, Matplotlib, NumPy
        ''',

    r'customer_lifetime_value': '''\
        # Customer Lifetime Value (LTV) Prediction

        ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
        ![Regression](https://img.shields.io/badge/ML-Regression-blueviolet)

        A high-level predictive analytics project designed to estimate the total monetary value a customer will bring over their entire relationship with a business.

        ## Pipeline Features
        - **RFM Modeling:** Structures the input data using Recency, Frequency, and Monetary parameters - the gold standard for LTV predictions.
        - **Gradient Boosting Regressor:** Utilizes advanced ensemble learning to capture non-linear relationships in customer purchasing habits.
        - **Extensive Evaluation:** Scores the model using R-Squared and Root Mean Squared Error (RMSE).
        - **Feature Importance:** Extracts and ranks the impact of each variable (e.g., Frequency vs Recency) on the final LTV prediction.

        ## Tech Stack
        - **Language:** Python
        - **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
        '''
}

print("Writing READMEs...")
for repo_folder, content in readmes.items():
    repo_path = os.path.join(base_dir, repo_folder)
    if os.path.exists(repo_path):
        readme_path = os.path.join(repo_path, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(textwrap.dedent(content))
        
        subprocess.run(['git', 'add', 'README.md'], cwd=repo_path)
        subprocess.run(['git', 'commit', '-m', 'Update README with professional project description and badges'], cwd=repo_path)
        subprocess.run(['git', 'push'], cwd=repo_path)
        print(f"Pushed README for {repo_folder}")

print("All READMEs updated and pushed successfully!")
