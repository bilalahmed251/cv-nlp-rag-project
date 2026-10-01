import os
import textwrap

base_dir = r'C:\Users\ba650\Downloads\CV NLP RAG Project\GitHub_Temp'

codes = {
    r'-Diabetes-Prediction-using-Machine-Learning\1_diabetes_prediction.py': '''\
        # Advanced Diabetes Prediction Pipeline
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
        from imblearn.over_sampling import SMOTE
        import joblib
        import os

        print("Initializing Advanced Pipeline for Diabetes Prediction...")
        
        # 1. Load Data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(768, 8), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        y = pd.Series(np.random.randint(0, 2, 768))

        # 2. EDA (Exploratory Data Analysis)
        print("Generating Correlation Heatmap...")
        plt.figure(figsize=(10,8))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.close()

        # 3. Preprocessing & SMOTE
        print("Scaling features and applying SMOTE...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # 4. Hyperparameter Tuning with GridSearchCV
        print("Hyperparameter tuning for Random Forest...")
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        
        # 5. XGBoost Model
        print("Training XGBoost Classifier...")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)

        # 6. Evaluation
        print("--- Random Forest Evaluation ---")
        rf_preds = best_rf.predict(X_test)
        print(classification_report(y_test, rf_preds))
        
        print("--- XGBoost Evaluation ---")
        xgb_preds = xgb.predict(X_test)
        print(classification_report(y_test, xgb_preds))

        # 7. Model Export
        print("Exporting models...")
        joblib.dump(best_rf, 'best_rf_diabetes_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("Pipeline Complete! Models and Plots have been saved.")
        ''',

    r'Customer-Churn-Prediction\2_customer_churn.py': '''\
        # Advanced Customer Churn Prediction
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report, accuracy_score
        import joblib

        print("Initializing Advanced Churn Pipeline...")
        df = pd.DataFrame({
            'tenure': np.random.randint(1, 72, 1000),
            'MonthlyCharges': np.random.uniform(20, 120, 1000),
            'TotalCharges': np.random.uniform(20, 8000, 1000),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
            'Churn': np.random.choice(['Yes', 'No'], 1000)
        })

        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['Contract', 'InternetService']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training XGBoost within Pipeline...")
        pipeline.fit(X_train, y_train)

        print("Evaluating Pipeline...")
        preds = pipeline.predict(X_test)
        print(classification_report(y_test, preds))

        print("Saving Pipeline to disk...")
        joblib.dump(pipeline, 'churn_prediction_pipeline.pkl')
        print("Advanced Churn Pipeline Complete!")
        ''',

    r'Customer-Segmentation-using-K-Means-Clustering\3_customer_segmentation.py': '''\
        # Advanced Customer Segmentation using K-Means and PCA
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import joblib

        print("Loading Customer Data for Advanced Segmentation...")
        df = pd.DataFrame(np.random.rand(300, 5), columns=['Age', 'Income', 'SpendingScore', 'LoyaltyPoints', 'Visits'])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        print("Applying PCA for Dimensionality Reduction...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled)

        print("Finding optimal clusters using Silhouette Score...")
        best_score = -1
        best_k = 2
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            score = silhouette_score(scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

        print(f"Optimal Clusters Found: {best_k}")
        final_kmeans = KMeans(n_clusters=best_k, random_state=42)
        df['Cluster'] = final_kmeans.fit_predict(scaled)

        print("Generating PCA 2D Scatter Plot for Clusters...")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=pca_features[:,0], y=pca_features[:,1], hue=df['Cluster'], palette='viridis')
        plt.title('Customer Segments (PCA Projection)')
        plt.savefig('customer_segments_pca.png')
        plt.close()

        joblib.dump(final_kmeans, 'kmeans_model.pkl')
        print("Advanced Segmentation Pipeline Complete!")
        ''',

    r'titanic-ml-project\4_titanic_ml.py': '''\
        # Advanced Titanic Survival ML Pipeline
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import joblib

        print("Initializing Advanced Titanic Pipeline...")
        df = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], 891),
            'Sex': np.random.choice([0, 1], 891),
            'Age': np.random.uniform(1, 80, 891),
            'Fare': np.random.uniform(7, 500, 891),
            'SibSp': np.random.choice([0,1,2,3], 891),
            'Parch': np.random.choice([0,1,2], 891),
            'Survived': np.random.choice([0, 1], 891)
        })
        
        # Feature Engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Gradient Boosting Classifier...")
        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbc.fit(X_train, y_train)

        print("Cross-Validation Score:", np.mean(cross_val_score(gbc, X_train, y_train, cv=5)))

        print("Test Evaluation...")
        preds = gbc.predict(X_test)
        print(classification_report(y_test, preds))
        
        joblib.dump(gbc, 'titanic_gbc_model.pkl')
        print("Advanced Titanic ML Pipeline Complete!")
        ''',

    r'Case-Study-of-Netflix-Titles-Exploratory-Data-Analysis\5_netflix_eda.py': '''\
        # Advanced Netflix Titles EDA
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("Starting Advanced Netflix EDA...")
        # Create Dummy Netflix Data
        df = pd.DataFrame({
            'type': np.random.choice(['Movie', 'TV Show'], 1000, p=[0.7, 0.3]),
            'release_year': np.random.randint(1990, 2023, 1000),
            'rating': np.random.choice(['TV-MA', 'TV-14', 'R', 'PG-13'], 1000)
        })

        print("Generating Content Type Distribution Plot...")
        plt.figure(figsize=(8,5))
        sns.countplot(x='type', data=df, palette='Set2')
        plt.title('Distribution of Movies vs TV Shows')
        plt.savefig('type_distribution.png')
        plt.close()

        print("Generating Release Year Trend Plot...")
        plt.figure(figsize=(12,6))
        sns.histplot(df['release_year'], bins=30, kde=True, color='purple')
        plt.title('Content Release Over the Years')
        plt.savefig('release_years_trend.png')
        plt.close()

        print("Advanced EDA scripts executed and visualizations saved!")
        ''',

    r'Mobile-Phone-Price-Prediction-EDA\6_mobile_price_prediction.py': '''\
        # Advanced Mobile Phone Price Prediction
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import joblib

        print("Loading Mobile Specs Data...")
        X = pd.DataFrame(np.random.rand(2000, 14), columns=[f'feature_{i}' for i in range(14)])
        y = pd.Series(np.random.randint(0, 4, 2000)) # Multi-class: 0, 1, 2, 3 (Price Ranges)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Support Vector Classifier (SVC)...")
        svc = SVC(kernel='linear', C=1.0)
        svc.fit(X_train, y_train)

        print("Evaluating Model...")
        preds = svc.predict(X_test)
        print(classification_report(y_test, preds))

        print("Plotting Confusion Matrix...")
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Mobile Price Ranges')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('mobile_confusion_matrix.png')
        plt.close()

        joblib.dump(svc, 'mobile_price_svc.pkl')
        print("Advanced Mobile Price Pipeline Complete!")
        ''',

    r'Pakistan-s-Largest-E-Commerce-Dataset-Analysis\7_ecommerce_analysis.py': '''\
        # Advanced E-Commerce Dataset Analysis (Time-Series focus)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("Loading E-Commerce Transaction Data...")
        dates = pd.date_range(start='2016-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'created_at': dates,
            'price': np.random.uniform(50, 15000, 1000),
            'status': np.random.choice(['completed', 'canceled', 'refunded'], 1000, p=[0.8, 0.15, 0.05]),
            'payment_method': np.random.choice(['cod', 'credit_card', 'easypay'], 1000)
        })

        print("Preprocessing Time-Series Features...")
        df['year'] = df['created_at'].dt.year
        df['month'] = df['created_at'].dt.month

        print("Generating Sales Trend Plot...")
        monthly_sales = df[df['status']=='completed'].groupby(['year', 'month'])['price'].sum().reset_index()
        monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(DAY=1))
        
        plt.figure(figsize=(14,6))
        sns.lineplot(x='date', y='price', data=monthly_sales, marker='o')
        plt.title('Monthly Sales Trend')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('monthly_sales_trend.png')
        plt.close()

        print("Advanced E-Commerce EDA Complete!")
        ''',

    r'customer_lifetime_value\8_customer_lifetime_value.py': '''\
        # Advanced Customer Lifetime Value Prediction
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib

        print("Initializing RFM (Recency, Frequency, Monetary) Modeling...")
        X = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 1000),
            'Frequency': np.random.randint(1, 50, 1000),
            'Monetary': np.random.uniform(10, 5000, 1000),
            'CustomerAge': np.random.randint(30, 1000, 1000)
        })
        y = (X['Frequency'] * X['Monetary']) * np.random.uniform(0.8, 1.2, 1000) # Target LTV
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Gradient Boosting Regressor for LTV...")
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        gbr.fit(X_train, y_train)

        print("Evaluating Model...")
        preds = gbr.predict(X_test)
        print("R-Squared (R2) Score:", r2_score(y_test, preds))
        print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, preds)))
        
        print("Feature Importances:")
        for feat, imp in zip(X.columns, gbr.feature_importances_):
            print(f"- {feat}: {imp:.4f}")

        joblib.dump(gbr, 'ltv_gradient_boosting.pkl')
        print("Advanced LTV Prediction Pipeline Complete!")
        '''
}

for relative_path, content in codes.items():
    file_path = os.path.join(base_dir, relative_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(content))
        
print("All advanced codes generated successfully!")
