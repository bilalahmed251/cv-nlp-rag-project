$credRequest = "protocol=https
host=github.com
"
$token = ($credRequest | git credential fill) | Select-String "password=" | % { $_.ToString().Substring(9) }

$headers = @{
    "Authorization" = "Bearer $token"
    "Accept" = "application/vnd.github.v3+json"
}

$repos = @(
    @{ name="-Diabetes-Prediction-using-Machine-Learning"; desc="An advanced Machine Learning pipeline to predict the onset of diabetes using Random Forest and XGBoost with SMOTE."; topics=@("python","machine-learning","xgboost","scikit-learn","healthcare-ai") },
    @{ name="Customer-Churn-Prediction"; desc="A production-ready ML pipeline to predict customer churn using Scikit-Learn Pipelines and XGBoost."; topics=@("python","machine-learning","xgboost","churn-prediction","classification") },
    @{ name="Customer-Segmentation-using-K-Means-Clustering"; desc="Unsupervised machine learning project for customer segmentation using K-Means Clustering and PCA."; topics=@("python","machine-learning","unsupervised-learning","k-means","pca") },
    @{ name="titanic-ml-project"; desc="Advanced feature engineering and Gradient Boosting classification for the classic Titanic survival dataset."; topics=@("python","machine-learning","classification","gradient-boosting","feature-engineering") },
    @{ name="Case-Study-of-Netflix-Titles-Exploratory-Data-Analysis"; desc="A deep-dive analytical case study and Exploratory Data Analysis (EDA) of Netflix's content library."; topics=@("python","data-analysis","eda","seaborn","data-visualization") },
    @{ name="Mobile-Phone-Price-Prediction-EDA"; desc="Multi-class classification using Support Vector Machines (SVM) to predict mobile phone price brackets."; topics=@("python","machine-learning","svm","multi-class-classification","scikit-learn") },
    @{ name="Pakistan-s-Largest-E-Commerce-Dataset-Analysis"; desc="Time-series analysis and sales forecasting of Pakistan's largest e-commerce dataset."; topics=@("python","data-analysis","time-series","ecommerce","data-science") },
    @{ name="customer_lifetime_value"; desc="Predictive analytics using RFM modeling and Gradient Boosting Regressor to estimate customer lifetime value."; topics=@("python","machine-learning","regression","predictive-analytics","ltv") }
)

foreach ($r in $repos) {
    Write-Host "Updating $($r.name)..."
    
    $descBody = @{ description = $r.desc } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "https://api.github.com/repos/bilalahmed251/$($r.name)" -Method Patch -Headers $headers -Body $descBody -ContentType "application/json" | Out-Null
        Write-Host "  Description updated."
    } catch { Write-Host "  Error updating description: $_" }

    $topicsBody = @{ names = $r.topics } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "https://api.github.com/repos/bilalahmed251/$($r.name)/topics" -Method Put -Headers $headers -Body $topicsBody -ContentType "application/json" | Out-Null
        Write-Host "  Topics updated."
    } catch { Write-Host "  Error updating topics: $_" }
}

Write-Host "All repos successfully updated!"
