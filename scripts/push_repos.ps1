$repos = @(
    @{ repo='https://github.com/bilalahmed251/-Diabetes-Prediction-using-Machine-Learning.git'; file='1_diabetes_prediction.py'; name='-Diabetes-Prediction-using-Machine-Learning' },
    @{ repo='https://github.com/bilalahmed251/Customer-Churn-Prediction.git'; file='2_customer_churn.py'; name='Customer-Churn-Prediction' },
    @{ repo='https://github.com/bilalahmed251/Customer-Segmentation-using-K-Means-Clustering.git'; file='3_customer_segmentation.py'; name='Customer-Segmentation-using-K-Means-Clustering' },
    @{ repo='https://github.com/bilalahmed251/titanic-ml-project.git'; file='4_titanic_ml.py'; name='titanic-ml-project' },
    @{ repo='https://github.com/bilalahmed251/Case-Study-of-Netflix-Titles-Exploratory-Data-Analysis.git'; file='5_netflix_eda.py'; name='Case-Study-of-Netflix-Titles-Exploratory-Data-Analysis' },
    @{ repo='https://github.com/bilalahmed251/Mobile-Phone-Price-Prediction-EDA.git'; file='6_mobile_price_prediction.py'; name='Mobile-Phone-Price-Prediction-EDA' },
    @{ repo='https://github.com/bilalahmed251/Pakistan-s-Largest-E-Commerce-Dataset-Analysis.git'; file='7_ecommerce_analysis.py'; name='Pakistan-s-Largest-E-Commerce-Dataset-Analysis' },
    @{ repo='https://github.com/bilalahmed251/customer_lifetime_value.git'; file='8_customer_lifetime_value.py'; name='customer_lifetime_value' }
)

$baseDir = 'C:\Users\ba650\Downloads\CV NLP RAG Project'
$missingCodesDir = 'C:\Users\ba650\Downloads\CV NLP RAG Project\Missing_Codes'
$tempDir = 'C:\Users\ba650\Downloads\CV NLP RAG Project\GitHub_Temp'
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Set-Location $tempDir

foreach ($r in $repos) {
    Write-Host "Processing $($r.name)..."
    git clone $r.repo
    if (Test-Path $r.name) {
        Copy-Item -Path "$missingCodesDir\$($r.file)" -Destination "$($r.name)\$($r.file)" -Force
        Set-Location $r.name
        git add $($r.file)
        git commit -m "Add missing Python script for project pipeline"
        git push
        Set-Location $tempDir
    }
}
