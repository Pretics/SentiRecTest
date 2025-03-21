# PowerShell 스크립트 시작
param (
    [string]$size = "demo"  # 기본값을 "demo"로 설정
)

Write-Output "Starting data preprocessing..."

# 데이터셋별 디렉토리 설정
$datasetDir = "MIND/$size"
$datasetTrainDir = "$datasetDir/train"
$datasetTestDir = "$datasetDir/test"
$wordEmbeddingDir = "word_embeddings"

$processedDataDir = "processed"
$preTrainDir = "$processedDataDir/$size/train"
$preTestDir = "$processedDataDir/$size/test"

# 디렉토리 생성 (없으면)
$dirs = @($preTrainDir, $preTestDir)
foreach ($dir in $dirs) {
    if (!(Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

# Conda 환경 활성화
$env:PATH = "C:\Users\USER\anaconda3\Scripts;C:\Users\USER\anaconda3\bin;" + $env:PATH
conda activate newsrec

# 데이터 전처리 실행
Write-Output "Preprocessing training/test-set impression logs..."
python parse_behavior_combined.py --train-file "$datasetTrainDir/behaviors.tsv" --test-file "$datasetTestDir/behaviors.tsv" --train-out "$preTrainDir" --test-out "$preTestDir" --user2int "$preTrainDir/user2int.tsv"

Write-Output "Preprocessing training-set news content..."
python parse_news_combined.py --train-file "$datasetTrainDir/news.tsv" --test-file "$datasetTestDir/news.tsv" --train-out "$preTrainDir" --test-out "$preTestDir" --word-embeddings "$wordEmbeddingDir/glove.840B.300d.txt"

Write-Output "Data preprocessing complete!"