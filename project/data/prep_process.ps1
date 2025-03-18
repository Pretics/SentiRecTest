# PowerShell 스크립트 시작
param (
    [string]$size = "demo"  # 기본값을 "demo"로 설정
)

Write-Output "Starting data preprocessing..."

# 데이터셋별 디렉토리 설정
$datasetPath = "MIND/$size"
$datasetTrainPath = "$datasetPath/train"
$datasetTestPath = "$datasetPath/test"
$wordEmbeddingPath = "word_embeddings"

$processedDataPath = "processed"
$preTrainPath = "$processedDataPath/$size/train"
$preTestPath = "$processedDataPath/$size/test"

# 디렉토리 생성 (없으면)
$dirs = @($preTrainPath, $preTestPath)
foreach ($dir in $dirs) {
    if (!(Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

# Conda 환경 활성화
$env:PATH = "C:\Users\USER\anaconda3\Scripts;C:\Users\USER\anaconda3\bin;" + $env:PATH
conda activate newsrec

# 데이터 전처리 실행
Write-Output "Preprocessing training-set impression logs..."
python parse_behavior.py --in-file "$datasetTrainPath/behaviors.tsv" --out-dir "$preTrainPath" --mode train

Write-Output "Preprocessing test-set impression logs..."
python parse_behavior.py --in-file "$datasetTestPath/behaviors.tsv" --out-dir "$preTestPath" --mode test --user2int "$preTrainPath/user2int.tsv"

Write-Output "Preprocessing training-set news content..."
python parse_news.py --in-file "$datasetTrainPath/news.tsv" --out-dir "$preTrainPath" --mode train --word-embeddings "$wordEmbeddingPath/glove.840B.300d.txt"

Write-Output "Preprocessing test-set news content..."
python parse_news.py --in-file "$datasetTestPath/news.tsv" --out-dir "$preTestPath" --mode test --word-embeddings "$wordEmbeddingPath/glove.840B.300d.txt" --embedding-weights "$preTrainPath/embedding_weights.csv" --word2int "$preTrainPath/word2int.tsv" --category2int "$preTrainPath/category2int.tsv"

Write-Output "Data preprocessing complete!"