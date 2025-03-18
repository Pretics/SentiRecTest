# PowerShell 스크립트 시작
param (
    [string]$size = "demo"  # 기본값을 "demo"로 설정
)

# URL 매핑
$datasetURLs = @{
    "demo" = @{
        "train" = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
        "test"  = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
    }
    "small" = @{
        "train" = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip"
        "test"  = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip"
    }
    "large" = @{
        "train" = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip"
        "test"  = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip"
    }
    "word_embeddings" = @{
        "glove" = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    }
}

# 압축 파일 이름 매핑
$zipFileName = @{
    "large" = @{
        "train" = "MINDlarge_train.zip"
        "test"  = "MINDlarge_dev.zip"
    }
    "small" = @{
        "train" = "MINDsmall_train.zip"
        "test"  = "MINDsmall_dev.zip"
    }
    "demo" = @{
        "train" = "MINDdemo_train.zip"
        "test"  = "MINDdemo_dev.zip"
    }
    "word_embeddings" = @{
        "glove" = "glove.840B.300d.zip"
    }
}

# 데이터셋별 디렉토리 설정
$datasetPath = "MIND/$size"
$trainPath = "$datasetPath/train"
$testPath = "$datasetPath/test"
$wordEmbeddingPath = "word_embeddings"

# 데이터셋 선택 검증
if (-not $datasetURLs.ContainsKey($size)) {
    Write-Output "Invalid dataset size '$size'. Please use 'demo', 'small' or 'large'."
    exit 1
}

Write-Output "Starting data download for MIND-$size..."

# 디렉토리 생성 (없으면)
$dirs = @($datasetPath, $trainPath, $testPath, "word_embeddings")
foreach ($dir in $dirs) {
    if (!(Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

# 파일 다운로드 함수
function Get-File($url, $outputDir, $fileName) {
    $outputFile = "$outputDir/$fileName"
    if (!(Test-Path -Path $outputFile)) {
        if (!(Test-Path -Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force  # 폴더가 없으면 생성
        }
        Write-Output "Downloading $outputFile..."
        Invoke-WebRequest -Uri $url -OutFile $outputFile
    } else {
        Write-Output "$outputFile already exists. Skipping download."
    }
}

# 압축 해제 함수
function Expand-ArchiveFile($zipFileDir, $destination) {
    if (!(Test-Path -Path $destination)) {
        New-Item -ItemType Directory -Path $destination -Force  # 폴더가 없으면 생성
    }
    Write-Output "Extracting $zipFileDir..."
    Expand-Archive -Path $zipFileDir -DestinationPath $destination -Force
}

# 파일 다운로드 & 압축 해제 실행 여부를 결정하는 함수
function Invoke-Download($url, $zipFileDir, $FileName, $destination) {
    Get-File $url $zipFileDir $fileName
    Expand-ArchiveFile "$zipFileDir/$fileName" $destination
}

# 다운로드 관리
function Select-Download($type) {
    if ($type -eq "train") {
        $url = $datasetURLs[$size]["train"]
        $zipFileDir = $datasetPath
        $fileName = $zipFileName[$size]["train"]
        $destination = $trainPath
        Invoke-Download $url $zipFileDir $fileName $destination
    } elseif ($type -eq "test") {
        $url = $datasetURLs[$size]["test"]
        $zipFileDir = $datasetPath
        $fileName = $zipFileName[$size]["test"]
        $destination = $testPath
        Invoke-Download $url $zipFileDir $fileName $destination
    } elseif ($type -eq "word_embeddings") {
        $url = $datasetURLs["word_embeddings"]["glove"]
        $zipFileDir = $wordEmbeddingPath
        $fileName = $zipFileName["word_embeddings"]["glove"]
        $destination = $wordEmbeddingPath
        if (Test-Path -Path "$zipFileDir/$fileName") {
            Write-Output "word_embeddings file already exists. Skipping download & extract."
        } else {
            Invoke-Download $url $zipFileDir $fileName $destination
        }
    } else {
        Write-Output "$type is not correct."
    }
}

# 훈련 데이터 다운로드 및 압축 해제
Select-Download "train"

# 테스트 데이터 다운로드 및 압축 해제
Select-Download "test"

# GloVe 임베딩 다운로드 및 압축 해제
Select-Download "word_embeddings"

Write-Output "Data download complete!"