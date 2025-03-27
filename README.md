# SentiRec 수정 버전
### Original Code: https://github.com/MeteSertkan/newsrec
#### 변경 사항
### (정리 중)
1. 윈도우 환경에서 정상적으로 동작하도록 수정했습니다. (Ubuntu에서 테스트는 아직 못 해봤습니다.)
2. ``pytorch``, ``pytorch_lightning``, ``torchmetrics`` 등 학습 관련 라이브러리를 최신 버전으로 업그레이드 하고, 해당 버전에 맞게 코드 일부를 수정했습니다.
3. 데이터 전처리를 담당하는 ``parse_behavior.py``와 ``parse_news.py``에 인자 입력 코드만 남기고 전처리 실행 코드를 ``prep_behavior.py``와 ``prep_news.py``로 분리하였으며, 코드 구조를 (개인적으로는)좀 더 가독성 좋게 변경했습니다.
4. ``prep.sh``에 대응되는 ``prep_download.ps1``, ``prep_process.ps1``을 만들었습니다. 
5. 데이터 전처리 과정을 단계별로 관찰하기 용이하도록 ``prep.ipynb``를 만들었습니다.
6. 학습, 테스트를 단계별로 관찰하기 용이하도록 ``train.ipynb``와 ``test.ipynb``를 만들었습니다.
7. Train/Test 데이터셋을 따로따로 처리하는 기존의 코드 대신, 두 데이터를 한 번에 처리하는 ``_combined.`` 버전 코드를 만들었습니다. (원본 코드와 생성되는 결과물은 동일합니다.)
8. MIND 데이터셋의 demo/small/large 버전을 쉽게 변경하며 테스트하기 위해 폴더 구조를 획일화(원본 데이터셋: ``data/MIND/<size>/...``, 전처리 데이터: ``data/preprocess_data/<size>/...``)하고, 각종 명령어와 config 파일, 코드 일부를 변경 사항에 맞게 수정했습니다.
9. 데이터셋의 사이즈를 다양하게 구성해볼 수 있는 generate_manual_dataset.ipynb를 추가했습니다.
10. train.py와 test.py의 기능 대부분을 train_manager.py, test_manager.py로 이전, 클래스의 형태로 사용하기 쉽게 변경하였습니다.

# 환경 설정
## Conda
(테스트는 안 해봤습니다. 만약 이걸로 안 되면 PyTorch 라이브러리를 [해당 링크](https://pytorch.org/get-started/locally/)에서 구한 명령어로 설치한 뒤 pytorch_lightning, torchmetrics를 따로 설치해 주세요.)

1. 프로젝트에 동봉된 environment.yml로 가상 환경 설치
    ```
    conda env create -f environment.yml
    ```
2. env 활성화: 
    ```
    conda activate newsrec
    ```

# 데이터 전처리
powershell 스크립트, ``.ipynb``, 터미널 명령어 세 가지 가지 방법 중 하나만 실행하면 됩니다. 둘 다 동일하게 동작합니다.
## powershell 스크립트
(``project/data`` 폴더 내에서 실행)
1. 데이터셋 다운로드
```
./preprocess/prep_download.ps1 -size <size>
```
2. 데이터 전처리 진행
```
./preprocess/prep_process.ps1 -size <size>
```
``<size>``: demo, small, large 셋 중 하나 입력
## project/data 내의 prep.ipynb 혹은 prep_combined.ipynb
둘 다 결과물은 동일합니다. <br/>
``prep.ipynb``는 train과 test 데이터셋의 전처리를 따로따로 실행하며, ``prep_combined.ipynb``는 한 번에 진행합니다.
## 터미널 명령어 1
별로 추천하지 않는 방법입니다. <br/>
``<size>`` 부분을 사용할 데이터셋의 크기에 맞게 변경해야 하며, 데이터셋을 미리 다운 받아야 합니다. <br/>
(``project/data`` 폴더 내에서 실행)
1. train/test behaviors.tsv 전처리
```
python preprocess/parse_behavior_combined.py --train-file MIND/<size>/train/behaviors.tsv --test-file MIND/<size>/test/behaviors.tsv --train-out preprocessed_data/<size>/train --test-out preprocessed_data/<size>/test --user2int preprocessed_data/<size>/train/user2int.tsv
```
2. train/test news.tsv 전처리
```
python preprocess/parse_news_combined.py --train-file MIND/<size>/train/news.tsv --test-file MIND/<size>/test/news.tsv --train-out preprocessed_data/<size>/train --test-out preprocessed_data/<size>/test --word-embeddings word_embeddings/glove.840B.300d.txt
```
## 터미널 명령어 2
터미널 명령어 1과 동일한 결과물을 생성합니다. <br/>
``<size>`` 부분을 사용할 데이터셋의 크기에 맞게 변경해야 하며, 데이터셋을 미리 다운 받아야 합니다. <br/>
(``project/data`` 폴더 내에서 실행)
1. train behaviors.tsv 전처리
```
python parse_behavior.py --in-file MIND/<size>/train/behaviors.tsv --out-dir preprocessed_data/<size>/train --mode train
```
2. test behaviors.tsv 전처리
```
python parse_behavior.py --in-file MIND/<size>/test/behaviors.tsv --out-dir preprocessed_data/<size>/test --mode test --user2int preprocessed_data/<size>/train/user2int.tsv 
```
3. train news.tsv 전처리
```
python parse_news.py --in-file MIND/<size>/train/news.tsv --out-dir preprocessed_data/<size>/train --mode train --word-embeddings word_embeddings/glove.840B.300d.txt
```
4. test news.tsv 전처리
```
python parse_news.py --in-file MIND/<size>/test/news.tsv --out-dir preprocessed_data/<size>/test --mode test --word-embeddings word_embeddings/glove.840B.300d.txt --embedding-weights preprocessed_data/<size>/train/embedding_weights.csv  --word2int preprocessed_data/<size>/train/word2int.tsv --category2int preprocessed_data/<size>/train/category2int.tsv  
```


# 모델 학습
## 터미널 명령어
(``project`` 폴더 내에서 실행)
#### Train
```
python train.py --config <config_path>
```
#### Test
```
python test.py --config <config_path> --ckpt <ckpt_path>
```
``<config_path>`` example: ``./config/model/sentirec/vader_lambda0p4_mu10.yaml`` <br/>

``<ckpt_path>`` example: ``./logs/lightning_logs/checkpoints/sentirec/vader_lambda0p4_mu10/epoch=X-val_auc_epoch=0.XXXX.ckpt`` <br/>

# 모니터링
## 터미널 명령어
(``project`` 폴더 내에서 실행)
```
tensorboard --logdir <path_to_tensorboard_logdir>
```
``<path_to_tensorboard_logdir>`` example: ``logs/lightning_logs/tensorboard/sentirec/vader_lambda0p4_mu10`` <br/>

내부에 .0 파일이 여러개 존재할 경우 보고싶은 파일 하나를 골라 다른 폴더로 옮겨서 확인하는 것을 추천합니다.

이하 원본 repository의 README 내용입니다.

# NewsRec
Welcome 👋 to the repo of our paper:

**Diversifying Sentiments in News Recommendation** <br/>
Mete Sertkan, Sophia Althammer, Sebastian Hofstätter and Julia Neidhardt <br/>
[https://ceur-ws.org/Vol-3228/paper3.pdf](https://ceur-ws.org/Vol-3228/paper3.pdf)

**tldr;**  We aim to reproduce SentiRec [(Wu et al., 2020)](https://www.aclweb.org/anthology/2020.aacl-main.6.pdf), a sentiment diversity-aware news recommender that aims to counter the lack of sentiment-diversity in personalized news recommendations. We re-implemented the SentiRec model from scratch and used the Microsoft [MIND dataset](https://msnews.github.io) [(Wu et al., 2020)](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) for training and evaluation. We have evaluated and discussed our reproduction from different perspectives. In addition to comparing the recommendation list to the user’s interaction history, as the original paper did, we also analyzed the intra-list sentiment diversity of the recommendation list. We also studied the effect of sentiment diversification on topical diversity. Our results suggest that SentiRec does not generalize well to other data sets and that the compared baselines already perform well.

Checkout [our paper](https://ceur-ws.org/Vol-3228/paper3.pdf) for more details, and please feel free to use our source code and let us know if you have any questions or feedback ([mete.sertkan@tuwien.ac.at](mailto:mete.sertkan@tuwien.ac.at) or dm to[@m_sertkan](https://twitter.com/m_sertkan)).

We call this repo, NewsRec, since we also provide all (re-implemented) baselines and an adapted version of SentiRec, i.e. RobustSentiRec, 🤗.
Currently our repository contains following models:
- [LSTUR (An et al., 2019)](https://www.aclweb.org/anthology/P19-1033/)
- [NAML (WU et al., 2019)](https://arxiv.org/abs/1907.05576)
- [NRMS (Wu et al., 2019)](https://www.aclweb.org/anthology/D19-1671/)
- [SentiRec (Wu et al., 2020)](#sentirec)
- [RobustSentiRec (Sertkan et al, 2022)](#robustsentirec)

**Please cite our work as:**
```
@inproceedings{sertkan2022diversifying,
	title        = {Diversifying Sentiments in News Recommendation},
	author       = {Sertkan, Mete and Althammer, Sophia and Hofst{\"a}tter, Sebastian and Neidhardt, Julia},
	year         = 2022,
	booktitle    = {Proceedings of the Perspectives on the Evaluation of Recommender Systems Workshop 2022 co-located with the 16th ACM Conference on Recommender Systems (RecSys 2022)}
}
```

## SentiRec
![](figures/sentirec_framework.png)

SentiRec builds upon the NRMS model and learns through an auxiliary sentiment-prediction task in the news encoding sentiment-aware news representations and in turn penalizes recommendations, which have a similar sentiment orientation as the user’s history track.

## RobustSentiRec
![](figures/robust_sentirec_framework.png)

We adapt SentiRec by omitting the auxiliary sentiment-prediction task. Instead, we directly incorporate the sentiment-polarity scores determined by the sentiment-analyzer into the news representation by concatenating it with the output of the news encoder and applying a linear transformation.


# How to Train and Test
## Requirements
In our implementations we use the combo pytorch - pytorchlighnting - torchmetrics. Thus, a cuda device is suggested and more are even better! We are using conda and have exported the conda-environment we are working with, thus: 
1. Install [conda](https://docs.conda.io/en/latest/)
2. Create an environment with the provided environment.yml:

    ```
    conda env create -f environment.yml
    ```
3. Activate the environment as following: 
    ```
    conda activate newsrec
    ```
## Data
In our experiments we use the Microsoft News Dataset, i.e., MIND. In particular we have used MINDsmall_train in a 9:1 split for trainig and validation; and the MINDsmall_dev as our holdout. The datasets and more detail are provided [here](https://msnews.github.io/index.html). Furthermore, we use 300-dimensional Glove embeddings, which can be downloaded [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). 

In order to train our models, you need to pre-preprocess the data. Therefore, we provide ``./project/data/parse_behavior.py`` and ``./project/data/parse_news.py``. 

Run the prepreocessing as following (or use ``project/data/prep.sh`` ;) ):
1. Download the MIND dataset and the Glove embeddings
2. Create two directories ``<train_dir>`` and ``<test_dir>``; one for training and for the testing data.
3. Preprocess the impression logs of the mind-trainig data as follows: 
```
python parse_behavior.py --in-file mind/train/behaviors.tsv --out-dir <train_dir> --mode train
```
4. Preprocess the impressions logs of the mind-test data as follows:
```
python parse_behavior.py --in-file mind/test/behaviors.tsv --out-dir <test_dir> --mode test --user2int <train_dir>/user2int.tsv 
```
5. Preprocess the news content of the mind-train data as follows:
```
python parse_news.py --in-file mind/train/news.tsv --out-dir <train_dir> --mode train --word-embeddings glove.840B.300d.txt
```
6. Preprocess the news content of the mind-test data as follows: 
```
python parse_news.py --in-file mind/test/news.tsv --out-dir <teset_dir> --mode test --word-embeddings glove.840B.300d.txt --embedding-weights <train_dir>/embedding_weights.csv  --word2int <train_dir>/word2int.tsv --category2int <train_dir>/category2int.tsv  
```


## Config
We provide for each model example configs under ``./project/config/model/_model_name_``. The config files contain general information, hyperparameter, meta-information (logging, checkpointing, ...). All hyperparemters are either trained on the validation set or referr to the corresponding papers. The generall structure of the config is as follwing: 

- GENERAL: General information about the model like the name of the model to be trained (e.g., ``name: "sentirec"``). 
- DATA: Paths to (pre-processed) train and test data files and information about the underlying data (might be changed according to the pre-processing); Features to be used in the model, e.g. title, vader_sentiment, distillbert_sst2_sentiment, etc. 
- MODEL: Hyperparams of the model and model-architecture, e.g., learning_rate, dropout_probability, num_attention_heads, sentiment_classifier, etc. 
- TRAIN: Config for the Checkpointing, Logging, Earlystop; Dataloader information (e.g., batch_size, num_workers ,etc.); and Config  for the trainer (e.g., max_epochs, fast_dev_run, etc.)

Please, change the config according to your needs. You may want to change the paths to: i) train/test - data; ii) directory for the checkpoints; iii) directory for the logging.

## Train-Run
You can train now the models using ``./project/train.py``. Here an example for running SentiRec: 
```
CUDA_VISIBLE_DEVICES=0,1 ./project/train.py --config ./project/config/model/sentirec/vader_lambda0p4_mu10.yaml
```
The name parameter in the config defines which model to train, here it is set to "sentirec". With ``CUDA_VISIBLE_DEVICES=ids`` you can define on which CUDA devices to train. If you do not define any device and run just ``./project/train.py`` all visible devices will be used. 

Currently ``train.py`` saves the last checkpoint and the one of best three epochs (can be configured) based on validation-AUC. In case of an interruption it attempts a gracefull shutdown and saves the checkpoint aswell. You can resume training by providing a path to a checkpoint with  ``--resume`` like: 
```
CUDA_VISIBLE_DEVICES=0,1 ./project/train.py --config ./project/config/model/sentirec/vader_lambda0p4_mu10.yaml --resume _path_to_checkpoint_
```

## Test-Run
For Testing use the same config as you have trained, since it contains all necessary information, e.g., path to test data, config of test-dataloader, logging info, etc. It may take longer, but use only ONE GPU for testing (e.g., ``CUDA_VISIBLE_DEVICES=_one_gpu_id_of_your_choice_``). Only test once, right before you report/submit (after validation, tuning, ...). You have to provide a checkpoint of your choice (e.g., best performing hyperparam setting according to val, etc.) and can run a test as following:
```
CUDA_VISIBLE_DEVICES=0 ./project/test.py --config project/config/model/sentirec/vader_lambda0p4_mu10.yaml --ckpt path_to_ckpt_of_your_choice
```

## Monitoring
We use the default logger of pytorchlightning, i.e., tensorboard. You can monitor your traning, validation, and testing via tensorboard. You have to provide the path of your logs and can start tensorboard as following: 
```
tensorboard --logdir _path_to_tensorboard_logdir_
```


# Acknowledgments
We build upon [Microsoft's recommenders repo](https://github.com/microsoft/recommenders). 

    
