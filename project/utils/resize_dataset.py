from os import path
import csv
import random
import pandas as pd
from tqdm import tqdm

def pick_random_integers(min: int, max: int, k: int, seed: int):
    random.seed(seed)
    if k > (max - min + 1):
        raise ValueError("샘플 개수가 너무 많습니다.")
    elif k < 1:
        raise ValueError("샘플 개수는 0보다 커야합니다.")
    return random.sample(range(min, max + 1), k)

def generate_behaviors_dataset(
        original_path: str,
        out_path: str,
        data_number: int,
        seed: int
    ):
    # 원본 데이터셋을 불러옵니다.
    with open(original_path, 'r') as original_behavior_file:
        original_behaviors = original_behavior_file.readlines()

    # 원본 데이터셋에서 뽑아올 샘플의 인덱스를 랜덤으로 생성합니다.
    indexes = pick_random_integers(0, len(original_behaviors) - 1, data_number, seed)

    # 새로운 behaviors.tsv 파일을 생성합니다.
    with open(out_path, 'w', newline='') as behavior_out_file:
        behaviors_writer = csv.writer(behavior_out_file, delimiter='\t')
        for index in tqdm(indexes):
            behavior_data = original_behaviors[index]
            behaviors_writer.writerow(behavior_data.strip().split('\t'))

def generate_news_dataset(
        behaviors_path: str,
        original_news_path: str,
        out_path: str
    ):
    """
    새로 생성하는 데이터셋의 behaviors.tsv에 포함된 뉴스 목록으로 news.tsv를 생성합니다.<br/>

    Parameters
    -------------
    `behaviors_path`: 새로 생성하는 데이터셋의 behaviors.tsv 경로입니다. <br/>
    `original_news_path`: 원본 데이터셋의 news.tsv 경로입니다. <br/>
    `out_path`: 새로 생성하는 데이터셋의 news.tsv를 저장할 경로입니다. <br/>
    """
    # 생성한 behaviors.tsv 데이터셋을 불러옵니다.
    with open(behaviors_path, 'r') as original_behavior_file:
        behaviors = original_behavior_file.readlines()

    # 원본 news.tsv 데이터셋을 불러옵니다.
    news_columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entitles", "abstract_entities"]
    original_news_df = pd.read_csv(original_news_path, sep='\t', header=None, names=news_columns, encoding='utf-8', index_col="news_id")

    # manual/train/behaviors.tsv로 선별한 모든 샘플에 포함된 뉴스 목록을 news.tsv에 저장하기 위한 전처리 과정을 시작합니다.
    news_collection = set()
    for behavior_data in tqdm(behaviors):
        imp_id, user_id, time, history, impressions = behavior_data.strip().split('\t')
        # NewsID가 들어 있는 history, impressions에서 ID만 빼옵니다.
        history = history.split(' ')
        impressions = [s.split('-')[0] for s in impressions.split(' ')]
        # 집합에 추가해서 중복을 제거합니다.
        # 가끔 history나 impressions가 없으면 ['']이 저장되는데
        # 이걸 집합에 추가하면 Dataframe 생성시 문제가 생기므로
        # 조건문으로 걸러줍니다.
        if history[0] != '':
            news_collection.update(history)
        if impressions[0] != '':
            news_collection.update(impressions)

    # 중복 없이 뽑아낸 모든 뉴스ID의 데이터를 선별합니다.
    news_df = original_news_df.loc[list(news_collection)]

    # 선별한 데이터를 news.tsv에 저장합니다.
    news_df.to_csv(out_path, sep='\t', header=None, encoding='utf-8')
    return news_df

def resize_dataset(dataset_dir, original_dataset_dir, train_data_num, test_data_num, random_seed):
    generate_behaviors_dataset(
        path.join(original_dataset_dir, "train", "behaviors.tsv"),
        path.join(dataset_dir, "train", "behaviors.tsv"),
        train_data_num,
        random_seed
    )
    generate_behaviors_dataset(
        path.join(original_dataset_dir, "test", "behaviors.tsv"),
        path.join(dataset_dir, "test", "behaviors.tsv"),
        test_data_num,
        random_seed
    )
    generate_news_dataset(
        path.join(dataset_dir, "train", "behaviors.tsv"),
        path.join(original_dataset_dir, "train", "news.tsv"),
        path.join(dataset_dir, "train", "news.tsv")
    )
    generate_news_dataset(
        path.join(dataset_dir, "test", "behaviors.tsv"),
        path.join(original_dataset_dir, "test", "news.tsv"),
        path.join(dataset_dir, "test", "news.tsv")
    )