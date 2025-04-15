import os
from os import path

PROJECT_DIR = path.abspath(path.join(os.getcwd(), "..", ".."))

import math
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict

class NPMINewsrecDataset:
    def __init__(self, dataset_size: str):
        self.dataset_dir = path.join(PROJECT_DIR, "data", "MIND", dataset_size)

        self.train_behaviors_path = path.join(self.dataset_dir, "train", "behaviors.tsv")
        self.train_news_path = path.join(self.dataset_dir, "train", "news.tsv")

        self.test_behaviors_path = path.join(self.dataset_dir, "test", "behaviors.tsv")
        self.test_news_path = path.join(self.dataset_dir, "test", "news.tsv")

        self.dev_behaviors_path = path.join(self.dataset_dir, "dev", "behaviors.tsv")
        self.dev_news_path = path.join(self.dataset_dir, "dev", "news.tsv")

        self.train_news_df = pd.read_csv(
            self.train_news_path,
            sep='\t',
            header=None,
            names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        )
        self.train_news_df = self.train_news_df[["news_id"]].fillna("")

        self.train_behaviors_df = pd.read_csv(
            self.train_behaviors_path,
            sep='\t',
            header=None,
            names=["imp_id", "user_id", "time", "history", "impressions"]
        )
        self.train_behaviors_df = self.train_behaviors_df[["user_id", "history", "impressions"]].fillna("")

        self.start_processing()

    def start_processing(self):
        history_idxs_series, impr_idxs_series, labels_series = self._basic_process()
        self._start_calculate_NPMI(history_idxs_series)
        self._create_batch(history_idxs_series, impr_idxs_series, labels_series)

    def _basic_process(self):
        print("NPMI 계산 전, history와 impressions에 대한 전처리를 진행합니다.")
        print("1. news2int를 생성합니다.")
        # 모든 news_id를 추출하여 news_id -> idx 변환용 딕셔너리를 생성
        all_news_id = set(self.train_news_df["news_id"].tolist())
        self.news2int = {news_id: idx+1 for idx, news_id in enumerate(sorted(all_news_id))}

        print("2. Dataframe의 history를 정리합니다.")
        # history 문자열을 news_id로 쪼개기
        history_split = self.train_behaviors_df["history"].str.split(" ")
        self.train_behaviors_df.drop("history", axis=1, inplace=True)
        # 공백으로 split한 history의 뉴스 목록을 인덱스로 [변환, 중복 제거, 정렬] 후 저장
        history_idxs_series = history_split.apply(
            lambda history: sorted(set([self.news2int[nid] for nid in history if nid in self.news2int]))
        )

        print("3. Dataframe의 impressions를 정리합니다.")
        # impressions 문자열을 impr_idxs, labels로 쪼개기
        impr_split = self.train_behaviors_df["impressions"].apply(self._split_impressions)
        self.train_behaviors_df.drop("impressions", axis=1, inplace=True)
        impr_idxs_series = impr_split.apply(lambda x: x[0])
        labels_series = impr_split.apply(lambda x: x[1])

        # 자료구조 생성에 필요한 값 추출
        news_num = len(all_news_id)
        self.news_num = news_num
        self.user_num = self.train_behaviors_df["user_id"].nunique()
        self.train_behaviors_df.drop("user_id", axis=1, inplace=True)

        print("전처리를 완료했습니다.")
        return history_idxs_series, impr_idxs_series, labels_series

    def _start_calculate_NPMI(self, history_idxs_series):
        # NPMI 계산 시작
        print("각 뉴스와 뉴스 쌍의 소비 비율을 계산하기 위해, history에서의 등장 횟수를 측정합니다.(중복 제외)")
        news_count, news_pair_count = self._history_news_count(history_idxs_series)
        print("측정한 횟수를 바탕으로 모든 뉴스 쌍의 NPMI 점수를 계산합니다.")
        self.npmi_dict = self._get_npmi_dict(news_count, news_pair_count)

    def _create_batch(self, history_idxs_series, impr_idxs_series, labels_series):
        print("batch를 생성합니다.")
        print("train 데이터셋으로 계산한 뉴스 쌍의 NPMI 점수와 모든 사용자의 history를 기반으로, impression의 각 뉴스에 대한 최대 NPMI 점수를 찾습니다.")

        # dataset용
        batch = []
        for history_idxs, impr_idxs, labels in tqdm(
            zip(history_idxs_series, impr_idxs_series, labels_series),
            total=len(history_idxs_series)
        ):
            scores = []
            for impr_idx in impr_idxs:
                pref_score = 0.0
                for hist_idx in history_idxs:
                    i, j = impr_idx, hist_idx
                    if i > j:
                        i, j = j, i
                    key = (i, j)
                    if key in self.npmi_dict:
                        score = self.npmi_dict[key]
                        if pref_score == 0.0:
                            pref_score = score
                        elif score > pref_score:
                            pref_score = score
                scores.append(pref_score)
            batch.append({
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long)
            })
        self.batch = batch
        print("batch생성을 완료했습니다.")

    def _split_impressions(self, impr_str):
        pairs = impr_str.split()
        impr_news = [self.news2int[p.split("-")[0]] for p in pairs]
        labels = [int(p.split("-")[1]) for p in pairs]
        return impr_news, labels
        
    def _history_news_count(self, history_idxs_series):
        news_count = np.zeros(self.news_num+1, dtype=np.int32)

        # sparse 용 쌍 (i, j) + 값 리스트
        pair_count_dict = defaultdict(int)

        for history_idxs in tqdm(history_idxs_series):
            for idx in history_idxs:
                news_count[idx] += 1
            for idx1, idx2 in combinations(history_idxs, 2):
                pair_count_dict[(idx1, idx2)] += 1

        # sparse tensor 구성
        indices = torch.tensor(list(pair_count_dict.keys()), dtype=torch.long).T  # shape (2, N)
        values = torch.tensor(list(pair_count_dict.values()), dtype=torch.int32)
        news_pair_count = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.news_num+1, self.news_num+1),
            dtype=torch.int32
        )
        return news_count, news_pair_count

    def _get_npmi_dict(self, news_count: np.ndarray, news_pair_count: torch.Tensor) -> dict[tuple[int, int], float]:
        # Coalesce first
        coo = news_pair_count.coalesce()
        indices = coo.indices()  # shape (2, N)
        counts = coo.values().to(dtype=torch.float32)  # shape (N,)

        i = indices[0]
        j = indices[1]

        # 각 뉴스의 개별 등장 횟수
        c_i = torch.tensor(news_count, dtype=torch.float32)[i]
        c_j = torch.tensor(news_count, dtype=torch.float32)[j]

        # 유저 수 (정규화에 필요)
        N = float(self.user_num)

        # 벡터화된 NPMI 계산
        valid = (counts > 0) & (c_i > 0) & (c_j > 0)  # 유효한 경우만 계산

        # 안전한 log 연산을 위해 eps 추가
        eps = 1e-10
        c_ij = counts[valid]
        c_i = c_i[valid]
        c_j = c_j[valid]
        i = i[valid]
        j = j[valid]

        # PMI = log( c_ij * N / (c_i * c_j) )
        pmi = torch.log((c_ij * N) / (c_i * c_j + eps))
        denominator = -torch.log(c_ij / N + eps)
        npmi = pmi / (denominator + eps)

        print("모든 뉴스 pair에 대한 npmi 점수를 저장한 dict를 생성합니다.")
        npmi_dict = {}
        for idx in tqdm(range(i.size(0))):
            i_idx = i[idx].item()
            j_idx = j[idx].item()
            score_val = npmi[idx].item()
            npmi_dict[(i_idx, j_idx)] = score_val
        print("생성을 완료했습니다.")

        return npmi_dict

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        return self.batch[idx]