import os
from os import path
from typing import Union

PROJECT_DIR = path.abspath(path.join(os.getcwd(), "..", ".."))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize
from utils.dataset_frame_loader import DatasetFrameLoader

class MINDAnalyzer:
    def __init__(self, dataset_size):
        loader = DatasetFrameLoader(dataset_size, is_dev=False)
        loader.load_all()

        self.train_news_df = loader.train_news_df
        self.train_behaviors_df = loader.train_behaviors_df
        self.test_news_df = loader.test_news_df
        self.test_behaviors_df = loader.test_behaviors_df

        # title 단어 길이 계산
        self.train_news_df["title_len"] = self.train_news_df["title"].str.split().str.len()
        # title 토큰 길이 계산
        self.train_news_df["title_len_tok"] = self.train_news_df["title"].apply(lambda x: len(word_tokenize(x)))

        # abstract 단어 길이 계산
        self.train_news_df["abstract_len"] = self.train_news_df["abstract"].str.split().str.len()
        # abstract 토큰 길이 계산
        self.train_news_df["abstract_len_tok"] = self.train_news_df["abstract"].apply(lambda x: len(word_tokenize(x)))

        # behavior 데이터의 time 문자열을 timestamp 형식으로 변경
        self.train_behaviors_df["time"] = pd.to_datetime(self.train_behaviors_df["time"], format="%m/%d/%Y %I:%M:%S %p")

        # history 길이 계산
        self.train_behaviors_df["history_len"] = self.train_behaviors_df["history"].str.split().str.len()

    def show_len_graph(
            self,
            counts,
            data_label: str,
            x_label: str,
            y_label: str,
            title: str,
            scale_text: Union[str, None] = None,
            x_tick_start: int = 0,
            x_tick_step: int = 2,
            y_ticks: Union[list[float], None] = None,
            figsize: tuple[int, int] = (8, 4),
            range_x: tuple[int, int] = (0, 31),
            padding: tuple[int, int] = (0, 0),
        ):
        # x 범위 지정
        min_len = counts.index.min()
        max_len = counts.index.max()
        x_vals = list(range(min_len, max_len + 1))

        # 높이/위치 계산
        title_height = [counts.get(x, 0) for x in x_vals]
        max_y = max(title_height)
        bar_width = 0.8
        bar_x = [x - 0.5 for x in x_vals]

        # 그래프 시각화
        plt.figure(figsize=figsize)
        plt.bar(bar_x, title_height, width=bar_width, label=data_label, align='center')

        # 현재 축 가져오기
        ax = plt.gca()

        # x축 눈금 -> 짝수만 표시
        xtick_positions_filtered = [bar_x[i] for i, x in enumerate(x_vals) if x % x_tick_step == x_tick_start]
        xtick_labels = [x for x in x_vals if x % x_tick_step == x_tick_start]
        plt.xticks(xtick_positions_filtered, xtick_labels)

        # x축 범위 제한
        plt.xlim(range_x[0], range_x[1])

        # y축 눈금선만 수동으로 추가
        ax.yaxis.grid(True, linestyle='dotted', alpha=1, color='#000') 

        # 위쪽 여백 주기
        plt.ylim(0, max_y * (1 + padding[1]))

        # 데이터 실제 축척 표시하기
        if scale_text is not None:
            ax.text(0, max_y * (1.07 + padding[1]), scale_text, fontsize=11, verticalalignment='top')

        if y_ticks is not None:
            plt.yticks(y_ticks)
            
        plt.xlabel(x_label)
        plt.ylabel(y_label, fontsize=11, fontname='DejaVu Sans')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_title_len(
            self,
            is_token_len: bool = True,
            size: tuple[int, int] = (8, 4),
            range_x: tuple[int, int] = (0, 31)
        ):
        if is_token_len:
            # 토큰 개수 비율 계산
            title_counts = self.train_news_df["title_len_tok"].value_counts(normalize=True).sort_index().apply(lambda x: x*10)
            self.show_len_graph(
                title_counts,
                data_label="title",
                x_label="Title Length",
                y_label="Ratio",
                title="Distribution of Title Lengths",
                scale_text="1e-1",
                x_tick_start=1,
                x_tick_step=2,
                y_ticks=np.arange(0, 1.26, 0.25).tolist(),
                figsize=size,
                range_x=range_x,
                padding=(0, 0.2)
            )
        else:
            # (공백 구분) 단어 수 계산
            title_counts = self.train_news_df["title_len"].value_counts(normalize=False).sort_index()
            self.show_len_graph(
                title_counts,
                data_label="title",
                x_label="Title Length",
                y_label="Ratio",
                title="Distribution of Title Lengths",
                x_tick_start=1,
                x_tick_step=2,
                figsize=size,
                range_x=range_x,
                padding=(0, 0.05)
            )

    def show_abstract_len(
            self,
            is_token_len: bool = True,
            is_remove_zero: bool = True,
            size: tuple[int, int] = (8, 4),
            range_x: tuple[int, int] = (0, 31)
        ):
        if is_token_len:
            # 토큰 개수 비율 계산
            abstract_counts = self.train_news_df["abstract_len_tok"].value_counts(normalize=True).sort_index().apply(lambda x: x*100)
            if is_remove_zero:
                abstract_counts = abstract_counts[abstract_counts.index != 0]
            self.show_len_graph(
                abstract_counts,
                data_label="Abstract",
                x_label="Abstract Length (Token)",
                y_label="Ratio",
                title="Distribution of Abstract Lengths",
                scale_text="1e-2",
                x_tick_start=0,
                x_tick_step=5,
                y_ticks=np.arange(0, 3.01, 0.5).tolist() if is_remove_zero else np.arange(0, 5.01, 0.5).tolist(),
                figsize=size,
                range_x=range_x,
                padding=(0, 0.05)
            )
        else:
            # (공백 구분) 단어 수 계산
            abstract_counts = self.train_news_df["abstract_len"].value_counts(normalize=False).sort_index()
            if is_remove_zero:
                abstract_counts = abstract_counts[abstract_counts.index != 0]
            self.show_len_graph(
                abstract_counts,
                data_label="abstract",
                x_label="Abstract Length",
                y_label="Sample Number",
                title="Distribution of Abstract Lengths",
                x_tick_start=0,
                x_tick_step=5,
                figsize=size,
                range_x=range_x,
                padding=(0, 0.05)
            )

    def show_history_len(
            self,
            size: tuple[int, int] = (8, 4),
            range_x: tuple[int, int] = (0, 81)
        ):
        history_counts = self.train_behaviors_df["history_len"].value_counts(normalize=False).sort_index()

        self.show_len_graph(
            history_counts,
            data_label="history",
            x_label="History Length",
            y_label="Sample Number",
            title="Distribution of History Lengths",
            x_tick_start=0,
            x_tick_step=5,
            figsize=size,
            range_x=range_x,
            padding=(0, 1.05)
        )

    def show_impr_time_distribution(self):
        time_group = self.train_behaviors_df["time"].dt.floor("h")
        clicks_by_time = time_group.value_counts().sort_index()

        # 3시간 간격 x축
        start = clicks_by_time.index.min().floor("3h")
        end = clicks_by_time.index.max().ceil("3h")
        xtick_locs = pd.date_range(start=start, end=end, freq="3h")
        xtick_labels = [ts.strftime("%H:%M") for ts in xtick_locs]

        # 날짜 표시용 라벨 위치 (12시간마다)
        date_ticks = pd.date_range(start=start, end=end, freq="24h")
        date_labels = [ts.strftime("%Y-%m-%d") for ts in date_ticks]

        # 그래프
        fig, ax = plt.subplots(figsize=(14, 6))

        # 선 그래프
        ax.plot(clicks_by_time.index, clicks_by_time.values, marker='.')

        # 시간 단위 x축 눈금
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, rotation=90)

        # 날짜 레이블을 2번째 x축에 별도로 표시
        for dt, label in zip(date_ticks, date_labels):
            ax.text(dt, -max(clicks_by_time.values) * 0.15,  # 아래 여백
                    label, rotation=90, ha='center', va='top', fontsize=9, color='dimgray')

        # 기타 설정
        ax.set_xlabel("")
        ax.set_ylabel("Number of Impressions")
        ax.set_title("Hourly Impression Distribution")
        ax.grid(True, linestyle="--")
        plt.tight_layout()
        plt.show()

    def get_survival_time_distribution(self):
        records = []

        for time, impression_str in zip(self.train_behaviors_df["time"], self.train_behaviors_df["impressions"]):
            for item in impression_str.split():
                news_id = item.split('-')[0]  # N12345-1 → N12345
                records.append((news_id, time))

        # DataFrame 생성
        impression_records = pd.DataFrame(records, columns=["news_id", "time"])

        # datetime 변환
        impression_records["time"] = pd.to_datetime(impression_records["time"])

        # 그룹별 최소/최대 노출 시간
        news_time_span = impression_records.groupby("news_id")["time"].agg(["min", "max"])

        # 생존 시간 계산 (일 단위)
        news_time_span["survival_hours"] = (news_time_span["max"] - news_time_span["min"]).dt.total_seconds() / (60 * 60)

        # round down to nearest 0.2 day
        bins = pd.cut(news_time_span["survival_hours"], bins=pd.interval_range(start=0, end=news_time_span["survival_hours"].max() + 0.5, freq=0.5))
        survival_distribution = bins.value_counts().sort_index()

        return survival_distribution

    def show_survival_time(self):
        survival_distribution = self.get_survival_time_distribution()

        # x축: bin의 중간값
        x = [interval.left + 0.05 for interval in survival_distribution.index]
        y = survival_distribution.values

        plt.figure(figsize=(12, 5))
        plt.bar(x, y, width=0.4, label="impressions", align='edge')

        # x축 범위 제한
        plt.xlim(0, 30)

        # x축 눈금 간격 고정
        plt.xticks(np.arange(0, 30.1, 1))

        plt.xlabel("Survival Time (0.5h range)")
        plt.ylabel("Number of News Articles")
        plt.title("News Survival Time Distribution")
        ax = plt.gca()
        ax.yaxis.grid(True, linestyle='dotted', alpha=1, color='#000')

        plt.tight_layout()
        plt.show()