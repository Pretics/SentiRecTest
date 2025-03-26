import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

class TestMetricsViewer:
    """
    Show test results by line graphs
    """
    """
    데이터 구조
    test_results = {
        experiment1: {
            metric1: score1,
            metric2: score2,
            ...
        }, experiment2: {
            metric1: score1,
            metric2: score2,
            ...
        },
        ...
    }
    """
    test_results: dict[dict[str, float]]
    size: Union[tuple[int, int], None]
    title: str
    xlabel: str
    ylabel: str
    legend: str
    is_grid_visible: bool
    is_tight_layout: bool

    def __init__(self, test_results: Union[dict[dict[str, float]], None]):
        if test_results is not None:
            self.test_results = test_results
            self.set_config()

    def change_data(self, test_results: dict[dict[str, float]]):
        self.test_results = test_results

    def set_config(
        self,
        size: Union[tuple[int, int], None] = (10, 6),
        title: str = "Evaluation Metrics Comparison",
        xlabel: str = "Metric Name",
        ylabel: str = "Metric Score",
        legend: str = "Experiments",
        is_grid_visible: bool = True,
        is_tight_layout: bool = False
    ):
        self.size = size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.is_grid_visible = is_grid_visible
        self.is_tight_layout = is_tight_layout

    def get_result_df(self):
        df = pd.DataFrame(self.test_results)
        return df

    def show(self):
        self.plt_show(
            results=self.test_results,
            size=self.size,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            legend=self.legend,
            is_grid_visible=self.is_grid_visible,
            is_tight_layout=self.is_tight_layout
        )

    def plt_show(
        self,
        results: dict[dict[str, float]],
        size: Union[tuple[int, int], None],
        title: str,
        xlabel: str,
        ylabel: str,
        legend: str,
        is_grid_visible: bool,
        is_tight_layout: bool
    ):
        # 1. 실험 결과를 DataFrame으로 변환
        test_results_df = pd.DataFrame(results)

        # 2. 빈 그래프 생성
        fig = plt.figure(figsize=size)

        # 3. 데이터 등록
        for exp_name in test_results_df.columns:
            plt.plot(
                test_results_df.index,
                test_results_df[exp_name],
                marker='o',
                label=f"{exp_name}"
            )

        # 4. 스타일링
        plt.xticks(rotation=90)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=legend)
        plt.grid(is_grid_visible)
        if is_tight_layout:
            plt.tight_layout()

        # 5. 출력
        plt.show()

        # 6. 그래프 데이터 제거
        plt.close(fig)