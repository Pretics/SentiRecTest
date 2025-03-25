import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

class TestMetricsViewer:
    test_results: dict[dict[str, float]]
    """
    test_results = {
        experiment1: {
            metrics1: score1,
            metrics2: score2,
            ...
        },
        experiment2: {
            metrics1: score1,
            metrics2: score2,
            ...
        },
        ...
    }
    """
    size: Union[tuple[int, int], None]
    title: str
    xlabel: str
    ylabel: str
    legend: str
    grid: bool


    def __init__(self, test_results: Union[dict[dict[str, float]], None]):
        if test_results is not None:
            self.test_results = test_results
            self.set_config()

    def set_config(
        self,
        size: Union[tuple[int, int], None] = (10, 6),
        title: str = "Evaluation Metrics Comparison",
        xlabel: str = "Metric Name",
        ylabel: str = "Metric Score",
        legend: str = "Experiments",
        grid: bool = True
    ):
        self.size = size,
        self.title = title,
        self.xlabel = xlabel,
        self.ylabel = ylabel,
        self.legend = legend,
        self.grid = grid

    def show(self):
        self.plt_show(
            size=self.size,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            legend=self.legend,
            grid=self.grid
        )

    def plt_show(
        self,
        size: Union[tuple[int, int], None],
        title: str,
        xlabel: str,
        ylabel: str,
        legend: str,
        grid: bool
    ):
        # 1. 실험 결과를 DataFrame으로 변환
        test_results_df = pd.DataFrame(self.test_results)

        # 2. 데이터 등록
        for exp_name in test_results_df.columns:
            plt.plot(
                test_results_df.index,
                test_results_df[exp_name],
                marker='o',
                label=f"{exp_name}"
            )

        # 3. 스타일링
        plt.figure(figsize=size)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=legend)
        plt.grid(grid)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # 4. 출력
        plt.show()