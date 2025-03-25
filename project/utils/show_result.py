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


    def __init__(self, test_results: Union[dict[dict[str, float]], None]):
        if test_results is not None:
            self.test_results = test_results

    def show(
        self,
        size: Union[tuple[int, int], None],

    ):
        # 1. 실험 결과를 DataFrame으로 변환
        test_results_df = pd.DataFrame(self.test_results)
        
        # 2. 그래프용 데이터 등록
        plt.figure(figsize=size)

        for exp_name in test_results_df.columns:
            plt.plot(
                test_results_df.index,
                test_results_df[exp_name],
                marker='o',
                label=f"{exp_name}"
            )

        # 3. 스타일링
        plt.xticks(rotation=90)
        plt.ylabel("Metric Score")
        plt.xlabel("Metric Name")
        plt.title("Evaluation Metrics Comparison")
        plt.legend(title="Experiments")
        plt.grid(True)
        plt.tight_layout()

        # 4. 출력
        plt.show()