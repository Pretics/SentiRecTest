import os
from os import path

PROJECT_DIR = path.abspath(path.join(os.getcwd(), "..", ".."))

import torch
from tqdm import tqdm
from time import perf_counter
from models.metrics import NDCG, MRR, AUC, SentiMRR, Senti, TopicMRR, Topic, ILS_Senti, ILS_Topic
from torchmetrics import MetricCollection
import torch.nn.functional as F

from npmi_newsrec_dataset import NPMINewsrecDataset

class NPMINewsrec:
    def __init__(self, dataset_size: str):
        self.test_performance_metrics = MetricCollection({
            'val_auc': AUC(),
            'val_mrr': MRR(),
            'val_ndcg@5': NDCG(k=5),
            'val_ndcg@10': NDCG(k=10)
        })

        self.dataset = NPMINewsrecDataset(dataset_size)

    def check_npmi(self, test_count: int = 100):
        for i in range(1, test_count):
            for j in range(i+1, test_count):
                key = (i, j)
                if key not in self.dataset.npmi_dict:
                    continue
                npmi = self.dataset.npmi_dict[key]
                if npmi != 0.0:
                    print(f"({i}, {j}): {npmi}")

    def run_eval(self, min_valid_ratio=0.1):
        self.test_performance_metrics.reset()
        total, used = 0, 0

        for data in tqdm(self.dataset):
            scores = data["scores"]
            total += 1
            nonzero_ratio = (scores != 0.0).sum().item() / scores.numel()
            if nonzero_ratio < min_valid_ratio:
                continue  # skip

            y: torch.Tensor = data["labels"].unsqueeze(0)
            y_preds = F.softmax(scores.unsqueeze(0), dim=1)
            self.test_performance_metrics.update(y_preds, y)
            used += 1

        print(f"Used {used}/{total} samples.")
        return self.test_performance_metrics.compute()
    
    def run_eval_random(self):
        self.test_performance_metrics.reset()

        for data in tqdm(self.dataset):
            y: torch.Tensor = data["labels"].unsqueeze(0)
            y_preds = torch.rand(y.shape)
            self.test_performance_metrics(y_preds, y)
            
        return self.test_performance_metrics.compute()

    def benchmark(func, name, kwargs):
        start = perf_counter()
        func(**kwargs)
        end = perf_counter()
        print(f"{name}: {end - start:.4f} seconds")