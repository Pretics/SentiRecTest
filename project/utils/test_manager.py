from torchmetrics import MetricCollection
from models.metrics import NDCG, MRR, AUC, SentiMRR, Senti, TopicMRR, Topic, ILS_Senti, ILS_Topic
import torch.nn.functional as F
from torch import Tensor

class TestManager:
    def __init__(self, device):
        self.device = device
        # test metrics
        self.test_performance_metrics_total = MetricCollection({
            'test_auc': AUC(),
            'test_mrr': MRR(),
            'test_ndcg@5': NDCG(k=5),
            'test_ndcg@10': NDCG(k=10)
        }).to(device)
        self.test_sentiment_diversity_metrics_vader_total = MetricCollection({
            'test_senti_mrr_vader': SentiMRR(),
            'test_senti@5_vader': Senti(k=5),
            'test_senti@10_vader': Senti(k=10)
        }).to(device)
        self.test_sentiment_diversity_metrics_bert_total = MetricCollection({
            'test_senti_mrr_bert': SentiMRR(),
            'test_senti@5_bert': Senti(k=5),
            'test_senti@10_bert': Senti(k=10)
        }).to(device)
        self.test_topic_diversity_metrics_total = MetricCollection({
            'test_topic_mrr': TopicMRR(),
            'test_topic_div@5': Topic(k=5),
            'test_topic_div@10': Topic(k=10)
        }).to(device)
        self.test_ils_senti_metrics_vader_total = MetricCollection({
            'test_ils_senti@5_vader': ILS_Senti(k=5),
            'test_ils_senti@10_vader': ILS_Senti(k=10) 
        }).to(device)
        self.test_ils_senti_metrics_bert_total = MetricCollection({
            'test_ils_senti@5_bert': ILS_Senti(k=5),
            'test_ils_senti@10_bert': ILS_Senti(k=10) 
        }).to(device)
        self.test_ils_topic_metrics_total = MetricCollection({
            'test_ils_topic@5': ILS_Topic(k=5),
            'test_ils_topic@10': ILS_Topic(k=10) 
        }).to(device)

        # test metrics
        self.test_performance_metrics_batch = MetricCollection({
            'test_auc': AUC(),
            'test_mrr': MRR(),
            'test_ndcg@5': NDCG(k=5),
            'test_ndcg@10': NDCG(k=10)
        }).to(device)
        self.test_sentiment_diversity_metrics_vader_batch = MetricCollection({
            'test_senti_mrr_vader': SentiMRR(),
            'test_senti@5_vader': Senti(k=5),
            'test_senti@10_vader': Senti(k=10)
        }).to(device)
        self.test_sentiment_diversity_metrics_bert_batch = MetricCollection({
            'test_senti_mrr_bert': SentiMRR(),
            'test_senti@5_bert': Senti(k=5),
            'test_senti@10_bert': Senti(k=10)
        }).to(device)
        self.test_topic_diversity_metrics_batch = MetricCollection({
            'test_topic_mrr': TopicMRR(),
            'test_topic_div@5': Topic(k=5),
            'test_topic_div@10': Topic(k=10)
        }).to(device)
        self.test_ils_senti_metrics_vader_batch = MetricCollection({
            'test_ils_senti@5_vader': ILS_Senti(k=5),
            'test_ils_senti@10_vader': ILS_Senti(k=10) 
        }).to(device)
        self.test_ils_senti_metrics_bert_batch = MetricCollection({
            'test_ils_senti@5_bert': ILS_Senti(k=5),
            'test_ils_senti@10_bert': ILS_Senti(k=10) 
        }).to(device)
        self.test_ils_topic_metrics_batch = MetricCollection({
            'test_ils_topic@5': ILS_Topic(k=5),
            'test_ils_topic@10': ILS_Topic(k=10) 
        }).to(device)

    def test_step(self, model, batch: dict[str, list[Tensor]], is_senti: bool = False):
        preds = model(batch)
        preds = F.softmax(preds, dim=1)
        labels = batch["labels"]
        labels = labels.to(preds.device)

        # only batch result
        r1 = self.test_performance_metrics_batch(preds, labels)
        self.test_performance_metrics_batch.reset()
        # 누적
        self.test_performance_metrics_total.update(preds, labels)

        if is_senti:
            # determine candidate sentiment and overall sentiment orientation
            s_c_vader, s_c_bert, s_mean_vader, s_mean_bert = model.sentiment_evaluation_helper(batch)
            # topical diversity
            cat_u, cat_c =  model.topical_diversity_helper(batch)

            r2 = self.test_sentiment_diversity_metrics_vader_batch(preds.flatten(), s_c_vader, s_mean_vader)
            r3 = self.test_sentiment_diversity_metrics_bert_batch(preds.flatten(), s_c_bert, s_mean_bert)
            r4 = self.test_topic_diversity_metrics_batch(preds.flatten(), cat_c, cat_u)
            r5 = self.test_ils_senti_metrics_vader_batch(preds.flatten(), s_c_vader)
            r6 = self.test_ils_senti_metrics_bert_batch(preds.flatten(), s_c_bert)
            r7 = self.test_ils_topic_metrics_batch(preds.flatten(), cat_c)

            self.test_sentiment_diversity_metrics_vader_batch.reset()
            self.test_sentiment_diversity_metrics_bert_batch.reset()
            self.test_topic_diversity_metrics_batch.reset()
            self.test_ils_senti_metrics_vader_batch.reset()
            self.test_ils_senti_metrics_bert_batch.reset()
            self.test_ils_topic_metrics_batch.reset()

            # 누적
            self.test_sentiment_diversity_metrics_vader_total.update(preds.flatten(), s_c_vader, s_mean_vader)
            self.test_sentiment_diversity_metrics_bert_total.update(preds.flatten(), s_c_bert, s_mean_bert)
            self.test_topic_diversity_metrics_total.update(preds.flatten(), cat_c, cat_u)
            self.test_ils_senti_metrics_vader_total.update(preds.flatten(), s_c_vader)
            self.test_ils_senti_metrics_bert_total.update(preds.flatten(), s_c_bert)
            self.test_ils_topic_metrics_total.update(preds.flatten(), cat_c)
        
            return [r1, r2, r3, r4, r5, r6, r7]
        else:
            return [r1]
        
    def test_step_final(self, is_senti: bool = False):
        r1 = self.test_performance_metrics_total.compute()
        if is_senti:
            r2 = self.test_sentiment_diversity_metrics_vader_total.compute()
            r3 = self.test_sentiment_diversity_metrics_bert_total.compute()
            r4 = self.test_topic_diversity_metrics_total.compute()
            r5 = self.test_ils_senti_metrics_vader_total.compute()
            r6 = self.test_ils_senti_metrics_bert_total.compute()
            r7 = self.test_ils_topic_metrics_total.compute()
            return [r1, r2, r3, r4, r5, r6, r7]
        else:
            return [r1]