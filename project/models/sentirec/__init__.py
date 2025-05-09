import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.sentirec.news_encoder import NewsEncoder
from models.sentirec.user_encoder import UserEncoder
from models.utils import TimeDistributed
from models.metrics import NDCG, MRR, AUC, SentiMRR, Senti, TopicMRR, Topic, ILS_Senti, ILS_Topic
from torch import Tensor

class SENTIREC(pl.LightningModule):
    """
    SENTIREC network.
    """

    def __init__(
            self, 
            config=None, 
            pretrained_word_embedding:Tensor=None
        ):
        super(SENTIREC, self).__init__()
        self.config = config
        news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.news_encoder = TimeDistributed(news_encoder, batch_first=True)
        self.user_encoder = UserEncoder(config)
        self.sentiment_predictor = nn.Linear(config.word_embedding_dim, 1)
        # val metrics
        self.val_performance_metrics = MetricCollection({
            'val_auc': AUC(),
            'val_mrr': MRR(),
            'val_ndcg@5': NDCG(k=5),
            'val_ndcg@10': NDCG(k=10)
        })
        self.val_sentiment_diversity_metrics_vader = MetricCollection({
            'val_senti_mrr_vader': SentiMRR(),
            'val_senti@5_vader': Senti(k=5),
            'val_senti@10_vader': Senti(k=10)
        })
        self.val_sentiment_diversity_metrics_bert = MetricCollection({
            'val_senti_mrr_bert': SentiMRR(),
            'val_senti@5_bert': Senti(k=5),
            'val_senti@10_bert': Senti(k=10)
        })
        # test metrics
        self.test_performance_metrics = MetricCollection({
            'test_auc': AUC(),
            'test_mrr': MRR(),
            'test_ndcg@5': NDCG(k=5),
            'test_ndcg@10': NDCG(k=10)
        })
        self.test_sentiment_diversity_metrics_vader = MetricCollection({
            'test_senti_mrr_vader': SentiMRR(),
            'test_senti@5_vader': Senti(k=5),
            'test_senti@10_vader': Senti(k=10)
        })
        self.test_sentiment_diversity_metrics_bert = MetricCollection({
            'test_senti_mrr_bert': SentiMRR(),
            'test_senti@5_bert': Senti(k=5),
            'test_senti@10_bert': Senti(k=10)
        })
        self.test_topic_diversity_metrics = MetricCollection({
            'test_topic_mrr': TopicMRR(),
            'test_topic_div@5': Topic(k=5),
            'test_topic_div@10': Topic(k=10)
        })
        self.test_ils_senti_metrics_vader = MetricCollection({
            'test_ils_senti@5_vader': ILS_Senti(k=5),
            'test_ils_senti@10_vader': ILS_Senti(k=10) 
        })
        self.test_ils_senti_metrics_bert = MetricCollection({
            'test_ils_senti@5_bert': ILS_Senti(k=5),
            'test_ils_senti@10_bert': ILS_Senti(k=10) 
        })  
        self.test_ils_topic_metrics = MetricCollection({
            'test_ils_topic@5': ILS_Topic(k=5),
            'test_ils_topic@10': ILS_Topic(k=10) 
        })
        

        
    def forward(self, batch):
        """
        # embedding_weights <-> word2int 관련 index out of bounds 테스트
        word_embedding = self.news_encoder.module.word_embedding
        
        print(f"batch['c_title'].max(): {batch['c_title'].max()}, embedding weight size: {word_embedding.weight.shape[0]}")
        #assert batch["c_title"].max() < word_embedding.weight.shape[0], "ERROR: Index out of bounds!"

        print(f"batch['c_title'].min(): {batch['c_title'].min()}")
        #assert batch["c_title"].min() >= 0, "ERROR: Negative index found in batch['c_title']!"
        """
        # encode candidate news
        candidate_news_vector = self.news_encoder(batch["c_title"])
        # encode history 
        clicked_news_vector = self.news_encoder(batch["h_title"])
        # encode user
        user_vector = self.user_encoder(clicked_news_vector)
        # compute scores for each candidate news
        clicks_score = torch.bmm(
            candidate_news_vector,
            user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        # sentiment-prediction task
        s_pred = self.sentiment_predictor(candidate_news_vector)
        
        return clicks_score, s_pred

    def training_step(self, batch, batch_idx):
        # forward pass
        y_pred, s_pred = self(batch)
        y_pred = torch.sigmoid(y_pred)
        # RECOMMENDATION LOSS
        y = torch.zeros(len(y_pred), dtype=torch.long, device=self.device)
        loss = F.cross_entropy(y_pred, y)
        # SENTIMENT PREDICTION LOSS
        s_c = batch["c_" + self.config.sentiment_classifier].flatten()
        loss += self.config.sentiment_prediction_loss_coeff * F.l1_loss(s_pred.flatten(), s_c) #MAE
        # SENTIMENT REGULARIZATION LOSS
        if(self.config.sentiment_regularization): 
            s_hist = batch["h_" + self.config.sentiment_classifier].flatten()
            s_mean = s_hist.mean()
            # batch_size, 1+K // sentiment diversity score
            p = F.relu(s_mean * s_c * y_pred.flatten())
            # sentiment diversity loss
            loss += self.config.sentiment_diversity_loss_coeff * p.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, _ = self(batch)
        y_pred = F.softmax(y_pred, dim=1)
        y = batch["labels"]
        # determine candidate sentiment and overall sentiment orientation
        s_c_vader, s_c_bert, s_mean_vader, s_mean_bert = self.sentiment_evaluation_helper(batch)
        # compute metrics
        self.val_performance_metrics(y_pred, y)
        self.val_sentiment_diversity_metrics_vader(y_pred.flatten(), s_c_vader, s_mean_vader)
        self.val_sentiment_diversity_metrics_bert(y_pred.flatten(), s_c_bert, s_mean_bert)
        # log metric
        self.log_dict(self.val_performance_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.val_sentiment_diversity_metrics_vader, on_step=True, on_epoch=True)
        self.log_dict(self.val_sentiment_diversity_metrics_bert, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_pred, _ = self(batch)
        # y_pred = torch.rand(y_pred.size(), device=self.device)
        y_pred = F.softmax(y_pred, dim=1)
        y = batch["labels"]
        # determine candidate sentiment and overall sentiment orientation
        s_c_vader, s_c_bert, s_mean_vader, s_mean_bert = self.sentiment_evaluation_helper(batch)
        # topical diversity
        cat_u, cat_c =  self.topical_diversity_helper(batch)
        # compute metrics
        self.test_performance_metrics(y_pred, y)
        self.test_sentiment_diversity_metrics_vader(y_pred.flatten(), s_c_vader, s_mean_vader)
        self.test_sentiment_diversity_metrics_bert(y_pred.flatten(), s_c_bert, s_mean_bert)
        self.test_topic_diversity_metrics(y_pred.flatten(), cat_c, cat_u)
        self.test_ils_senti_metrics_vader(y_pred.flatten(), s_c_vader)
        self.test_ils_senti_metrics_bert(y_pred.flatten(), s_c_bert)
        self.test_ils_topic_metrics(y_pred.flatten(), cat_c)
        # log metric
        self.log_dict(self.test_performance_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.test_sentiment_diversity_metrics_vader, on_step=True, on_epoch=True)
        self.log_dict(self.test_sentiment_diversity_metrics_bert, on_step=True, on_epoch=True)
        self.log_dict(self.test_topic_diversity_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.test_ils_senti_metrics_vader, on_step=True, on_epoch=True)
        self.log_dict(self.test_ils_senti_metrics_bert, on_step=True, on_epoch=True)
        self.log_dict(self.test_ils_topic_metrics, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.config.learning_rate)

    def sentiment_evaluation_helper(self, batch):
        # sentiment scores of candidate news
        # (determined through sentiment classifier)
        s_c_vader = batch["c_vader_sentiment"].flatten()
        s_c_bert = batch["c_bert_sentiment"].flatten()
        # calc mean sentiment score from browsed news
        # (using sentiment classifier
        s_clicked_vader = batch["h_vader_sentiment"].flatten()
        s_clicked_bert = batch["h_bert_sentiment"].flatten()
        s_mean_vader = s_clicked_vader.mean()
        s_mean_bert = s_clicked_bert.mean()

        return s_c_vader, s_c_bert, s_mean_vader, s_mean_bert

    def topical_diversity_helper(self, batch):
        #  user vector
        cat_u  = torch.zeros((batch["h_category"].size(1), self.config.num_categories), dtype=torch.float, device=self.device)
        cat_u[torch.arange(cat_u.size(0)), batch["h_category"]] = 1.
        cat_u[torch.arange(cat_u.size(0)), batch["h_subcategory"]] = 1.
        cat_u = cat_u[:,1:]
        cat_u = F.normalize(torch.sum(cat_u, 0), dim=-1)
        # candidate vectors
        cat_c  = torch.zeros((batch["c_category"].size(1), self.config.num_categories), dtype=torch.float, device=self.device)
        cat_c[torch.arange(cat_c.size(0)), batch["c_category"]] = 1.
        cat_c[torch.arange(cat_c.size(0)), batch["c_subcategory"]] = 1.
        cat_c = cat_c[:,1:]
        return cat_u, cat_c
