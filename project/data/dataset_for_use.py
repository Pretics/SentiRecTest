from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class BaseDataset(Dataset):
    def __init__(self, behavior_path, news_path, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self.behaviors_parsed = []
        news_parsed = {}
        self.run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        # loading and preparing news collection
        #
        with open(news_path, 'r') as file:
            news_collection = file.readlines()
            for news in tqdm(news_collection):
                nid, cat, subcat, title, abstract, vader_sent, bert_sent = news.split("\t")
                news_parsed[nid] = {
                    'category': torch.tensor(int(cat), device=self.run_device),
                    'subcategory': torch.tensor((int(subcat)), device=self.run_device),
                    'title': torch.tensor([int(i) for i in title.split(" ")], device=self.run_device), 
                    'abstract': torch.tensor([int(i) for i in abstract.split(" ")], device=self.run_device),
                    'vader_sentiment': torch.tensor(float(vader_sent), device=self.run_device),
                    'bert_sentiment': torch.tensor(float(bert_sent), device=self.run_device)
                    }
        #
        # loading and preparing behaviors
        #
        # padding for news
        padding = {
            'category': torch.tensor(0, device=self.run_device),
            'subcategory': torch.tensor(0, device=self.run_device),
            'title': torch.tensor([0] * config.num_words_title, device=self.run_device),
            'abstract': torch.tensor([0] * config.num_words_abstract, device=self.run_device),
            'vader_sentiment': torch.tensor(0.0, device=self.run_device), 
            'bert_sentiment': torch.tensor(0.0, device=self.run_device)
        }

        with open(behavior_path, 'r') as file:
            behaviors = file.readlines()
            for behavior in tqdm(behaviors):
                uid, hist, candidates, clicks = behavior.split("\t")
                user = torch.tensor(int(uid))
                if hist:
                    history = [news_parsed[i] for i in hist.split(" ")]
                    if len(history) > config.max_history: 
                        history = history[:config.max_history]
                    else:
                        repeat = config.max_history - len(history)
                        history = [padding]*repeat + history
                else:
                    history = [padding]*config.max_history
                candidates = [news_parsed[i] for i in candidates.split(" ")]
                labels = torch.tensor([int(i) for i in clicks.split(" ")])
                self.behaviors_parsed.append(
                    {
                        'user': user,
                        'h_title': torch.stack([h['title'] for h in history]).to(self.run_device),
                        'h_abstract': torch.stack([h['abstract'] for h in history]).to(self.run_device),
                        'h_category': torch.stack([h['category'] for h in history]).to(self.run_device),
                        'h_subcategory': torch.stack([h['subcategory'] for h in history]).to(self.run_device),
                        'h_vader_sentiment': torch.stack([h['vader_sentiment'] for h in history]).to(self.run_device),
                        'h_bert_sentiment': torch.stack([h['bert_sentiment'] for h in history]).to(self.run_device),
                        'history_length': torch.tensor(len(history)).to(self.run_device),
                        'c_title': torch.stack([c['title'] for c in candidates]).to(self.run_device),
                        'c_abstract': torch.stack([c['abstract'] for c in candidates]).to(self.run_device),
                        'c_category': torch.stack([c['category'] for c in candidates]).to(self.run_device),
                        'c_subcategory': torch.stack([c['subcategory'] for c in candidates]).to(self.run_device),
                        'c_vader_sentiment': torch.stack([c['vader_sentiment'] for c in candidates]).to(self.run_device),
                        'c_bert_sentiment': torch.stack([c['bert_sentiment'] for c in candidates]).to(self.run_device),
                        'labels': labels
                    }
                )

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        return self.behaviors_parsed[idx]
