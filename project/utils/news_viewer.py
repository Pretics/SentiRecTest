import pandas as pd

class NewsViewer:
    news2int: pd.DataFrame
    news: pd.DataFrame

    def __init__(self, news_path, news2int_path):
        self.news = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            encoding='utf-8'
        )
        self.news = self.news.set_index('news_id')

        self.news2int = pd.read_csv(news2int_path, sep='\t', header=None, names=['news_id', 'news_index'], encoding='utf-8')
        self.news2int = self.news2int.set_index('news_index')

    def show_news_by_index(self, news_index):
        if news_index == 0:
            print("padding")
            return
        news_id = self.news2int.loc[news_index, 'news_id']
        news_data = self.news.loc[news_id]
        print(f" - News ID: {news_id}")
        print(f" - Category: {news_data['category']}")
        print(f" - SubCategory: {news_data['subcategory']}")
        print(f" - Title: {news_data['title']}")
        print(f" - Abstract: {news_data['abstract']}")

    def get_news_by_index(self, news_index):
        if news_index == 0:
            return {}
        news_id = self.news2int.loc[news_index, 'news_id']
        news_data = self.news.loc[news_id]
        return {
            'news_index': news_index,
            'news_id': news_id,
            'category': news_data['category'],
            'subcategory': news_data['subcategory'],
            'title': news_data['category'],
            'abstract': news_data['abstract']
        }