from os import path
from typing import Union
import itertools

from torch import Tensor, device, cuda
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule

from data.dataset import BaseDataset
from utils.base_manager import BaseManager, ManagerArgs
from utils.configs import ModelConfig
from utils.news_viewer import NewsViewer

import pandas as pd

class ModelManager(BaseManager):
    """
    BaseManager를 편하게 사용할 수 있도록 만든 클래스입니다.
    """
    project_dir: str
    current_args: ManagerArgs
    args_list: list[ManagerArgs]
    is_multiple_args: bool
    config: ModelConfig
    train_dataset: BaseDataset
    train_loader: DataLoader
    val_dataset: BaseDataset
    val_loader: DataLoader
    test_dataset: BaseDataset
    test_loader: DataLoader
    pretrained_word_embedding: Tensor
    model: LightningModule
    checkpoint_callback: ModelCheckpoint
    trainer: Trainer
    run_device: device
    news_viewer: NewsViewer

    def __init__(
        self,
        project_dir: str,
        args: Union[list[ManagerArgs], ManagerArgs, None],
        mode: str = "train",
        prepare_immediately: bool = True
    ):
        self.project_dir = project_dir
        self.run_device = device("cuda" if cuda.is_available() else "cpu")
        self.news_viewer = NewsViewer(
            path.join(project_dir, "data", "MIND", "demo", "test", "news.tsv"),
            path.join(project_dir, "data", "preprocessed_data", "demo", "test", "news2int.tsv")
        )
        self.change_args(args, mode, prepare_immediately)

    def change_args(
        self,
        args:Union[list[ManagerArgs], ManagerArgs, None],
        mode: str = "train",
        prepare_immediately:bool = True
    ):
        if args is None:
            return
        # single args
        elif isinstance(args, ManagerArgs):
            self.is_multiple_args = False
            if prepare_immediately:
                self.update_by_args(args, mode)
        # multiple args
        elif type(args) == list:
            self.args_list = args
            self.is_multiple_args = True
            if prepare_immediately:
                self.update_by_args(args[0], mode)

    def prepare_config_paths(self, config: ModelConfig):
        config.project_dir = self.project_dir
        config.checkpoint.dirpath = path.join(self.project_dir, config.checkpoint.dirpath)
        config.logger.save_dir = path.join(self.project_dir, config.logger.save_dir)
    
    def update_by_args(self, args: ManagerArgs, mode: str):
        # configs
        assert(mode in ["train", "test"])
        self.mode = mode
        self.current_args = args
        self.config = self.load_model_config(args)
        self.prepare_config_paths(self.config)
        # load common data
        self.pretrained_word_embedding = self.load_embedding_weights(self.config)

        if self.mode == "train":
            # load data
            self.train_dataset, self.train_loader = self.create_train_dataloader(self.config)
            self.val_dataset, self.val_loader = self.create_val_dataloader(self.config)
            # init model
            if args.resume_ckpt_path is None:
                self.model = self.create_model(self.config, self.pretrained_word_embedding, self.run_device)
            else:
                self.model = self.load_model_from_checkpoint(args.resume_ckpt_path, self.config, self.pretrained_word_embedding, self.run_device)
            # init trainer
            self.trainer, self.checkpoint_callback = self.create_train_trainer(self.config)
        elif self.mode == "test":
            # load data
            self.test_dataset, self.test_loader = self.create_test_dataloader(self.config)
            # init model
            if args.test_ckpt_path is None:
                self.model = self.create_model(self.config, self.pretrained_word_embedding, self.run_device)
            else:
                self.model = self.load_model_from_checkpoint(args.test_ckpt_path, self.config, self.pretrained_word_embedding, self.run_device)
            self.model.eval()
            # init trainer
            self.trainer = self.create_test_trainer(self.config)

    def change_to_train(self, resume_ckpt_path: str = None):
        """
        `resume_ckpt_path`의 `ckpt` 파일은 `self.current_args.config`파일로 생성된 모델이어야 합니다.<br/>
        `self.current_args`를 바꾸려면 `change_args(args, mode)` 를 사용해주세요.
        """
        self.current_args.resume_ckpt_path = resume_ckpt_path
        self.update_by_args(self.current_args, "train")

    def change_to_test(self, test_ckpt_path):
        """
        `test_ckpt_path`의 `ckpt` 파일은 `self.current_args.config`파일로 생성된 모델이어야 합니다.<br/>
        `self.current_args`를 바꾸려면 `change_args(args, mode)` 를 사용해주세요.
        """
        self.current_args.test_ckpt_path = test_ckpt_path
        self.update_by_args(self.current_args, "test")
    
    def fit(self):
        """
        ModelManager에 설정한 멤버 변수로 학습을 진행합니다.
        학습 완료 후 가장 평가가 좋은 ckpt경로를 반환합니다.
        """
        self.trainer.fit(
            model=self.model, 
            train_dataloaders=self.train_loader, 
            val_dataloaders=self.val_loader,
            ckpt_path=self.current_args.resume_ckpt_path
        )
        return self.checkpoint_callback.best_model_path
    
    def fit_all(self):
        """
        다수의 args를 통해 여러 모델의 학습을 순차적으로 진행합니다.
        """
        if self.is_multiple_args == False:
            print("[Warning] args.config를 1개만 입력했을 경우 .fit() 함수를 사용해주세요.")
            return
        
        best_ckpt_paths = []
        for args in self.args_list:
            self.update_by_args(args, "train")
            ckpt_path = self.fit()
            best_ckpt_paths.append(ckpt_path)
        return best_ckpt_paths
    
    def test(self):
        test_result = self.start_test(self.trainer, self.model, self.test_loader)
        exp_name = f"{self.trainer.logger.name}#{self.trainer.logger.version}"
        return { exp_name: test_result[0] }
    
    def test_all(self, ckpt_paths):
        """
        다수의 args와 ckpt를 통해 여러 모델의 학습을 순차적으로 진행합니다.<br/>
        ckpt_paths는 동일한 인덱스의 args.test_ckpt_path로 자동 매핑됩니다. 
        """
        test_results = {}
        for index, args in enumerate(self.args_list):
            args.test_ckpt_path = ckpt_paths[index]
            self.update_by_args(args, "test")
            test_result = self.test()
            test_results.update(test_result)
        return test_results

    def train_test_all(self):
        """
        다수의 args로 모델 생성, 학습, 평가 과정을 자동으로 진행합니다.
        """
        ckpt_paths = self.fit_all()
        test_results = self.test_all(ckpt_paths)
        return test_results
    
    def get_batch_from_dataloader(self, index: int):
        """
        DataLoader는 torch.utils.data.Dataset 클래스를 상속받아 정의된 데이터셋의 인스턴스를 받아,
        데이터를 설정한 batch size에 맞게 묶어준 뒤 iterator의 형태로 하나씩 뽑아쓸 수 있게 만들어져 있습니다.<br/>
        따라서 iter(dataloader)로 batch data를 하나씩 뽑아볼 수 있는 iterator를 생성하고,
        itertools로 index번째 데이터만 잘라내서 next()로 값을 뽑아내어 반환합니다.
        """
        iterator = iter(self.test_loader)
        item: dict = next(itertools.islice(iterator, index, index + 1))
        return item

    def get_model_output_by_batch(self, batch_data: dict):
        result: Tensor = self.model(batch_data) # model.forward(batch_data) 와 동일하게 동작합니다.
        return result
    
    def get_sorted_output(self, batch_data: dict) -> list[dict[str, int]]:
        result: Tensor = self.get_model_output_by_batch(batch_data)
        click_scores = result.tolist()[0]
        ranks = []
        labels = batch_data["labels"].tolist()[0]
        """
        해당시점에서 result는
        [index 0의 score, index 2의 score, ...] 이런 데이터 형태입니다.
        해당 인덱스가 가리키는 impression 뉴스의 label은 labels의 동일한 index위치에 저장되어 있습니다.
        """
        for index, label in enumerate(labels):
            ranks.append([label, click_scores[index], index])
        ranks.sort(key=lambda x: x[1], reverse=True)

        sorted_result = []
        # 각 row 출력
        for rank, data in enumerate(ranks):
            sorted_result.append({
                'rank': rank+1,
                'label': data[0],
                'score': data[1],
                'index': data[2]
            })
        return sorted_result
    
    @staticmethod
    def show_output(sorted_result):
        # 헤더 출력
        print(f"{'Rank':<5} {'Score':^10} {'Label':^6} {'index':^6}")
        print("-" * 32)
        # 각 row 출력
        for data in sorted_result:
            rank = data['rank']
            label = data['label']
            score = data['score']
            index = data['index']
            print(f"{rank:<5} {score:^10.5f} {label:^6} {index:^6}")

    def show_output_by_batch(self, batch_data: dict) -> list[dict[str, int]]:
        sorted_result = self.get_sorted_output(batch_data)
        ModelManager.show_output(sorted_result)
        return sorted_result
    
    def show_output_by_index(self, index):
        batch_data = self.get_batch_from_dataloader(index)
        sorted_result = self.show_output_by_batch(batch_data)
        return sorted_result
    
    @staticmethod
    def show_batch_struct(batch_data: dict):
        print(type(batch_data))
        print("{")
        for key in list(batch_data.keys()):
            value: Tensor = batch_data[key]
            items = value.tolist()
            inner_type = type(items[0]).__name__
            if inner_type == "list":
                inner_type = f"list[{type(items[0][0]).__name__}]"
                if inner_type == "list[list]":
                    inner_type = f"list[list[{type(items[0][0][0]).__name__}]]"
            print(f"\t{key}:\ttype={type(value).__name__}, shape={tuple(value.shape)}, inner_type={inner_type}", end="")
            if inner_type == int:
                print(f", value:{items[0]}")
            else:
                print("")
        print("}")

    def get_word2int(self):
        word2int_path = path.join(
            self.config.project_dir, "data", "preprocessed_data", self.config.dataset_size, self.config.word2int
        )
        word2int = pd.read_csv(word2int_path, sep='\t', header=None, names=['word', 'word_index'], encoding='utf-8')
        return word2int
    
    def show_history(self, batch_index, sample_num):
        sample_num = max(1, sample_num)
        batch_data = self.get_batch_from_dataloader(batch_index)
        news_idxs = batch_data['h_idxs'][0].tolist()
        print("==================================================================")
        print(f" {sample_num} Samples of History (User: {batch_data['user'].item()}) ")
        print("==================================================================")
        count = 0
        print("------------------------------------------------------------------")
        for news_idx in news_idxs:
            if news_idx == 0:
                continue
            print(f"[ Sample {count+1} ]")
            self.news_viewer.show_news_by_index(news_idx)
            print("------------------------------------------------------------------")
            count += 1
            if count >= sample_num:
                break

    def show_topN_result(self, batch_index, topN):
        topN = max(1, topN)
        batch_data = self.get_batch_from_dataloader(batch_index)
        news_idxs = batch_data['c_idxs'][0]
        result = self.get_sorted_output(batch_data)
        print("==================================================================")
        print(f" Top {topN} Impressions Ranked by Model (User: {batch_data['user'].item()}) ")
        print("==================================================================")
        print("------------------------------------------------------------------")
        for ranking_data in result[:min(topN, len(result))]:
            rank = ranking_data['rank']
            label = ranking_data['label']
            score = ranking_data['score']
            index = ranking_data['index']
            news_idx = news_idxs[index].item()
            
            print(f"[ rank: {rank}, score: {score:>.5f}, label: {label} ]")
            self.news_viewer.show_news_by_index(news_idx)
            print("------------------------------------------------------------------")