from os import path
from dataclasses import dataclass, asdict
from typing import Union

from tqdm import tqdm
import numpy as np

from torch import Tensor, from_numpy
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule, Callback, seed_everything

from models.lstur import LSTUR
from models.nrms import NRMS
from models.naml import NAML
from models.naml_simple import NAML_Simple
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC

from data.dataset import BaseDataset
from utils.configs import BaseConfig, load_config_from_yaml


@dataclass
class ManagerArgs:
    """
    ``config``: config파일 경로 <br/>
    ``resume_ckpt_path``: 학습을 이어서 할 ckpt파일 경로 <br/>
    ``test_ckpt_path``: 테스트를 진행할 모델의 ckpt파일 경로
    """
    config_path: str                 # all, require
    resume_ckpt_path: Union[str, None] = None    # for train
    test_ckpt_path: Union[str, None] = None      # for test


class BaseManager:
    """
    가독성을 위해 ModelManager의 내부적인 동작 코드는 모두 이곳에 몰아 넣었습니다.
    """
    # ===============
    # for load data
    # ===============
    @staticmethod
    def load_model_config(args: ManagerArgs):
        #with open(args.config_path, 'r') as ymlfile:
        #    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config: BaseConfig = load_config_from_yaml(args.config_path, BaseConfig)
        assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])
        seed_everything(config.seed)
        return config
    
    @staticmethod
    def load_embedding_weights(config: BaseConfig):
        # load embedding pre-trained embedding weights
        embedding_weights=[]
        with open(path.join(config.preprocess_data_dir, config.embedding_weights), 'r') as file: 
            lines = file.readlines()
            for line in tqdm(lines):
                weights = [float(w) for w in line.split(" ")]
                embedding_weights.append(weights)
        pretrained_word_embedding = from_numpy(
            np.array(embedding_weights, dtype=np.float32)
        )
        return pretrained_word_embedding
    
    @staticmethod
    def create_dataloader(config: BaseConfig, behavior_path, news_path, config_loader):
        dataset = BaseDataset(behavior_path, news_path, config)
        loader = DataLoader(
            dataset,
            **asdict(config_loader))
        return dataset, loader

    def create_train_dataloader(self, config: BaseConfig):
        train_dataset, train_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.train_behavior),
            path.join(config.preprocess_data_dir, config.train_news),
            config.train_dataloader
        )
        return train_dataset, train_loader

    def create_val_dataloader(self, config: BaseConfig):
        val_dataset, val_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.val_behavior),
            path.join(config.preprocess_data_dir, config.train_news),
            config.val_dataloader
        )
        return val_dataset, val_loader
    
    def create_test_dataloader(self, config: BaseConfig):
        test_dataset, test_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.test_behavior),
            path.join(config.preprocess_data_dir, config.test_news),
            config.test_dataloader
        )
        return test_dataset, test_loader

    # ===============
    # for init model
    # ===============
    @staticmethod
    def create_model(config: BaseConfig, pretrained_word_embedding: Tensor):
        if config.name == "lstur":
            model = LSTUR(config, pretrained_word_embedding)
        elif config.name == "nrms":
            model = NRMS(config, pretrained_word_embedding)
        elif config.name == "naml":
            model = NAML(config, pretrained_word_embedding)
        elif config.name == "naml_simple":
            model = NAML_Simple(config, pretrained_word_embedding)
        elif config.name == "sentirec":
            model = SENTIREC(config, pretrained_word_embedding)
        elif config.name == "robust_sentirec":
            model = ROBUST_SENTIREC(config, pretrained_word_embedding)
        return model
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, config: BaseConfig, pretrained_word_embedding: Tensor):
        if config.name == "lstur":
            model = LSTUR.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "nrms":
            model = NRMS.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "naml":
            model = NAML.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "naml_simple":
            model = NAML_Simple.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "sentirec":
            model = SENTIREC.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "robust_sentirec":
            model = ROBUST_SENTIREC.load_from_checkpoint(
                checkpoint_path, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        return model

    # =============
    # for Trainer
    # =============
    @staticmethod
    def create_callbacks(config: BaseConfig):
        checkpoint_callback = ModelCheckpoint(
            **asdict(config.checkpoint)
        )
        early_stop_callback = EarlyStopping(
            **asdict(config.early_stop)
        )
        return checkpoint_callback, early_stop_callback

    @staticmethod
    def create_logger(config: BaseConfig):
        logger = TensorBoardLogger(**asdict(config.logger))
        return logger

    @staticmethod
    def create_trainer(
            config: BaseConfig,
            callbacks: Union[list[Callback], Callback, None],
            logger: TensorBoardLogger
        ):
        trainer = Trainer(
            **asdict(config.trainer),
            callbacks=callbacks,
            logger=logger
        )
        return trainer

    def create_train_trainer(self, config: BaseConfig):
        # init callbacks & logging
        checkpoint_callback, early_stop_callback = self.create_callbacks(config)
        logger = self.create_logger(config)
        # init trainer
        trainer = self.create_trainer(
            config=config,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger
        )
        return trainer, checkpoint_callback
    
    def create_test_trainer(self, config: BaseConfig):
        # init logging
        logger = self.create_logger(config)
        # init trainer
        trainer = self.create_trainer(
            config=config,
            callbacks=None,
            logger=logger
        )
        return trainer

    # ==================
    # for Training/Test
    # ==================
    @staticmethod
    def start_train(
            args: ManagerArgs,
            model: LightningModule,
            trainer: Trainer,
            train_loader: DataLoader,
            val_loader: DataLoader
        ):
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=args.resume_ckpt_path
        )
    
    @staticmethod
    def start_test(
            trainer: Trainer,
            model: LightningModule,
            test_loader: DataLoader
        ) -> list[dict[str, float]]:
        test_result = trainer.test(
            model=model, 
            dataloaders=test_loader
        )
        return test_result