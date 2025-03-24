from os import path
from dataclasses import dataclass
from typing import Union
from tqdm import tqdm
from dotmap import DotMap

import numpy as np
import yaml

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

@dataclass
class ModelArgs():
    """
    ``config``: config파일 경로 <br/>
    ``resume``: 학습을 재개할 ckpt파일 경로 <br/>
    ``ckpt``: 테스트를 진행할 모델의 ckpt파일 경로
    """
    config: str                 # all, require
    resume: Union[str, None]    # for train
    ckpt: Union[str, None]      # for test

class ModelManager:
    args: ModelArgs
    config: DotMap
    train_dataset: BaseDataset
    train_loader: DataLoader
    val_dataset: BaseDataset
    val_loader: DataLoader
    test_dataset: BaseDataset
    test_loader: DataLoader
    pretrained_word_embedding: Tensor
    model: LightningModule
    trainer: Trainer

    def __init__(self, args:Union[ModelArgs, None], mode: str):
        if args is not None:
            self.update_by_args(args, mode)

    def update_all(self):
        self.update_by_args(self.args, self.mode)

    def update_by_args(self, args: ModelArgs, mode: str):
        # configs
        assert(mode in ["train", "test"])
        self.mode = mode
        self.args = args
        self.config = self.load_model_config(args)
        # load common data
        self.pretrained_word_embedding = self.load_embedding_weights(self.config)

        if self.mode == "train":
            # load data
            self.train_dataset, self.train_loader = self.create_train_dataloader(self.config)
            self.val_dataset, self.val_loader = self.create_val_dataloader(self.config)
            # init model, trainer
            self.model = self.create_model(self.config, self.pretrained_word_embedding)
            self.trainer = self.create_train_trainer(self.config)
        elif self.mode == "test":
            # load data
            self.test_dataset, self.test_loader = self.create_test_dataloader(self.config)
            # init model, trainer
            self.model = self.load_model_from_checkpoint(args, self.config, self.pretrained_word_embedding)
            self.trainer = self.create_test_trainer(self.config)

    def change_to_train(self, resume: str):
        self.args.resume = resume
        self.mode = "train"
        self.update_all()

    def change_to_test(self, ckpt_filename):
        self.args.ckpt = path.join(self.config.checkpoint.dirpath, ckpt_filename)
        self.mode = "test"
        self.update_all()

    def load_model_config(self, args: ModelArgs):
        with open(args.config, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            config = DotMap(config)
        assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])
        seed_everything(config.seed)
        return config

    def create_callbacks(self, config: DotMap):
        checkpoint_callback = ModelCheckpoint(
            **config.checkpoint
        )
        early_stop_callback = EarlyStopping(
            **config.early_stop
        )
        return [checkpoint_callback, early_stop_callback]

    def create_logger(self, config: DotMap):
        logger = TensorBoardLogger(**config.logger)
        return logger

    def create_dataloader(self, config, behavior_path, news_path, config_loader):
        dataset = BaseDataset(behavior_path, news_path, config)
        loader = DataLoader(
            dataset,
            **config_loader)
        return dataset, loader

    def create_train_dataloader(self, config):
        train_dataset, train_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.train_behavior),
            path.join(config.preprocess_data_dir, config.train_news),
            config.train_dataloader
        )
        return train_dataset, train_loader

    def create_val_dataloader(self, config):
        val_dataset, val_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.val_behavior),
            path.join(config.preprocess_data_dir, config.train_news),
            config.val_dataloader
        )
        return val_dataset, val_loader
    
    def create_test_dataloader(self, config: DotMap):
        test_dataset, test_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.test_behavior),
            path.join(config.preprocess_data_dir, config.test_news),
            config.test_dataloader
        )
        return test_dataset, test_loader

    def load_embedding_weights(self, config: DotMap):
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

    # from train_manager
    def create_model(self, config: DotMap, pretrained_word_embedding: Tensor):
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
    
    # from test_manager
    def load_model_from_checkpoint(self, args: ModelArgs, config: DotMap, pretrained_word_embedding: Tensor):
        if config.name == "lstur":
            model = LSTUR.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "nrms":
            model = NRMS.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "naml":
            model = NAML.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "naml_simple":
            model = NAML_Simple.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "sentirec":
            model = SENTIREC.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        elif config.name == "robust_sentirec":
            model = ROBUST_SENTIREC.load_from_checkpoint(
                args.ckpt, 
                config=config, 
                pretrained_word_embedding=pretrained_word_embedding
            )
        return model

    def create_trainer(
            self, 
            config: DotMap,
            callbacks: Union[list[Callback], Callback, None],
            logger: TensorBoardLogger
        ):
        trainer = Trainer(
            **config.trainer,
            callbacks=callbacks,
            logger=logger
        )
        return trainer

    def create_train_trainer(self, config: DotMap):
        # init callbacks & logging
        callbacks = self.create_callbacks(config)
        logger = self.create_logger(config)
        # init trainer
        trainer = self.create_trainer(
            config=config,
            callbacks=callbacks,
            logger=logger
        )
        return trainer
    
    def create_test_trainer(self, config: DotMap):
        # init logging
        logger = self.create_logger(config)
        # init trainer
        trainer = self.create_trainer(
            config=config,
            callbacks=None,
            logger=logger
        )
        return trainer

    def start_train(
            self, 
            args: ModelArgs,
            model: LightningModule,
            trainer: Trainer,
            train_loader: DataLoader,
            val_loader: DataLoader
        ):
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=args.resume
        )
    
    def start_test(
            self, 
            trainer: Trainer,
            model: LightningModule,
            test_loader: DataLoader
        ) -> list[dict[str, float]]:
        test_result = trainer.test(
            model=model, 
            dataloaders=test_loader
        )
        return test_result
    
    def fit(self):
        self.trainer.fit(
            model=self.model, 
            train_dataloaders=self.train_loader, 
            val_dataloaders=self.val_loader,
            ckpt_path=self.args.resume
        )

    def test(self):
        self.test_result = self.start_test(self.trainer, self.model, self.test_loader)
        return self.test_result