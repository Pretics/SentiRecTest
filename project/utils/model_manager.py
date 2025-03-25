from os import path
from typing import Union

from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule

from data.dataset import BaseDataset
from utils.base_manager import BaseManager, ManagerArgs
from utils.configs import BaseConfig


class ModelManager(BaseManager):
    current_args: ManagerArgs
    args_list: list[ManagerArgs]
    is_multiple_args: bool
    config: BaseConfig
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

    def __init__(self, args:Union[list[ManagerArgs], ManagerArgs, None], mode: str):
        if args is None:
            return
        elif isinstance(args, ManagerArgs):
            self.update_by_args(args, mode)
        elif type(args) == list:
            self.args_list = args
            self.update_by_args(args[0], mode)

    def update(self):
        self.update_by_args(self.current_args, self.mode)

    def update_by_args(self, args: ManagerArgs, mode: str):
        # configs
        assert(mode in ["train", "test"])
        self.mode = mode
        self.current_args = args
        self.config = self.load_model_config(args)
        # load common data
        self.pretrained_word_embedding = self.load_embedding_weights(self.config)

        if self.mode == "train":
            # load data
            self.train_dataset, self.train_loader = self.create_train_dataloader(self.config)
            self.val_dataset, self.val_loader = self.create_val_dataloader(self.config)
            # init model
            if args.resume_ckpt_path is None:
                self.model = self.create_model(self.config, self.pretrained_word_embedding)
            else:
                self.model = self.load_model_from_checkpoint(args.resume_ckpt_path, self.config, self.pretrained_word_embedding)
            # init trainer
            self.trainer, self.checkpoint_callback = self.create_train_trainer(self.config)
        elif self.mode == "test":
            # load data
            self.test_dataset, self.test_loader = self.create_test_dataloader(self.config)
            # init model
            self.model = self.load_model_from_checkpoint(args.test_ckpt_path, self.config, self.pretrained_word_embedding)
            # init trainer
            self.trainer = self.create_test_trainer(self.config)

    def change_to_train(self, resume: str):
        self.current_args.resume_ckpt_path = resume
        self.mode = "train"
        self.update()

    def change_to_test(self, ckpt_filename):
        self.current_args.test_ckpt_path = path.join(self.config.checkpoint.dirpath, ckpt_filename)
        self.mode = "test"
        self.update()
    
    def fit(self):
        self.trainer.fit(
            model=self.model, 
            train_dataloaders=self.train_loader, 
            val_dataloaders=self.val_loader,
            ckpt_path=self.current_args.resume_ckpt_path
        )

    def fit_all(self):
        if self.is_multiple_args == False:
            print("args.config를 1개만 입력했을 경우 .fit() 함수를 사용해주세요.")
            return
        
        for args in self.args_list:
            self.update_by_args()

    def test(self):
        self.test_result = self.start_test(self.trainer, self.model, self.test_loader)
        return self.test_result