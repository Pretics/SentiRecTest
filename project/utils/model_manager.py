from typing import Union

from torch import Tensor, device, cuda
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule

from data.dataset import BaseDataset
from utils.base_manager import BaseManager, ManagerArgs
from utils.configs import BaseConfig

from os import path


class ModelManager(BaseManager):
    """
    BaseManager를 편하게 사용할 수 있도록 만든 클래스입니다.
    """
    project_dir: str
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
    run_device: device

    def __init__(self, project_dir: str, args: Union[list[ManagerArgs], ManagerArgs, None], mode: str = "train"):
        self.project_dir = project_dir
        self.run_device = device("cuda" if cuda.is_available() else "cpu")
        self.change_args(args, mode)

    def change_args(self, args:Union[list[ManagerArgs], ManagerArgs, None], mode: str = "train"):
        if args is None:
            return
        # single args
        elif isinstance(args, ManagerArgs):
            self.update_by_args(args, mode)
        # multiple args
        elif type(args) == list:
            self.args_list = args
            self.is_multiple_args = True
            self.update_by_args(args[0], mode)

    def prepare_config_paths(self, config: BaseConfig):
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
                self.model = self.load_model_from_checkpoint(self.config.project_dir, args.resume_ckpt_path, self.config, self.pretrained_word_embedding, self.run_device)
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
