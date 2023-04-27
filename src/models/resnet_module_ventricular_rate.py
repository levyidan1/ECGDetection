from typing import Any, List

import torch
import torchvision
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.classification.matthews_corrcoef import MulticlassMatthewsCorrCoef
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.regression.mae import MeanAbsoluteError


class ResNetVentricularRate(LightningModule):
    """LightningModule for ECG ventricular rate prediction that uses a ResNet.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = torchvision.models.resnet18(weights=None)
        linear_size = list(self.net.children())[-1].in_features        
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(linear_size, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, self.hparams.num_classes),
            torch.nn.Softmax(dim=1),
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = MeanAbsoluteError()
        self.val_acc = MeanAbsoluteError()
        self.test_acc = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for averaging recall, precision and f1 across batches
        self.train_recall = MeanMetric()
        self.val_recall = MeanMetric()
        self.test_recall = MeanMetric()

        self.train_precision = MeanMetric()
        self.val_precision = MeanMetric()
        self.test_precision = MeanMetric()

        self.train_f1 = MeanMetric()
        self.val_f1 = MeanMetric()
        self.test_f1 = MeanMetric()

        # for averaging auroc across batches
        self.train_auroc = MeanMetric()
        self.val_auroc = MeanMetric()
        self.test_auroc = MeanMetric()

        # for averaging stat scores across batches
        self.train_stat_scores = MeanMetric()
        self.val_stat_scores = MeanMetric()
        self.test_stat_scores = MeanMetric()

        # for averaging matthews correlation coefficient across batches
        self.train_mcc = MeanMetric()
        self.val_mcc = MeanMetric()
        self.test_mcc = MeanMetric()
        # for averaging confusion matrix across batches
        self.train_confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes, task="multiclass")
        self.val_confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes, task="multiclass")
        self.test_confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes, task="multiclass")

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        y = y.long()
        # modify y_loss to map ventricular rate to its respective index
        y_loss = torch.nn.functional.one_hot(y, num_classes=self.hparams.num_classes).float().squeeze(1)
        loss = self.criterion(logits, y_loss)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y.view(-1)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_recall(preds, targets)
        self.train_precision(preds, targets)
        self.train_f1(preds, targets)
        self.train_auroc(preds, targets)
        # self.train_stat_scores(preds, targets)
        self.train_mcc(preds, targets)
        # self.train_confusion_matrix(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", self.train_auroc(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/stat_scores", self.train_stat_scores(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mcc", self.train_mcc(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/confusion_matrix", self.train_confusion_matrix(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_recall(preds, targets)
        self.val_precision(preds, targets)
        self.val_f1(preds, targets)
        self.val_auroc(preds, targets)
        # self.val_stat_scores(preds, targets)
        self.val_mcc(preds, targets)
        # self.val_confusion_matrix(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/stat_scores", self.val_stat_scores(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mcc", self.val_mcc(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/confusion_matrix", self.val_confusion_matrix(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `validation_step()`
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_recall(preds, targets)
        self.test_precision(preds, targets)
        self.test_f1(preds, targets)
        self.test_auroc(preds, targets)
        # self.test_stat_scores(preds, targets)
        self.test_mcc(preds, targets)
        # self.test_confusion_matrix(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/stat_scores", self.test_stat_scores, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mcc", self.test_mcc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/confusion_matrix", self.test_confusion_matrix, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
