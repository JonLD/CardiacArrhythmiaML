from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ecg_model import MLECG
from ecg_data_module import ECGDataModule
import pytorch_lightning as pl

def train_mnist(config, data_dir=None, num_epochs=10, num_gpus=0):
    model = MLECG(config, 3)
    dm = ECGDataModule(
        data_dir=data_dir, num_workers=1, batch_size=config["batch_size"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)