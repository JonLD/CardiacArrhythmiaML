import pytorch_lightning as pl
import math
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ecg_model import MLECG
from ecg_data_module import ECGDataModule
from ray import tune

def train_ecg(config, data_dir=None, num_epochs=10, num_gpus=0):
    model = MLECG(config)
    dm = ECGDataModule(
        data_dir=data_dir, num_workers=1, batch_size=config["batch_size"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)
    
config = {
 "lstm_size": tune.choice([1, 2, 3, 4, 5]),
 "lr": tune.loguniform(1e-4, 1e-1),
 "batch_size": tune.choice([32, 64, 128, 256, 512])
}

num_samples = 10
num_epochs = 1
gpus_per_trial = 1 # set this to higher if using GPU

trainable = tune.with_parameters(
    train_ecg,
    data_dir=None,
    num_epochs=num_epochs,
    num_gpus=gpus_per_trial)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": gpus_per_trial
    },
    metric="loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    name="tune_mnist")

print(analysis.best_config)