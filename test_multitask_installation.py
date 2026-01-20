from pathlib import Path
import torch
from lightning import Trainer
import lightning as pl
import numpy as np

from contextpert.cellvsnet.multitask.data import MultitaskCorrelationDataModule
from contextpert.cellvsnet.multitask.model import MultitaskContextualizedUnivariateRegression
from contextpert.cellvsnet.multitask.callbacks import MmapEdgeWriter
from contextpert.cellvsnet.multitask.utils import measure_mses, regression_to_correlation


c_dim = 50
feature_embedding_dim = 8
# Because we represent network feature indices with embeddings
# we can predict and test on different feature sets than we train on
# as long as the embeddings are the same dimension.
train_x_dim = 10
train_feature_embeddings = torch.randn(train_x_dim, feature_embedding_dim)
val_x_dim = 10
val_feature_embeddings = torch.randn(val_x_dim, feature_embedding_dim)
test_x_dim = 4 
test_feature_embeddings = torch.randn(test_x_dim, feature_embedding_dim)
predict_x_dim = 4
predict_feature_embeddings = torch.randn(predict_x_dim, feature_embedding_dim)

# Setup training
data = MultitaskCorrelationDataModule(
    C_train=torch.randn(100, c_dim),
    C_val=torch.randn(50, c_dim),
    C_test=torch.randn(50, c_dim),
    C_predict=torch.randn(50, c_dim),
    X_train=torch.randn(100, train_x_dim),
    X_val=torch.randn(50, val_x_dim),
    X_test=torch.randn(50, test_x_dim),
    X_predict=torch.zeros(50, predict_x_dim),
    train_feature_embeddings=train_feature_embeddings,
    val_feature_embeddings=val_feature_embeddings,
    test_feature_embeddings=test_feature_embeddings,
    predict_feature_embeddings=predict_feature_embeddings,
    batch_size=4,
)

# BYO-Encoder for architecture flexibility.
# Must always have output_dim=2 for (beta, mu)
class MLPEncoder(torch.nn.Module):
    """
    Simple MLP Encoder.
    """
    def __init__(self, context_dim, feature_embedding_dim, hidden_dims, output_dim, activation=torch.nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = context_dim + feature_embedding_dim + feature_embedding_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, contexts, predictor_embeddings, outcome_embeddings):
        x = torch.cat([contexts, predictor_embeddings, outcome_embeddings], dim=-1)
        outputs = self.network(x)
        return {
            'betas': outputs[:, 0],
            'mus': outputs[:, 1],
        }
model = MultitaskContextualizedUnivariateRegression(
    encoder=MLPEncoder(
        context_dim=c_dim,
        feature_embedding_dim=feature_embedding_dim,
        hidden_dims=[64, 32],
        output_dim=2,  # Always 2: beta, mu
    ),
)

# Train
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    filename='best_model',
)
logger = pl.pytorch.loggers.WandbLogger(
    project='contextpert',
    name='cell_line_context',
    log_model=True,
    save_dir='logs/',
)
trainer = Trainer(
    max_epochs=1,
    accelerator='auto',
    devices='auto',
    callbacks=[checkpoint_callback],
    logger=logger,
)
trainer.fit(model, data)
trainer.test(model, data)

# Get test predictions, check losses match
test_output_dir = Path(checkpoint_callback.best_model_path).parent / 'test_predictions'
writer_callback = MmapEdgeWriter(
    mmap_dir=test_output_dir,
    n_samples=data.X_test.shape[0],
    x_dim=test_x_dim,
    y_dim=test_x_dim,
    dtype=np.float32,
    write_interval='batch',
)
trainer = Trainer(
    accelerator='auto',
    devices='auto',
    callbacks=[writer_callback],
)
_ = trainer.predict(model, data.test_dataloader())

preds = np.memmap(
    test_output_dir / "edges_rank0.dat",  # Note: if using multi-device, must compile memory maps from all ranks first
    dtype=np.float32,
    mode="r",
    shape=(data.X_test.shape[0], test_x_dim, test_x_dim, 2)
)
test_betas = preds[:, :, :, 0]
test_mus = preds[:, :, :, 1]
test_mses = measure_mses(test_betas, test_mus, data.X_test.numpy())
print(f"Test MSE: {test_mses.mean()}")

# Finally, get pred predictions, convert to correlation
pred_output_dir = Path(checkpoint_callback.best_model_path).parent / 'predict_predictions'
writer_callback = MmapEdgeWriter(
    mmap_dir=pred_output_dir,
    n_samples=data.X_predict.shape[0],
    x_dim=predict_x_dim,
    y_dim=predict_x_dim,
    dtype=np.float32,
    write_interval='batch',
)
trainer = Trainer(
    accelerator='auto',
    devices='auto',
    callbacks=[writer_callback],
)
_ = trainer.predict(model, data.predict_dataloader())
preds = np.memmap(
    pred_output_dir / "edges_rank0.dat",  # Note: if using multi-device, must compile memory maps from all ranks first
    dtype=np.float32,
    mode="r",
    shape=(data.X_predict.shape[0], predict_x_dim, predict_x_dim, 2)
)
pred_betas = preds[:, :, :, 0]
pred_mus = preds[:, :, :, 1]
pred_correlations = regression_to_correlation(pred_betas)
print(f"Pred correlations shape: {pred_correlations.shape}")