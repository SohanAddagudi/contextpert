import torch
import lightning as pl


class MultitaskContextualizedUnivariateRegression(pl.LightningModule):
    """Multitask Contextualized Univariate Regression Model

    Args
        encoder: The encoder model to process contexts. 
            Output must be exactly 2: beta and mu.

    """
    def __init__(
        self,
        encoder,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
    
    def configure_optimizers(self):
        """
        Set up optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
        outputs = self.encoder(
            batch["contexts"],
            batch["predictor_embeddings"],
            batch["outcome_embeddings"],
        )
        return outputs
    
    def _batch_loss(self, batch):
        outputs = self(batch)
        beta = outputs['betas']
        mu = outputs['mus']
        x_true = batch['predictors']
        y_true = batch['outcomes']
        y_pred = x_true * beta + mu
        loss = self.loss_fn(y_true, y_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._batch_loss(batch)
        self.log_dict({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._batch_loss(batch)
        self.log_dict({"test_loss": loss})
        return loss
    
    def predict_step(self, batch, batch_idx):
        outputs = self(batch)
        outputs.update({
            "sample_idx": batch["sample_idx"],
            "predictor_idx": batch["predictor_idx"],
            "outcome_idx": batch["outcome_idx"],
        })
        return outputs
 