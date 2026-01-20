import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset


class MultitaskUnivariateRegressionDataset(Dataset):
    """
    Multitask Univariate Dataset.
    Splits each sample into univariate X and Y feature pairs for univariate regression tasks.
    """ 
    def __init__(self, C, X, Y, x_feature_embeddings=None, y_feature_embeddings=None, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        # Use x_feature_embeddings if provided, else use identity matrix for one-hot encoding
        if x_feature_embeddings is not None:
            self.x_feature_embeddings = torch.tensor(x_feature_embeddings, dtype=dtype)
        else:
            self.x_feature_embeddings = torch.eye(self.x_dim, dtype=dtype)
        # Use y_feature_embeddings if provided, else use identity matrix for one-hot encoding
        if y_feature_embeddings is not None:
            self.y_feature_embeddings = torch.tensor(y_feature_embeddings, dtype=dtype)
        else:
            self.y_feature_embeddings = torch.eye(self.y_dim, dtype=dtype)
        self.dtype = dtype
    
    def __len__(self):
        return len(self.C) * self.x_dim * self.y_dim
 
    def __getitem__(self, idx):
        # Get task-split sample indices
        n_i = idx // (self.x_dim * self.y_dim)
        x_i = (idx // self.y_dim) % self.x_dim
        y_i = idx % self.y_dim
        return {
            "idx": idx,
            "contexts": self.C[n_i],  # (c_dim,)
            "predictor_embeddings": self.x_feature_embeddings[x_i],  # (x_embedding_dim,)
            "outcome_embeddings": self.y_feature_embeddings[y_i],  # (y_embedding_dim,)
            "predictors": self.X[n_i, x_i].unsqueeze(0),  # (1, )
            "outcomes": self.Y[n_i, y_i].unsqueeze(0),  # (1, )
            "sample_idx": n_i,
            "predictor_idx": x_i,
            "outcome_idx": y_i,
        }


class MultitaskCorrelationDataModule(LightningDataModule):
    def __init__(
        self,
        C_train,
        X_train,
        train_feature_embeddings,
        C_val,
        X_val,
        val_feature_embeddings,
        C_test,
        X_test,
        test_feature_embeddings,
        C_predict,
        X_predict,
        predict_feature_embeddings,
        batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.C_train = C_train
        self.X_train = X_train
        self.train_feature_embeddings = train_feature_embeddings
        self.C_val = C_val
        self.X_val = X_val
        self.val_feature_embeddings = val_feature_embeddings
        self.C_test = C_test
        self.X_test = X_test
        self.test_feature_embeddings = test_feature_embeddings
        self.C_predict = C_predict
        self.X_predict = X_predict
        self.predict_feature_embeddings = predict_feature_embeddings

    def setup(self, stage=None):
        # Called on every GPU
        self.train_dataset = MultitaskUnivariateRegressionDataset(
            self.C_train, 
            self.X_train, 
            self.X_train, 
            x_feature_embeddings=self.train_feature_embeddings, 
            y_feature_embeddings=self.train_feature_embeddings
        )
        self.val_dataset = MultitaskUnivariateRegressionDataset(
            self.C_val, 
            self.X_val, 
            self.X_val, 
            x_feature_embeddings=self.val_feature_embeddings, 
            y_feature_embeddings=self.val_feature_embeddings
        )
        self.test_dataset = MultitaskUnivariateRegressionDataset(
            self.C_test,
            self.X_test,
            self.X_test,
            x_feature_embeddings=self.test_feature_embeddings,
            y_feature_embeddings=self.test_feature_embeddings
        )
        self.predict_dataset = MultitaskUnivariateRegressionDataset(
            self.C_predict,
            self.X_predict,
            self.X_predict,
            x_feature_embeddings=self.predict_feature_embeddings,
            y_feature_embeddings=self.predict_feature_embeddings
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
    