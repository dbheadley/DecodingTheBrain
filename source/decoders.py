import torch
import numpy as np
from .loaders import EcogFingerData
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

def format_ecogfinger_data(data=None, finger='thumb'):
    """Format ECoGFinger data for decoding.
    
    Parameters
    ----------
    data : EcogFingerData
        Data to format.
    finger : str
        Finger to decode.

    Returns
    -------
    X : array, shape (n_epochs, n_features)
        Features for each epoch.
    y : array, shape (n_epochs,)
        Labels for each epoch.
    """

    # get flexion event times
    flex_events = data.detect_flex_onsets(finger)

    # get movement and null spec epochs, 1 s after each thumb flexion event
    _,_,flexes = data.get_spec(event_ts=flex_events, pre_t=0.2, post_t=0.2, freq_max=200)
    _,_,nulls = data.get_spec(event_ts=flex_events-1, pre_t=0.2, post_t=0.2, freq_max=200)

    # mean power across time for each epoch
    flexes = np.mean(flexes, axis=3)
    nulls = np.mean(nulls, axis=3)

    # z-score each frequency and channel
    _,_,total_data = data.get_spec(freq_max=200)
    z_mean = np.mean(total_data, axis=3).squeeze() # squeeze to remove singleton epoch dimension
    z_std = np.std(total_data, axis=3).squeeze() 
    flexes = (flexes - z_mean) / z_std
    nulls = (nulls - z_mean) / z_std

    # create labels for thumb movements and nulls
    lbls = np.hstack((np.ones(flexes.shape[0]), np.zeros(nulls.shape[0])))[:, np.newaxis]

    # stack flexes and thumb_nulls along first dimension
    feats = np.vstack((flexes, nulls))

    # reformat features so that each trial is a row and each column is a feature
    feats = feats.reshape(feats.shape[0],-1)

    return feats, lbls

class ECoGData(Dataset):
    def __init__(self, ecog_feat, ecog_lbl, transform=None, target_transform=None):
        # Parameters
        # ----------
        # ecog_feat : array-like
        #     Array of features, where each row is a trial and each column is a feature
        # ecog_lbl : array-like
        #     Array of labels, where each row is dummy coded indicator of finger for the corresponding row in ecog_feat
        # transform : callable, optional
        #     Optional transform to be applied to the ecog data
        # target_transform : callable, optional
        #     Optional transform to be applied to the trial label

        self.ecog_feat = ecog_feat
        self.ecog_lbl = ecog_lbl
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # Returns
        # -------
        # len : int
        #     Number of samples in the dataset

        return len(self.ecog_lbl)
    
    def __getitem__(self, idx):
        # Parameters
        # ----------
        # idx : int
        #     Index of the sample to return

        # Returns
        # -------
        # erp : array-like
        #     ECoG data for the selected sample
        # lbl : array-like
        #     Trial label for the selected sample
        
        feat = self.ecog_feat[idx].astype(np.float32) # get the ECoG data for the selected sample
        lbl = self.ecog_lbl[idx] #.astype(np.float32).reshape(-1,1) # get the trial label for the selected sample
        if self.transform is not None: # apply the transform to the ECoG data
            feat = self.transform(feat)
        if self.target_transform is not None: # apply the transform to the trial label
            lbl = self.target_transform(lbl)
        return feat, lbl
    
class LogRegPT():
    def __init__(self, lr=0.01, epochs=100, train_prop=0.8, batch_size=5, lam=0.0):
        # Parameters
        # ----------
        # lr : float, optional
        #     Learning rate for gradient descent
        # epochs : int, optional
        #     Number of epochs to train for
        # train_prop : float, optional
        #     Proportion of data to use for training
        # batch_size : int, optional
        #     Number of samples per batch
        # lam : float, optional
        #     Regularization parameter for L1 norm

        self.lr = lr
        self.epochs = epochs
        self.train_prop = train_prop
        self.batch_size = batch_size
        self.lam = lam
        self._logreg = None
        self.train_idxs = None
        self.test_idxs = None
    
    def _create_logreg(self, input_dim):
        # Parameters
        # ----------
        # input_dim : int
        #     Number of input features

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(input_dim, 1)

        # sigmoid layer is the activation function
        sig_layer = torch.nn.Sigmoid()

        # logistic regression model is a sequential combination of linear and sigmoid layers
        logreg = torch.nn.Sequential(
            lin_layer,
            sig_layer
        )
        self._logreg = logreg
    
    def loss(self, pred, lbl):
        # Parameters
        # ----------
        # pred : array-like
        #     Predicted probability of each sample being in class 1
        # lbl : array-like
        #     Array of labels, where each element is the label for the corresponding row in X

        # Returns
        # -------
        # loss : float
        #     Loss value

        loss_fn = torch.nn.BCELoss(reduction='mean')

        # calculate loss
        loss = loss_fn(pred, lbl)
        loss += self.lam*torch.sum(torch.abs(self._logreg[0].weight)) # add L1 regularization

        return loss

    def _create_optim(self):
        # initialize optimizer
        return torch.optim.SGD(self._logreg.parameters(), lr=self.lr)
    
    def fit(self, X, y):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature
        # y : array-like
        #     Array of labels, where each element is the label for the corresponding row in X

        # Returns
        # -------
        # score_test : float
        #     Balanced accuracy score for testing data
        # score_train : float
        #     Balanced accuracy score for training data

        # initialize model and fitting
        feat_num = X.shape[1]
        self._create_logreg(feat_num)
        optim = self._create_optim()
        
        # split data into train and test sets
        #train_num = int(self.train_prop*X.shape[0])
        strat = StratifiedKFold(n_splits=int(1/(1-self.train_prop)), shuffle=True)
        self.train_idxs, self.test_idxs = strat.split(X, y).__next__()
        #self.train_idxs = np.random.permutation(X.shape[0])[:train_num]
        #self.test_idxs = np.setdiff1d(np.arange(X.shape[0]), self.train_idxs)

        if self.train_idxs.size < self.batch_size:
            raise ValueError('Number of training samples smaller than batch size')
        
        # create data loaders for train and test sets
        train_ds = ECoGData(X[self.train_idxs,:], y[self.train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[self.test_idxs,:], y[self.test_idxs])
        
        # train model
        self._logreg.train()
        for epoch in range(self.epochs):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self._logreg(feat)
                loss = self.loss(pred, lbl.float())
                loss.backward()
                optim.step()
        
        
        # get predictions for train and test sets
        if self.train_prop < 1.0:
            score_test = self.score(test_ds.ecog_feat, test_ds.ecog_lbl)
        else:
            score_test = np.nan

        score_train = self.score(train_ds.ecog_feat, train_ds.ecog_lbl)
        
        # return test and train scores for evaluating model generalization
        return score_test, score_train

    def predict_proba(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # pred : array-like
        #     Predicted probability of each sample being in class 1

        if self._logreg is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self._logreg.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            pred = self._logreg(X)
            pred = pred.squeeze().numpy()

        return pred
    
    def predict(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # pred : array-like
        #     Predicted labels

        if self._logreg is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self._logreg.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            pred = self._logreg(X)
            pred = pred.squeeze().numpy()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
        return pred
    
    def get_coefs(self):
        # Returns
        # -------
        # coefs : array-like
        #     Coefficients for each feature

        if self._logreg is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self._logreg.eval()
        with torch.no_grad():
            coefs = self._logreg[0].weight.numpy().squeeze()
        return coefs
    
    def get_intercept(self):
        # Returns
        # -------
        # intercept : float
        #     Intercept value

        if self._logreg is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self._logreg.eval()
        with torch.no_grad():
            intercept = self._logreg[0].bias.numpy().squeeze()
        return intercept
    
    def score(self, X, y):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature
        # y : array-like
        #     Array of labels, where each row is the dummy coded label for the corresponding row in X

        # Returns
        # -------
        # score : float
        #     Balanced accuracy score

        pred = self.predict(X)
        score = balanced_accuracy_score(y, pred)
        return score*100