import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ECoGData(Dataset):
    def __init__(self, ecog_feat, ecog_lbl, transform=None, target_transform=None):
        # Parameters
        # ----------
        # ecog_feat : array-like
        #     Array of features, where each row is a trial and each column is a feature
        # ecog_lbl : array-like
        #     Array of labels, where each element is the label for the corresponding row in ecog_feat
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
        # n_folds : int, optional
        #     Number of folds for cross-validation
        # batch_size : int, optional
        #     Number of samples per batch
        # lam : float, optional
        #     Regularization parameter for L1 norm

        self.lr = lr
        self.epochs = epochs
        self.train_prop = train_prop
        self.batch_size = batch_size
        self.lam = lam
        self.logreg_ = None
    
    def create_logreg_(self, input_dim):
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
        self.logreg_ = logreg
    
    def create_loss_(self):
        # initialize loss function
        return torch.nn.BCELoss(reduction='mean')

    def create_optim_(self):
        # initialize optimizer
        return torch.optim.SGD(self.logreg_.parameters(), lr=self.lr)
    
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
        self.create_logreg_(feat_num)
        optim = self.create_optim_()
        loss_fn = self.create_loss_()
        
        # split data into train and test sets
        train_num = int(self.train_prop*X.shape[0])
        train_idxs = np.random.permutation(X.shape[0])[:train_num]
        test_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)

        if train_idxs.size < self.batch_size:
            raise ValueError('Number of training samples smaller than batch size')
        
        # create data loaders for train and test sets
        train_ds = ECoGData(X[train_idxs,:], y[train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[test_idxs,:], y[test_idxs])
        
        # train model
        self.logreg_.train()
        for epoch in range(self.epochs):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self.logreg_(feat)
                loss = loss_fn(pred, lbl.float().reshape(-1,1))

                # add L1 regularization
                loss += self.lam*torch.sum(torch.abs(self.logreg_[0].weight)) 
                
                loss.backward()
                optim.step()
            
        # get predictions for test set
        self.logreg_.eval()
        with torch.no_grad():
            pred = self.logreg_(torch.tensor(test_ds.ecog_feat.astype(np.float32)))
            pred = pred.squeeze().numpy()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
            score_test = balanced_accuracy_score(test_ds.ecog_lbl, pred)

        # get predictions for train set
        with torch.no_grad():
            pred = self.logreg_(torch.tensor(train_ds.ecog_feat.astype(np.float32)))
            pred = pred.squeeze().numpy()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
            score_train = balanced_accuracy_score(train_ds.ecog_lbl, pred)
        
        # return test and train scores for evaluating model generalization
        return score_test*100, score_train*100

    def predict(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # pred : array-like
        #     Predicted labels

        if self.logreg_ is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self.logreg_.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            pred = self.logreg_(X)
            pred = pred.squeeze().numpy()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
        return pred
    
    def get_coefs(self):
        # Returns
        # -------
        # coefs : array-like
        #     Coefficients for each feature

        if self.logreg_ is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self.logreg_.eval()
        with torch.no_grad():
            coefs = self.logreg_[0].weight.numpy().squeeze()
        return coefs
    
    def get_intercept(self):
        # Returns
        # -------
        # intercept : float
        #     Intercept value

        if self.logreg_ is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self.logreg_.eval()
        with torch.no_grad():
            intercept = self.logreg_[0].bias.numpy().squeeze()
        return intercept