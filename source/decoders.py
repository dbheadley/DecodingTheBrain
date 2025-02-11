import torch
import numpy as np
from .loaders import EcogFingerData
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

def format_ecogfinger_data_raw(data=None, flex_events=None, **kwargs):
    """Format ECoGFinger data for decoding.
    
    Parameters
    ----------
    data : EcogFingerData
        Data to format.
    finger : str
        Finger to decode.
    flex_events : array
        Flexion event times
    kwargs : dict
        Additional keyword arguments to pass to data.get_spec.


    Returns
    -------
    X : array, shape (n_epochs, n_channels, n_timepoints)
        Features for each epoch.
    y : array, shape (n_epochs,)
        Labels for each epoch.
    """

    # get movement and null spec epochs, 1 s after each thumb flexion event

    _,flexes = data.get_sig(event_times=flex_events, pre_t=0.2, post_t=0.2)
    _,nulls = data.get_sig(event_times=flex_events-1, pre_t=0.2, post_t=0.2)

    # # mean power across time for each epoch
    # flexes = np.mean(flexes, axis=3)
    # nulls = np.mean(nulls, axis=3)
    
    # z_mean = np.mean(total_data, axis=3).squeeze() # squeeze to remove singleton epoch dimension
    # z_std = np.std(total_data, axis=3).squeeze() 
    # flexes = (flexes - z_mean) / z_std
    # nulls = (nulls - z_mean) / z_std


    # create labels for thumb movements and nulls
    lbls =  np.concatenate((np.ones(flexes.shape[0]), np.zeros(nulls.shape[0])), axis=0)

    # stack flexes and thumb_nulls along first dimension
    feats = np.concatenate((flexes, nulls), axis=0)

    # reformat features so that each trial is a row and each column is a feature
    #feats = feats.reshape(feats.shape[0],-1)

    return feats, lbls

def format_ecogfinger_data_spec(data=None, flex_events=None, **kwargs):
    """Format ECoGFinger data for decoding.
    
    Parameters
    ----------
    data : EcogFingerData
        Data to format.
    finger : str
        Finger to decode.
    flex_events : array
        Flexion event times
    kwargs : dict
        Additional keyword arguments to pass to data.get_spec.


    Returns
    -------
    X : array, shape (n_epochs, n_features)
        Features for each epoch.
    y : array, shape (n_epochs,)
        Labels for each epoch.
    """

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
    lbls = np.hstack((np.ones(flexes.shape[0]), np.zeros(nulls.shape[0])))

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
        lbl = self.ecog_lbl[idx, np.newaxis] #.astype(np.float32).reshape(-1,1) # get the trial label for the selected sample
        if self.transform is not None: # apply the transform to the ECoG data
            feat = self.transform(feat)
        if self.target_transform is not None: # apply the transform to the trial label
            lbl = self.target_transform(lbl)
        return feat, lbl
    
    

class LogRegPT():
    def __init__(self, lr=0.01, epochs=100, train_prop=0.8, batch_size=5, lam=0.0, 
                 reg_layer=-2, shuffle_seed=None):
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
        # reg_layer : int, optional
        #     Layer to apply regularization to. Default is -2, which is the linear
        #     layer for logistic regression.
        # shuffle_seed : int, optional
        #     Seed for shuffling data before splitting into train and test sets
        #     Ensures same train and test sets are used across models. 
        #     Default is None, which will result in different train and test sets 
        #     for each model.

        self.lr = lr
        self.epochs = epochs
        self.train_prop = train_prop
        self.batch_size = batch_size
        self.lam = lam
        self.reg_layer = reg_layer
        self._model = None
        self.train_idxs = None
        self.test_idxs = None
        self._shuffle_seed = shuffle_seed
    
    def _create_model(self, X, y):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature.
        #     Used to determine the input dimension of the model.
        # y : array-like
        #     Array of labels, where each element is the label for the corresponding row in X
        #     Used to determine the number of classes in the model. Not important for binary classification.

        # input size is the number of features, input will be flattened
        input_size = np.prod(X.shape[1:])

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(input_size, 1)

        # sigmoid layer is the activation function
        sig_layer = torch.nn.Sigmoid()

        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(X.shape[1]), # normalize the input features of each channel (aids with convergence)
            torch.nn.Flatten(1), # flatten the input so that each trial is a row and each column is a feature
            lin_layer,
            sig_layer
        )
        self._model = model
    
    def loss(self, pred, act):
        # Parameters
        # ----------
        # pred : array-like
        #     Predicted probability of each sample being in class 1
        # act : array-like
        #     Array of labels, where each element is the label for the corresponding row in X

        # Returns
        # -------
        # loss : float
        #     Loss value

        act = act.float()
        loss_fn = torch.nn.BCELoss(reduction='mean')

        # calculate loss
        loss = loss_fn(pred, act)

        if self.lam > 0:
            # add L1 regularization
            loss += self.lam*torch.sum(torch.abs(self._model[self.reg_layer].weight))

        return loss

    def _create_optim(self):
        # initialize optimizer
        return torch.optim.SGD(self._model.parameters(), lr=self.lr)
    
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
        self._create_model(X, y)
        optim = self._create_optim()
        
        # split data into train and test sets
        strat = StratifiedKFold(n_splits=int(1/(1-self.train_prop)), shuffle=True, random_state=self._shuffle_seed)
        self.train_idxs, self.test_idxs = strat.split(X, y).__next__()

        if self.train_idxs.size < self.batch_size:
            raise ValueError('Number of training samples smaller than batch size')
        
        # create data loaders for train and test sets
        train_ds = ECoGData(X[self.train_idxs], y[self.train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[self.test_idxs], y[self.test_idxs])
        
        # train model
        self._model.train()
        for epoch in range(self.epochs):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self._model(feat)
                loss = self.loss(pred, lbl)
                loss.backward()
                optim.step()
            
            if np.mod(epoch, 10) == 0:
                # get predictions for train and test sets
                if self.train_prop < 1.0:
                    score_test = self.score(test_ds.ecog_feat, test_ds.ecog_lbl)
                else:
                    score_test = np.nan
                score_train = self.score(train_ds.ecog_feat, train_ds.ecog_lbl)
                print(f'Epoch {epoch}: Train: {score_train:.2f}, Test: {score_test:.2f}')
                self._model.train() # set model back to train mode for next epoch

        self._model.eval() # set model to eval mode once fitting is done

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

        if self._model is None:
            raise ValueError('Model has not been fit yet.')
        
        self._model.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            pred = self._model(X)
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

        if self._model is None:
            raise ValueError('Model has not been fit yet.')
        
        self._model.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            pred = self._model(X)
            pred = pred.squeeze().numpy()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
        return pred
    
    def get_coefs(self):
        # Returns
        # -------
        # coefs : array-like
        #     Model coefficients

        if self._model is None:
            raise ValueError('Model has not been fit yet.')
        
        self._model.eval()
        coefs = []
        with torch.no_grad():
            for layer in self._model:
                # if layer has weights, get them, otherwise empty array
                if hasattr(layer, 'weight'):
                    coefs.append(layer.weight.numpy())
                else:
                    coefs.append(np.array([]))

        return coefs
    
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
    

class LogRegPT_MC(LogRegPT):
    
    def __init__(self, reg_layer=-1, **kwargs):
        super().__init__(reg_layer=reg_layer, **kwargs)
    
    def _create_model(self, X, y):
        # Creates a logistic model for multi-class classification
        # No explicit softmax layer because we want to use CrossEntropyLoss

        n_features = np.prod(X.shape[1:])
        n_classes = len(np.unique(y))

        # linear layer is the weights and bias
        lin_layer = torch.nn.Linear(n_features, n_classes) 
        
        # logistic regression model, no sigmoid layer because we want to use softmax
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(X.shape[1]), # normalize the input features of each channel (aids with convergence)
            torch.nn.Flatten(1), # flatten the input so that each trial is a row and each column is a feature
            lin_layer,
        )
        self._model = model


    def loss(self, pred, act):
        # Parameters
        # ----------
        # pred : array-like
        #     Predicted probability of each sample being in class 1
        # act : array-like
        #     Dummy coded array of actual classes

        # Returns
        # -------
        # loss : float
        #     Loss value

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        # calculate loss
        loss = loss_fn(pred, act)

        # add L1 regularization at final layer
        if self.lam > 0:
            loss += self.lam*torch.sum(torch.abs(self._model[-1].weight)) 

        return loss
    

    def _create_optim(self):
        # initialize optimizer
        return torch.optim.SGD(self._model.parameters(), lr=self.lr)
    
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
        self._input_shape = X.shape
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)

        self._create_model()
        optim = self._create_optim()
        
        # split data into train and test sets
        strat = StratifiedKFold(n_splits=int(1/(1-self.train_prop)), shuffle=True, random_state=self._shuffle_seed)
        self.train_idxs, self.test_idxs = strat.split(X, y).__next__()

        if self.train_idxs.size < self.batch_size:
            raise ValueError('Number of training samples smaller than batch size')

        # create data loaders for train and test sets
        train_ds = ECoGData(X[self.train_idxs,:], y[self.train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[self.test_idxs,:], y[self.test_idxs])
        
        # train model
        for epoch in range(self.epochs):
            self._mdl.train()
            for feat, lbl in train_dl:
                optim.zero_grad()
                y_pred = self._mdl(feat)
                loss = self.loss(y_pred, self._lbl_to_onehot(lbl))
                loss.backward()
                optim.step()
            
            # evaluate model every 10 epochs
            if np.mod(epoch, 10) == 0:
                # get predictions for train and test sets
                if self.train_prop < 1.0:
                    score_test = self.score(test_ds.ecog_feat, self._lbl_to_onehot(test_ds.ecog_lbl))
                else:
                    score_test = np.nan

                score_train = self.score(train_ds.ecog_feat, self._lbl_to_onehot(train_ds.ecog_lbl))
                self._mdl.train()
        
        # return test and train scores for evaluating model generalization
        return score_test, score_train
    
    def _lbl_to_onehot(self, lbl):
        # Parameters
        # ----------
        # lbl : array-like
        #     Array of labels, where each element is the label for the corresponding row in X

        # Returns
        # -------
        # lbl_one : array-like
        #     Dummy coded array of actual classes

        if type(lbl) is np.ndarray:
            lbl = torch.tensor(lbl)

        lbl = lbl.type(torch.int64)
        lbl_one = torch.nn.functional.one_hot(lbl.squeeze(), num_classes=self._n_classes)
        lbl_one = lbl_one.type(torch.float32)
        return lbl_one