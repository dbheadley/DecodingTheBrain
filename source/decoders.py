import torch
import copy
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
                 shuffle_seed=None, verbose=False):
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
        # shuffle_seed : int, optional
        #     Seed for shuffling data before splitting into train and test sets
        #     Ensures same train and test sets are used across models. 
        #     Default is None, which will result in different train and test sets 
        #     for each model.
        # verbose : bool, optional
        #     Whether to print progress during training. Default is False.

        self.lr = lr
        self.epochs = epochs
        self.train_prop = train_prop
        self.batch_size = batch_size
        self.lam = lam
        self._model = None
        self.train_idxs = None
        self.test_idxs = None
        self._score_test = []
        self._score_train = []
        self._X_shape = None
        self._y_shape = None
        self._classes = None
        self._n_classes = None
        self._shuffle_seed = shuffle_seed
        self.verbose = verbose

    def _create_model(self):

        # layer where regularization is applied
        self.reg_layer = 0

        # input size is the number of features
        input_size = self._X_shape[1]

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(input_size, 1)

        # sigmoid layer is the activation function
        sig_layer = torch.nn.Sigmoid()

        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            lin_layer,
            sig_layer
        )
        self._model = model

    def _loss(self, pred, lbl):
        # create loss function
        loss_fn = torch.nn.BCELoss(reduction='mean')

        loss = loss_fn(pred, lbl.float())

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

        self._X_shape = X.shape
        self._y_shape = y.shape
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
        train_ds = ECoGData(X[self.train_idxs], y[self.train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[self.test_idxs], y[self.test_idxs])
        
        # train model
        self._model.train()
        for epoch in range(self.epochs):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self._model(feat)
                loss = self._loss(pred, lbl)

                # add L1 regularization
                if self.lam > 0:
                    loss += self.lam*torch.sum(torch.abs(self._model[self.reg_layer].weight))

                loss.backward()
                optim.step()
            
            if (np.mod(epoch, 10) == 0):
                # get predictions for train and test sets
                if self.train_prop < 1.0:
                    self._score_test.append(self.score(test_ds.ecog_feat, test_ds.ecog_lbl))
                else:
                    self._score_test.append(np.nan)
                self._score_train.append(self.score(train_ds.ecog_feat, train_ds.ecog_lbl))

                if self.verbose:
                    print(f'Epoch {epoch}: Train: {self._score_train[-1]:.1f}, Test: {self._score_test[-1]:.1f}')
                self._model.train() # set model back to train mode for next epoch

        self._model.eval() # set model to eval mode once fitting is done

        if self.train_prop < 1.0:
            self._score_test.append(self.score(test_ds.ecog_feat, test_ds.ecog_lbl))
        else:
            self._score_test.append(np.nan)
        self._score_train.append(self.score(train_ds.ecog_feat, train_ds.ecog_lbl))
        
        # return test and train scores for evaluating model generalization
        return self._score_test[-1], self._score_train[-1]

    def predict_proba(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # prob : array-like
        #     Predicted probability of each sample being in class 1

        if self._model is None:
            raise RuntimeError('Model has not been fit yet.')
        
        self._model.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            prob = self._model(X)
            prob = prob.squeeze().numpy()

        return prob
    
    def predict(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # pred : array-like
        #     Predicted labels

        prob = self.predict_proba(X)
        
        pred = np.zeros(prob.shape)
        pred[prob>=0.5] = 1

        return pred
    
    def get_coefs(self):
        # Returns
        # -------
        # coefs : array-like
        #     Model coefficients

        if self._model is None:
            raise RuntimeError('Model has not been fit yet.')
        
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
    
    def get_intercept(self):
        """
        Get intercept

        Returns
        -------
        intercept : float
            Intercept value
        """

        if self._model is None:
            raise ValueError('Logistic regression model has not been fit yet.')
        
        self._model.eval()
        with torch.no_grad():
            # find last layer with linear type
            for layer in reversed(self._model):
                if isinstance(layer, torch.nn.Linear):
                    intercept = layer.bias.numpy().squeeze()
                    break
            
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
    

class LogRegPT_MC(LogRegPT):
    
    
    def _create_model(self):
        
         # layer where regularization is applied
        self.reg_layer = 2

        # input size is the number of features, input will be flattened
        input_size = np.prod(self._X_shape[1:])
        n_ecog_ch = self._X_shape[1]

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(input_size, self._n_classes)

        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_ecog_ch), # normalize the input features of each channel (aids with convergence)
            torch.nn.Flatten(1), # flatten the input so that each trial is a row and each column is a feature
            lin_layer,
        )
        self._model = model

    def _loss(self, pred, lbl):
        # create loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        lbl = lbl.type(torch.int64).squeeze()

        loss = loss_fn(pred, lbl)

        return loss

    def _create_optim(self):
        # initialize optimizer
        return torch.optim.Adam(self._model.parameters(), lr=self.lr)
    
    def predict_proba(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # prob : array-like
        #     Predicted probability of each sample being in class 1

        if self._model is None:
            raise RuntimeError('Model has not been fit yet.')
        
        self._model.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            prob = self._model(X)
            prob = torch.nn.functional.softmax(prob, dim=1)
            prob = prob.squeeze().numpy()

        return prob
    
    def predict(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # pred : array-like
        #     Predicted labels

        prob = self.predict_proba(X)
        pred = np.argmax(prob, axis=1)
        
        return pred


class LogRegPT_CNN(LogRegPT_MC):

    def _create_model(self):

         # layer where regularization is applied
        self.reg_layer = -1

        n_ecog_ch = self._X_shape[1]
        n_conv_ch = 10
        mp_stride = 5
        lin_input_sz = n_conv_ch * (self._X_shape[2]//mp_stride) - n_conv_ch

        # CNN layer
        conv1 = torch.nn.Conv1d(n_ecog_ch, n_conv_ch, 21, stride=1, dilation=3, padding='same')
        mp = torch.nn.MaxPool1d(n_conv_ch,stride=mp_stride)

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(lin_input_sz, self._n_classes)

        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_ecog_ch), # normalize the input features of each channel (aids with convergence)
            conv1,
            torch.nn.ReLU(),
            mp,
            torch.nn.Flatten(1),
            torch.nn.BatchNorm1d(lin_input_sz),
            lin_layer,
        )
        self._model = model


class LogRegPT_CNN_Adam(LogRegPT_CNN):
    def _create_optim(self):
        # initialize optimizer
        return torch.optim.Adam(self._model.parameters(), lr=self.lr)
    

class LogRegPT_CNN_Dropout(LogRegPT_MC):

    def _create_optim(self):
        # initialize optimizer
        return torch.optim.Adam(self._model.parameters(), lr=self.lr)
    
    def _create_model(self):

         # layer where regularization is applied
        self.reg_layer = -1

        n_ecog_ch = self._X_shape[1]
        n_conv_ch = 20
        mp_stride = 5
        lin_input_sz = n_conv_ch * (self._X_shape[2]//mp_stride) - 3*n_conv_ch

        # CNN layer
        conv1 = torch.nn.Conv1d(n_ecog_ch, n_conv_ch, 21, stride=1, dilation=3, padding='same')
        mp = torch.nn.MaxPool1d(n_conv_ch, stride=mp_stride)
        dropout = torch.nn.Dropout(p=0.5)

        # linear layer is the weights and bias
        # input_dim is the number of input features and 1 is the number of output features
        # this is taking the dot product of the input features with the weights and adding the bias
        lin_layer = torch.nn.Linear(lin_input_sz, self._n_classes)

        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            conv1,
            mp,
            torch.nn.BatchNorm1d(n_conv_ch), # normalize the input features of each channel (aids with convergence)
            torch.nn.ReLU(),
            dropout,
            torch.nn.Flatten(1),
            lin_layer,
        )
        self._model = model


class LogRegPT_AE(LogRegPT_MC):

    # expand init method to include autoencoder parameters
    def __init__(self, epochs_ae=100, lr_ae=0.01, n_code=20, 
                 update_encode=False, **kwargs):
        # Parameters
        # ----------
        # lr : float, optional
        #     Learning rate for gradient descent
        # epochs : int, optional
        #     Number of epochs to train for
        
        super().__init__(**kwargs)
        self.epochs_ae = epochs_ae
        self.lr_ae = lr_ae
        self._autoencoder = None
        self._loss_ae = []
        self._n_code = n_code
        self.update_encode = update_encode

    def _create_model(self):

         # layer where regularization is applied
        self.reg_layer = 2

        # linear layer is the weights and bias
        lin_layer = torch.nn.Linear(self._n_code, self._n_classes)

        # load in the encoder layer from the autoencoder
        # and set whether it will be updated when training
        # the logistic regression model
        encode_layer = copy.deepcopy(self._autoencoder[0])
        if not self.update_encode:
            for param in encode_layer.parameters():
                param.requires_grad = False
        
        # logistic regression model is a sequential combination of linear and sigmoid layers
        model = torch.nn.Sequential(
            encode_layer,
            torch.nn.BatchNorm1d(self._n_code, track_running_stats=False), # normalize the input features of each channel (aids with convergence)
            lin_layer,
        )
        self._model = model

    def _create_autoencoder(self):
        n_ecog_ch = self._X_shape[1]
        n_conv_ch = 30
        lin_input_sz = n_conv_ch * (self._X_shape[2]//4)

        # CNN layers
        encoder = torch.nn.Sequential(
            torch.nn.Conv1d(n_ecog_ch, n_conv_ch, 11, padding='same'),
            torch.nn.MaxPool1d(4, stride=4),
            torch.nn.BatchNorm1d(n_conv_ch),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(lin_input_sz, self._n_code),
        )

        decoder = torch.nn.Sequential(
            torch.nn.Linear(self._n_code, lin_input_sz),
            torch.nn.Unflatten(1, (n_conv_ch, lin_input_sz//n_conv_ch)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=4),
            torch.nn.ConvTranspose1d(n_conv_ch, n_ecog_ch, 11, padding=5),
            torch.nn.BatchNorm1d(n_ecog_ch)
        )

        autoencoder = torch.nn.Sequential(
            encoder,
            decoder,
        )

        self._autoencoder = autoencoder

    # takes in data, returns encoded representation
    def encode(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # code : array-like
        #     Encoded representation of the input data

        if self._autoencoder is None:
            raise RuntimeError('Autoencoder has not been fit yet.')
        
        self._autoencoder.eval()

        X = torch.tensor(X.astype(np.float32))
        
        with torch.no_grad():
            code = self._autoencoder[0](X)
            code = code.squeeze().numpy()

        return code
    
    def _loss_autoencoder(self, pred, act):
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(pred, act)

        return loss
    
    def _create_optim_autoencoder(self):
        # initialize optimizer
        return torch.optim.SGD(self._autoencoder.parameters(), lr=self.lr_ae)
    
    def _fit_autoencoder(self, train_dl):
        # Parameters
        # ----------
        # train_dl : DataLoader
        #     DataLoader for training data

        # initialize model and fitting

        
        self._create_autoencoder()
        optim = self._create_optim_autoencoder()

        # train model
        self._autoencoder.train()
        for epoch in range(self.epochs_ae):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self._autoencoder(feat)
                loss = self._loss_autoencoder(pred, feat)
                loss.backward()
                optim.step()
                
            if np.mod(epoch, 10) == 0:
                # save and print loss value
                self._loss_ae.append(loss.item())
                if self.verbose:
                    print(f'Epoch {epoch}: Loss: {self._loss_ae[-1]:.2f}')

        self._autoencoder.eval() # set model to eval mode once fitting is done

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

        self._X_shape = X.shape
        self._y_shape = y.shape
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)


        # split data into train and test sets
        strat = StratifiedKFold(n_splits=int(1/(1-self.train_prop)), shuffle=True, random_state=self._shuffle_seed)
        self.train_idxs, self.test_idxs = strat.split(X, y).__next__()

        if self.train_idxs.size < self.batch_size:
            raise ValueError('Number of training samples smaller than batch size')
        
        # create data loaders for train and test sets
        train_ds = ECoGData(X[self.train_idxs], y[self.train_idxs])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = ECoGData(X[self.test_idxs], y[self.test_idxs])

        # fit autoencoder
        print('Fitting autoencoder')
        self._fit_autoencoder(train_dl)
        
        # initialize logistic regression model
        self._create_model()
        optim = self._create_optim()

        # train model
        print('Training logistic regression model')
        self._model.train()
        for epoch in range(self.epochs):
            for feat, lbl in train_dl:
                optim.zero_grad()
                pred = self._model(feat)
                loss = self._loss(pred, lbl)

                # add L1 regularization
                if self.lam > 0:
                    loss += self.lam*torch.sum(torch.abs(self._model[self.reg_layer].weight))

                loss.backward()
                optim.step()
            
            if np.mod(epoch, 10) == 0:
                # get predictions for train and test sets
                if self.train_prop < 1.0:
                    self._score_test.append(self.score(test_ds.ecog_feat, test_ds.ecog_lbl))
                else:
                    self._score_test.append(np.nan)
                self._score_train.append(self.score(train_ds.ecog_feat, train_ds.ecog_lbl))
                if self.verbose:
                    print(f'Epoch {epoch}: Train: {self._score_train[-1]:.2f}, Test: {self._score_test[-1]:.2f}')
                self._model.train() # set model back to train mode for next epoch

        self._model.eval() # set model to eval mode once fitting is done

        # return test and train scores for evaluating model generalization
        return self._score_test[-1], self._score_train[-1]

    """
    def predict_proba(self, X):
        # Parameters
        # ----------
        # X : array-like
        #     Array of features, where each row is a trial and each column is a feature

        # Returns
        # -------
        # prob : array-like
        #     Predicted probability of each sample being in class 1

        if self._model is None:
            raise RuntimeError('Model has not been fit yet.')

        self._model.eval()
        
        X_enc = self.encode(X)
        X_enc = torch.tensor(X_enc.astype(np.float32))
        

        with torch.no_grad():
            prob = self._model(X_enc)
            prob = torch.nn.functional.softmax(prob, dim=1)
            prob = prob.squeeze().numpy()

        return prob
        """