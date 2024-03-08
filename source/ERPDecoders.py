"""
This module contains the evoked response potential (ERP) decoders built in Weeks 3 and 4, which are used to decode the presence
of an ERP component EEG epoch using logistic regression.

Created by Drew Headley
"""

# import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class ERP_Decode_SK():
    """
    This class implements the ERP decoder using logistic regression from scikit-learn. The class is initialized with the
    number of folds for cross-validation (k) and the random seed for reproducibility (rand_seed). The class contains the
    following methods:
        
        set_rand_seed(rand_seed): set the random seed to generate new random folds
        erp_calc_(epochs): calculate the ERP
        erp_align_(epochs, erp): align the ERP to the origin
        fit(epochs, labels): fit the model
        decision_boundary(): get the decision boundary
        model_coef(): get the model coefficients
        model_intercept(): get the model intercept
        predict(epochs): predict the labels of new data using the trained model
    
    """

    def __init__(self, k=5, rand_seed=47, **kwargs):
        """
        This method initializes the ERP_Decode_SK class with the number of folds for cross-validation (k) and the random seed
        for reproducibility (rand_seed). The method also initializes the logistic regression model with the specified keyword
        arguments (**kwargs).

        Parameters
        ----------
        k : int, optional
            The number of folds for cross-validation. The default is 5.

        rand_seed : int, optional
            The random seed for reproducibility. The default is 47.
        """
        self._k = k # number of folds for cross-validation
        self._rand_seed = rand_seed # random seed for reproducibility
        self._test_acc = None 
        self._train_acc = None
        self.erp = None # ERP is the average of the ERP labeled epochs
        self._logreg = LogisticRegression(**kwargs) # **kwargs allows us to pass in arguments to the LogisticRegression class
        self._stratkfold = StratifiedKFold(n_splits=self._k, shuffle=True, random_state=self._rand_seed)
    
    def set_rand_seed(self, rand_seed):
        """
        This method sets the random seed to generate new random folds.
        
        Parameters
        ----------
        rand_seed : int
            The random seed for reproducibility.
        """
        # set random seed to generate new random folds
        self._rand_seed = rand_seed
        self._stratkfold = StratifiedKFold(n_splits=self._k, shuffle=True, random_state=self._rand_seed)
    
    def erp_calc_(self, epochs):
        """
        This method calculates the ERP.
        
        Parameters
        ----------
        epochs : np.ndarray
            The epochs of EEG data.

        Returns
        -------
        np.ndarray
            The ERP.
        """

        # calculate the ERP
        return np.mean(epochs, axis=0)
    
    def erp_align_(self, epochs, erp):
        """
        This method calculates the alignment between the ERP and epoched EEG data.

        Parameters
        ----------
        epochs : np.ndarray
            The epochs of EEG data.
        erp : np.ndarray
            The ERP.

        Returns
        -------
        np.ndarray
            The aligned epochs.
        """

        return np.dot(epochs, erp.T/np.linalg.norm(erp))[:,np.newaxis]
    
    def fit(self, epochs, labels):
        """
        This method fits the model.

        Parameters
        ----------
        epochs : np.ndarray
            The epochs of EEG data. Each epoch is a row, and each column is a time point.
        labels : np.ndarray
            The labels for the epochs.

        Returns
        -------
        float
            The training accuracy.
        float
            The testing accuracy.
        """
        # fit the model

        # get the training and testing indices for first fold
        train_idxs, test_idxs = next(self._stratkfold.split(epochs, labels))
        
        # get the training and testing data for first fold
        y_train = labels[train_idxs]
        X_train = epochs[train_idxs]
        train_cue_idxs = np.intersect1d(np.where(y_train)[0], train_idxs)
        self.erp = self.erp_calc_(X_train[train_cue_idxs])
        X_train = self.erp_align_(X_train, self.erp)
        X_test = self.erp_align_(epochs[test_idxs], self.erp)
        y_test = labels[test_idxs]
        
        # fit the model
        self._logreg.fit(X_train, y_train)
        self._train_acc = self._logreg.score(X_train, y_train)*100
        self._test_acc = self._logreg.score(X_test, y_test)*100
        
        # return the training and testing accuracies
        return self._train_acc, self._test_acc
    
    def decision_boundary(self):
        """
        This method gets the decision boundary.
        
        Returns
        -------
        float
            The decision boundary.
        """
        # get the decision boundary
        return -self._logreg.intercept_[0]/self._logreg.coef_[0,0]

    def model_coef(self):
        """
        This method gets the model coefficient/weight.

        Returns
        -------
        float
            The model coefficient/weight.
        """

        # get the model coefficients
        return self._logreg.coef_[0,0]
    
    def model_intercept(self):
        """
        This method gets the model intercept.

        Returns
        -------
        float
            The model intercept.
        """

        # get the model intercept
        return self._logreg.intercept_[0]
    
    def predict(self, epochs):
        """
        This method predicts the labels of new data using the trained model.

        Parameters
        ----------
        epochs : np.ndarray
            The epochs of EEG data.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """

        # predict the labels of new data using the trained model
        return self._logreg.predict(self.erp_align_(epochs, self.erp))