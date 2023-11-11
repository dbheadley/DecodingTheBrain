import torch
from torch.utils.data import Dataset, DataLoader


class ERPData(Dataset):
    def __init__(self, erp_pred, trial_lbl, transform=None, target_transform=None):
        # Parameters
        # ----------
        # erp_pred : array-like
        #     ERP data
        # trial_lbl : array-like
        #     Trial labels
        # transform : callable, optional
        #     Optional transform to be applied to the ERP data
        # target_transform : callable, optional
        #     Optional transform to be applied to the trial labels

        self.erp_pred = erp_pred
        self.trial_lbl = trial_lbl
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # Returns
        # -------
        # len : int
        #     Number of samples in the dataset

        return len(self.trial_lbl)
    
    def __getitem__(self, idx):
        # Parameters
        # ----------
        # idx : int
        #     Index of the sample to return

        # Returns
        # -------
        # erp : array-like
        #     ERP data for the selected sample
        # lbl : array-like
        #     Trial label for the selected sample
        
        erp = self.erp_pred[idx] # get the ERP data for the selected sample
        lbl = self.trial_lbl[idx] # get the trial label for the selected sample
        if self.transform is not None: # apply the transform to the ERP data
            erp = self.transform(erp)
        if self.target_transform is not None: # apply the transform to the trial label
            lbl = self.target_transform(lbl)
        return erp, lbl

erp_ds = ERPData(X, y)
erp_dl = DataLoader(erp_ds, batch_size=5, shuffle=True)

lin_trans = torch.nn.Linear(1, 1, bias=True)
log_trans = torch.nn.Sigmoid()
log_mdl = torch.nn.Sequential(
    lin_trans,
    log_trans
)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(log_mdl.parameters(), lr=0.001, momentum=0.9)

# get a batch of data with the dataloader
erps, lbls = next(iter(erp_dl))

# predict the labels for the batch
y_hat = log_mdl(erps.float())

# calculate the loss
loss = loss_fn(y_hat, lbls.float().reshape(-1,1))

optimizer.zero_grad()
loss.backward()

optimizer.step()


# create a linear transformation with 1 input, 1 output, and a bias
lin_trans = torch.nn.Linear(1, 1, bias=True)
log_trans = torch.nn.Sigmoid()
log_mdl = torch.nn.Sequential(
    lin_trans,
    log_trans
)

# initialize binary cross-entropy loss function and SGD optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(log_mdl.parameters(), lr=0.001, momentum=0.9)

acc = []
for i in range(200):
    y_pred = log_mdl(torch.tensor(X.astype(np.float32))).detach().numpy()>0.5
    acc.append(np.mean(y_pred.ravel() == y.ravel())*100)
    for erp, lbl in erp_dl:
        y_hat = log_mdl(erp.float())
        loss = loss_fn(y_hat, lbl.float().reshape(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if i%10 == 0:
        print('Epoch {}: loss = {:.4f}, accuracy = {:.2f}%'.format(i, loss, acc[-1]))
