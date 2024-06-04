import math
import torch
import torchmetrics
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial
import torchmetrics.functional as tm_f
import torch.distributions as dist
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from torchmetrics import Metric
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision,MultilabelAUROC
import numpy as np
from torch import nn
from scipy import stats
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from torchmetrics.utilities import dim_zero_cat


class CorrectAggregatedMetric(Metric):
    """This is needed to calculate some metrics b/c small batch sizes cause aggregation via a simple
        average to be off, as some classes might not be present in batch but will get penalized with a 0."""
    def __init__(self, class_idx: int, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_idx = torch.tensor(class_idx)
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, numerator, denominator, preds, y) -> tuple:
        raise NotImplemented

    def update(self, logits: torch.Tensor, y: torch.Tensor):
        # update metric states
        preds = torch.argmax(logits, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
        self.numerator, self.denominator = self._update(self.numerator, self.denominator, preds, y)

    def compute(self):
        # compute final result
        value = self.numerator.float() / self.denominator if self.denominator > 0 else torch.tensor(0.0)
        return value

    def reset(self):
        self.numerator = torch.tensor(0.0)
        self.denominator = torch.tensor(0.0)

class AccuracyPerClass(CorrectAggregatedMetric):
    """Calculate per class accuracy, i.e. P(y_hat = class_idx AND y = class_idx OR y_hat != class_idx AND y != class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == class_idx).sum()
        denominator += relevant_idxs.sum()
        relevant_idxs = (y != class_idx)
        numerator += (preds[relevant_idxs] != class_idx).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator

class PrecisionPerClass(CorrectAggregatedMetric):
    """Calculate per class precision, i.e. P(y_hat = y | y_hat = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (preds == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


class RecallPerClass(CorrectAggregatedMetric):
    """Calculate per class recall, i.e. P(y_hat = y | y = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator



def mcc(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy())
    


def last_k_ppl(logits, y, seq_len=1024, k=None):
    '''
    Calculate perplexity for last k tokens in a sequence.

    logits: (batch_size * seq_len, vocab_size), note, already flattened
    y: (batch_size * seq_len), note, already flattened
    seq_len: int, length of each sequence in the batch
    k: if None, use all tokens in sequence
    
    returns: (batch_size,)  ppl for each sequence in the batch
    '''

    if k is None:
        k = 0  # use the entire sequence

    # need to reshape logits and y to be (batch_size, seq_len, vocab_size) and (batch_size, seq_len)
    # respectively
    # breakpoint()
    logits = logits.view(-1, seq_len, logits.shape[-1])
    y = y.view(-1, seq_len)

    # only use the last k values of seq dim in logits and y
    logits = logits[:, -k:, :]
    y = y[:, -k:]

    # reshape to flatten the batch and seq_len dimensions
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    # get avg and put on cpu
    return F.cross_entropy(logits, y, reduction='none').view(y.shape[0], -1).mean().exp().cpu()


def _student_t_map(mu, sigma, nu):
    sigma = F.softplus(sigma)
    nu = 2.0 + F.softplus(nu)
    return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

def student_t_loss(outs, y):
    mu, sigma, nu = outs[..., 0], outs[..., 1], outs[..., 2]
    mu, sigma, nu = _student_t_map(mu, sigma, nu)
    y = y.squeeze(axis=-1)

    nup1_half = (nu + 1.0) / 2.0
    part1 = 1.0 / nu * torch.square((y - mu) / sigma)
    Z = (
        torch.lgamma(nup1_half)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(math.pi * nu)
        - torch.log(sigma)
    )

    ll = Z - nup1_half * torch.log1p(part1)
    return -ll.mean()

def gaussian_ll_loss(outs, y):
    mu, sigma = outs[..., 0], outs[..., 1]
    y = y.squeeze(axis=-1)
    sigma = F.softplus(sigma)
    ll = -1.0 * (
        torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + 0.5 * torch.square((y - mu) / sigma)
    )
    return -ll.mean()

def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
    # return nn.BCEWithLogitsLoss(logits.squeeze(-1), y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()

def padded_cross_entropy(logits, y, pad_mask, pad_value=-1):
    """Will ignore the pad value in label (eg, -1)
    
    logits: (batch_size, seq_len, vocab_size)
    y: (batch_size, seq_len)
    pad_mask: (batch_size, seq_len)
    
    """

    # need to apply pad mask to y
    y_pad = y + pad_mask * pad_value

    logits = logits.view(-1, logits.shape[-1])
    y_pad = y_pad.view(-1)
    return F.cross_entropy(logits, y_pad, ignore_index=pad_value)


def cross_entropy(logits, y, ignore_index=-100):
    # logits = logits.view(-1, logits.shape[-1])
    logits = logits.reshape(-1, logits.shape[-1])
    #if y dim=3, apply argmax to get class label
    if y.dim() == 3:
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)
def genomic_structure_hff_mse(logit,y,ignore_index=-100):

    pred,_=logit
    target=y
    length=int(pred.shape[-1])
    if torch.isnan(target).all():
        target=torch.rand_like(target)

    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFI643OYP9.rebinned.mcool.expected.res1000.npy")

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.nanmean(
                torch.nanmean(torch.reshape(target, (target.shape[0], length, 1, length, 1)), axis=4),
                axis=2,
            )
    target_r = torch.log(((target + eps) / (normmat_r + eps)))
    target_cuda = target_r

    loss = (
        (
            pred[~torch.isnan(target)]
            - target_cuda[~torch.isnan(target)]
        )
        ** 2
    ).mean()



    return loss

def genomic_structure_hff_corr(logit,y,ignore_index=-100):


    (pred,pred_1d)=logit
    target=y
    length=int(pred.shape[-1])
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFI643OYP9.rebinned.mcool.expected.res1000.npy")        

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.log(((target + eps) / (normmat_r + eps)))
    corr=[]
    #convert to numpy
    target=target.cpu().numpy().reshape((pred.shape[0], -1))
    pred=pred.detach().cpu().numpy().reshape((pred.shape[0], -1))
    for j in range(pred.shape[0]):
            if np.mean(np.isnan(target[j, :])) < 0.7:
                corr.append(
                    pearsonr(
                        pred[j, ~np.isnan(target[j, :])],
                        target[j, ~np.isnan(target[j, :])],
                    )[0]
                )
            else:
                corr.append(np.nan)
    corr=np.nanmean(corr)
    return corr

def genomic_structure_h1esc_corr(logit,y,ignore_index=-100):


    (pred,pred_1d)=logit
    target=y
    length=int(pred.shape[-1])
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res1000.npy")        

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.log(((target + eps) / (normmat_r + eps)))
    corr=[]
    #convert to numpy
    target=target.cpu().numpy().reshape((pred.shape[0], -1))
    pred=pred.detach().cpu().numpy().reshape((pred.shape[0], -1))
    for j in range(pred.shape[0]):
            if np.mean(np.isnan(target[j, :])) < 0.7:
                corr.append(
                    pearsonr(
                        pred[j, ~np.isnan(target[j, :])],
                        target[j, ~np.isnan(target[j, :])],
                    )[0]
                )
            else:
                corr.append(np.nan)
    corr=np.nanmean(corr)
    return corr

def genomic_structure_hctnoc_corr(logit,y,ignore_index=-100):


    (pred,pred_1d)=logit
    target=y
    length=int(pred.shape[-1])
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool.expected.res1000.npy")        

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.log(((target + eps) / (normmat_r + eps)))
    corr=[]
    #convert to numpy
    target=target.cpu().numpy().reshape((pred.shape[0], -1))
    pred=pred.detach().cpu().numpy().reshape((pred.shape[0], -1))
    for j in range(pred.shape[0]):
            if np.mean(np.isnan(target[j, :])) < 0.7:
                corr.append(
                    pearsonr(
                        pred[j, ~np.isnan(target[j, :])],
                        target[j, ~np.isnan(target[j, :])],
                    )[0]
                )
            else:
                corr.append(np.nan)
    corr=np.nanmean(corr)
    return corr

def genomic_structure_mse_1d(logit,y,ignore_index=-100):

    (pred,pred_1d)=logit
    (target,target_1d)=y
    
    
    loss_1d=nn.BCEWithLogitsLoss()(pred_1d,target_1d)



    return loss_1d

def genomic_structure_hff_loss(logit,y,ignore_index=-100):
    pred,_=logit
    target=y
    length=int(pred.shape[-1])
    #check if target is all nan
    if torch.isnan(target).all():
        #give random positive value to avoid nan loss
        target=torch.rand_like(target)
    
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFI643OYP9.rebinned.mcool.expected.res1000.npy")

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.nanmean(
                torch.nanmean(torch.reshape(target, (target.shape[0], length, 1, length, 1)), axis=4),
                axis=2,
            )
    target_r = torch.log(((target + eps) / (normmat_r + eps)))
    target_cuda = target_r
    
    loss = (
        (
            pred[~torch.isnan(target)]
            - target_cuda[~torch.isnan(target)]
        )
        ** 2
    ).mean()
    #warn if loss is nan
    if torch.isnan(loss):
        print("nan loss")

    return loss
    
def genomic_structure_h1esc_loss(logit,y,ignore_index=-100):
    pred,_=logit
    target=y
    length=int(pred.shape[-1])
    #check if target is all nan
    if torch.isnan(target).all():
        #give random positive value to avoid nan loss
        target=torch.rand_like(target)
    
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res1000.npy")

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.nanmean(
                torch.nanmean(torch.reshape(target, (target.shape[0], length, 1, length, 1)), axis=4),
                axis=2,
            )
    target_r = torch.log(((target + eps) / (normmat_r + eps)))
    target_cuda = target_r
    
    loss = (
        (
            pred[~torch.isnan(target)]
            - target_cuda[~torch.isnan(target)]
        )
        ** 2
    ).mean()
    #warn if loss is nan
    if torch.isnan(loss):
        print("nan loss")

    return loss

def genomic_structure_hctnoc_loss(logit,y,ignore_index=-100):
    pred,_=logit
    target=y
    length=int(pred.shape[-1])
    #check if target is all nan
    if torch.isnan(target).all():
        #give random positive value to avoid nan loss
        target=torch.rand_like(target)
    
    normmat_bydist = np.exp(
        np.load("/orca/resources/resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool.expected.res1000.npy")

    )[:length]
    normmat = normmat_bydist[np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])]
    normmat_r = torch.from_numpy(normmat).float().cuda()
    eps = torch.min(normmat_r)
    target = torch.nanmean(
                torch.nanmean(torch.reshape(target, (target.shape[0], length, 1, length, 1)), axis=4),
                axis=2,
            )
    target_r = torch.log(((target + eps) / (normmat_r + eps)))
    target_cuda = target_r
    
    loss = (
        (
            pred[~torch.isnan(target)]
            - target_cuda[~torch.isnan(target)]
        )
        ** 2
    ).mean()
    #warn if loss is nan
    if torch.isnan(loss):
        print("nan loss")

    return loss

def soft_cross_entropy(logits, y, label_smoothing=0.0):
    logits = logits.view(-1, logits.shape[-1])
    # target is now 2d (no target flattening)
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()

def accuracy_multilabel(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    #logit into 0-1
    logits = torch.sigmoid(logits)
    y = y.view(-1, y.shape[-1])
    preds = (logits > 0.5).int()
    return torch.eq(preds,y).float().mean()

def accuracy_ignore_index(logits, y, ignore_index=-100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    accuracy = tm_f.classification.accuracy(preds, y, 'multiclass', num_classes=num_classes, ignore_index=ignore_index, average='micro')
    return accuracy


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()


def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    # y = y.view(-1)
    # y_hat = torch.argmax(logits, dim=-1)
    y_hat = (logits > 0.5).int()
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")

def pr_auc_mean(logits,y):
    # compute metrics based on stored labels, predictions, ...
    metrics = {}
    y, p = y, torch.sigmoid(logits)
    #convert three dimension into 2 dimension
    y=y.reshape(-1,y.shape[-1]).cpu().numpy()
    p=p.reshape(-1,p.shape[-1]).cpu().detach().numpy()
    # compute pr-auc for each class independetly
    for label in [0, 1, 2]:
        y_label = y[:, label]
        p_label = p[:, label]
        if not np.isnan(p_label).any():
            try:
                pr_auc = roc_auc_score(y_label, p_label)
            except ValueError:
                #calculate accurate rate for this label
                #convert p_label to 0-1
                p_label = (p_label > 0.5).astype(int)
                accurate_rate = (y_label == p_label).sum() / len(y_label)
                #if accurate rate is 1, set pr_auc to 1
                pr_auc = accurate_rate
        else:
            pr_auc = np.nan
        # to be compatible with sklearn 1.1+
        metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
    metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2']) / 2
    return metrics['pr_auc_mean']

def pr_auc_1(logits,y):
    # compute metrics based on stored labels, predictions, ...
    metrics = {}
    y, p = y, torch.sigmoid(logits)
    #convert three dimension into 2 dimension
    y=y.reshape(-1,y.shape[-1]).cpu().numpy()
    p=p.reshape(-1,p.shape[-1]).cpu().detach().numpy()
    # compute pr-auc for each class independetly
    for label in [0, 1, 2]:
        y_label = y[:, label]
        p_label = p[:, label]
        if not np.isnan(p_label).any():
            try:
                pr_auc = roc_auc_score(y_label, p_label)
            except ValueError:
                #calculate accurate rate for this label
                #convert p_label to 0-1
                p_label = (p_label > 0.5).astype(int)
                accurate_rate = (y_label == p_label).sum() / len(y_label)
                #if accurate rate is 1, set pr_auc to 1
                pr_auc = accurate_rate
        else:
            pr_auc = np.nan
        # to be compatible with sklearn 1.1+
        metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
    metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2']) / 2
    return metrics['pr_auc_1']

def pr_auc_2(logits,y):
    # compute metrics based on stored labels, predictions, ...
    metrics = {}
    y, p = y, torch.sigmoid(logits)
    #convert three dimension into 2 dimension
    y=y.reshape(-1,y.shape[-1]).cpu().numpy()
    p=p.reshape(-1,p.shape[-1]).cpu().detach().numpy()
    # compute pr-auc for each class independetly
    for label in [0, 1, 2]:
        y_label = y[:, label]
        p_label = p[:, label]
        if not np.isnan(p_label).any():
            try:
                pr_auc = roc_auc_score(y_label, p_label)
            except ValueError:
                #calculate accurate rate for this label
                #convert p_label to 0-1
                p_label = (p_label > 0.5).astype(int)
                accurate_rate = (y_label == p_label).sum() / len(y_label)
                #if accurate rate is 1, set pr_auc to 1
                pr_auc = accurate_rate
        else:
            pr_auc = np.nan
        # to be compatible with sklearn 1.1+
        metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
    metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2']) / 2
    return metrics['pr_auc_2']

def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.cpu().numpy()
    # y = y.view(-1)
    # y_hat = torch.argmax(logits, dim=-1)
    logits = torch.sigmoid(logits)
    y_hat = (logits > 0.5).int().cpu().numpy()
    metrics = {}
    f1 = np.zeros(y.shape[-1], dtype=np.float32)
    for i in range(y.shape[-1]):
        try:
            f1[i] = f1_score(y[:, i], y_hat[:, i],average='micro')
        except ValueError:
            pass
    metrics['TF_median_f1'] = np.median(f1[125:125 + 690])
    metrics['DHS_median_f1'] = np.median(f1[:125])
    metrics['HM_median_f1'] = np.median(f1[125 + 690:125 + 690 + 104])
    metrics['mean_f1'] = (metrics['TF_median_f1'] + metrics['DHS_median_f1'] + metrics['HM_median_f1']) / 3.0
    return metrics['mean_f1']


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")

def spearmanr(logits,y):
    #compute spearmanr correlation for each class
    output={}
    logits = logits.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    metrices = []
    for i in range(logits.shape[0]):
        spearmanrs = stats.spearmanr(logits[i, :], y[i, :])[0]
        metrices.append(spearmanrs)
    spearmanrs=np.nanmean(metrices,axis=0)
    output['spearmanr']=spearmanrs
    # for i in range(logits.shape[1]):
    #     spearmanrs = stats.spearmanr(logits[:, i], y[:, i])[0]
    #     metrices.append(spearmanrs)
    # spearmanrs=np.nanmean(metrices,axis=0)
    # output['spearmanr_batch']=spearmanrs
    return output
def R2(logits,y):
    from sklearn.metrics import r2_score
    #compute R2 for each class
    logits = logits.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    metrices = []
    for i in range(logits.shape[0]):
        r2s = r2_score(y[i, :], logits[i, :])
        metrices.append(r2s)
    r2s=np.nanmean(metrices,axis=0)
    return r2s
def roc_auc_macro(logits, y):
    # logits = logits.view(
    #     -1, logits.shape[-1]
    # ).detach()  # KS: had to add detach to eval while training
    # # y = y.view(-1)
    # # return roc_auc_score(
    # #     y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
    # # )
    # metrics = {}
    # logits =torch.sigmoid(logits)
    # aucs = np.zeros(y.shape[-1], dtype=np.float32)
    # y=y.cpu().numpy()
    # logits=logits.cpu().numpy()
    # for i in range(y.shape[-1]):
    #     try:
    #         aucs[i] = roc_auc_score(y[:, i], logits[:, i],average='weighted')
    #     except ValueError:
    #         pass
    # metrics['TF_median_auc'] = np.median(aucs[125:125 + 690])
    # metrics['DHS_median_auc'] = np.median(aucs[:125])
    # metrics['HM_median_auc'] = np.median(aucs[125 + 690:125 + 690 + 104])
    # metrics['mean_auc'] = (metrics['TF_median_auc'] + metrics['DHS_median_auc'] + metrics['HM_median_auc']) / 3.0
    # return metrics
    auc_roc=MultilabelAUROC(num_labels=919,average='macro',thresholds=None)
    value=auc_roc(logits,y)
    return value


def roc_auc_micro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    # y = y.view(-1)
    # return roc_auc_score(
    #     y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
    # )
    metrics = {}
    logits =torch.sigmoid(logits)
    aucs = np.zeros(y.shape[-1], dtype=np.float32)
    y=y.cpu().numpy()
    logits=logits.cpu().numpy()
    for i in range(y.shape[-1]):
        try:
            aucs[i] = roc_auc_score(y[:, i], logits[:, i])
        except ValueError:
            aucs[i] = 0
    metrics['TF_median_auc'] = np.median(aucs[125:125 + 690])
    metrics['DHS_median_auc'] = np.median(aucs[:125])
    metrics['HM_median_auc'] = np.median(aucs[125 + 690:125 + 690 + 104])
    value= (metrics['TF_median_auc'] + metrics['DHS_median_auc'] + metrics['HM_median_auc']) / 3.0
    return value


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    # if len(y.shape) < len(outs.shape):
    #     assert outs.shape[-1] == 1
    #     outs = outs.squeeze(-1)
    if len_batch is None:
        # return F.mse_loss(outs, y)
        loss = (
                            (
                                outs[~torch.isnan(y)]
                                - y[~torch.isnan(y)]
                            )
                            ** 2
                        ).mean()
        #check if y include nan
        if torch.isnan(outs).any():
            nan_indices_outs=torch.nonzero(torch.isnan(outs), as_tuple=False)
            print(nan_indices_outs)
        
        if torch.isnan(y).any():
            nan_indices=torch.nonzero(torch.isnan(y), as_tuple=False)
            print(nan_indices)
        return loss
    else:
        # Computes the loss of the first `lens` items in the batches
        # TODO document the use case of this
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)

def forecast_rmse(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    return torch.sqrt(F.mse_loss(outs, y, reduction='none').mean(1)).mean()

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()
def pearsonr_cage(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    metrics = []
    outs=outs.detach()
    # for i in range(50):
    #     y_true = y[:, :,i].cpu().numpy()
    #     outs_i = outs[:, :,i].cpu().numpy()
    
    #     r = stats.pearsonr(y_true.flatten(), outs_i.flatten())[0]
    #     metrics.append(r)
    
    for i in range(outs.shape[-1]):
        for j in range(outs.shape[0]):
            y_true = y[j, :,i].cpu().numpy()
            outs_i = outs[j, :,i].cpu().numpy()
            y_mean = y_true.mean()
            outs_mean = outs_i.mean()
            y_true=y_true.flatten()
            outs_i=outs_i.flatten()
        
            r=stats.pearsonr(y_true, outs_i)[0]
            metrics.append(r)
    #output non nan mean of metrics
    output=np.nanmean(metrics)

    # x_centered = outs - outs.mean(dim = 1, keepdim = True)
    # y_centered = y - y.mean(dim = 1, keepdim = True)
    # output=F.cosine_similarity(x_centered, y_centered, dim = 1).mean()

    return output

def pearsonr_1(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    metrics = {}
    outs=outs.detach()
    for i, label in enumerate(['dev', 'hk']):
        y_true = y[:, i].cpu().numpy()
        p = outs[:, i].cpu().numpy()
        r = stats.pearsonr(y_true, p)[0]
        metrics[f'pearsonr_{label}'] = r
        metrics[f'pearsonr2_{label}'] = r ** 2
    metrics['pearsonr'] = (metrics['pearsonr_dev'] + metrics['pearsonr_hk']) / 2
    return metrics


def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)


# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "padded_cross_entropy": padded_cross_entropy,
    "binary_accuracy": binary_accuracy,
    "precision": MulticlassPrecision,
    "precision_per_class": PrecisionPerClass,
    "recall": MulticlassRecall,
    "recall_per_class": RecallPerClass,
    "accuracy": accuracy,
    "accuracy_per_class": AccuracyPerClass,
    "accuracy_ignore_index": accuracy_ignore_index,
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    'accuracy_multilabel': accuracy_multilabel,
    "mcc": mcc,
    "mse": mse,
    "mae": mae,
    "forecast_rmse": forecast_rmse,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
    "soft_cross_entropy": soft_cross_entropy,  # only for pytorch 1.10+
    "student_t": student_t_loss,
    "gaussian_ll": gaussian_ll_loss,
    "genomic_structure_hff_loss": genomic_structure_hff_loss,
    "genomic_structure_h1esc_loss": genomic_structure_h1esc_loss,
    "genomic_structure_hctnoc_loss":genomic_structure_hctnoc_loss,
    "genomic_structure_mse_1d":genomic_structure_mse_1d,
    "genomic_structure_hff_corr":genomic_structure_hff_corr,
    "genomic_structure_h1esc_corr":genomic_structure_h1esc_corr,
    "genomic_structure_hctnoc_corr":genomic_structure_hctnoc_corr,
    
    'pearsonr': pearsonr_1,
    # 'pearsonr': PearsonrPerClass,
    'pr_auc_mean':pr_auc_mean,
    'pr_auc_1':pr_auc_1,
    'pr_auc_2':pr_auc_2,
    'pearsonr_cage':pearsonr_cage,
    'poisson_loss':poisson_loss,
    'spearmanr': spearmanr,
    'R2':R2,
}

loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9

