from itertools import islice
from functools import partial
import os
import functools
# import json
# from pathlib import Path
# from pyfaidx import Fasta
# import polars as pl
# import pandas as pd
import torch
from random import randrange, random
import numpy as np
from pathlib import Path


from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded
from src.dataloaders.base import default_data_path
from selene_sdk.samplers.dataloader import SamplerDataLoader
"""

Genomic Benchmarks Dataset, from:
https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


"""


# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5

# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

#build a map from into to genmoic sequenc
map_to_genomic_sequence = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, 'must be one hot encoding with last dimension equal to 4'
    return torch.flip(one_hot, (-1, -2))





"""
This module provides the `SamplerDataLoader` and  `SamplerDataSet` classes,
which allow parallel sampling for any Sampler or FileSampler using
torch DataLoader mechanism.
"""

import  sys
import collections

import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

class Genomic_Structure_Dataset_Orca(data.Dataset):
    """
    This class provides a Dataset interface for a Sampler or FileSampler. 
    `_SamplerDataset` is intended to be used with `SamplerDataLoader`.
    
    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.
    """
    def __init__(self, sampler,total_size,max_length):
        super(Genomic_Structure_Dataset_Orca, self).__init__()
        self.sampler = sampler
        self.total_size=total_size
        self.max_length=max_length  
        self.dataloader = SamplerDataLoader(sampler, batch_size=1)
    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()
    
    def __getitem__(self, index):
        """
        Retrieve sample(s) from self.sampler. Only index length affects the 
        size the samples. The index values are not used.

        Parameters
        ----------
        index : int or any object with __len__ method implemented
            The size of index is used to determined size of the samples 
            to return.

        Returns
        ----------
        datatuple : tuple(numpy.ndarray, ...) or tuple(tuple(numpy.ndarray, ...), ...)
            A tuple containing the sampler.sample() output which can be a tuple 
            of arrays or a tuple of tuple of arrays (can be a mix of tuple and arrays). 
            The output dimension depends on the input of ` __getitem__`: if the
            index is an int the output is without the batch dimension. This fits
            the convention of most __getitem__ implementations and works with 
            DataLoader.
        """
        self.sampler.mode='train'
        if isinstance(index, int):
            batch_size = 1
            reduce_dim = True
        else: 
            batch_size = len(index)
            reduce_dim = False

        # sampled_data = self.sampler.sample(batch_size=batch_size)
        i=0
        for sample_data in self.dataloader:
            sampled_data=sample_data
            i=i+1
            if i==1:
                break
            
        #check if sampled_data[1] include nan,resample
        # while np.isnan(sampled_data[1]).any():
        #     sampled_data = self.sampler.sample(batch_size=batch_size)

        

        # seq_ids =torch.from_numpy(sampled_data[0]).squeeze(0)
        seq_ids=torch.squeeze(sampled_data[0])
        #convert seq_ids to float32
        seq_ids = seq_ids.float()
        # target = torch.from_numpy(sampled_data[1]).squeeze(0)
        target = torch.squeeze(sampled_data[1])
        target = target.float()

        #replace sample_data[0] with sequence

        #repeat until sampled_data is not nan

    
        
        return seq_ids, target

    def __len__(self):
        """
        Implementing __len__ is required by the DataLoader. So as a workaround,
        this returns `sys.maxsize` which is a large integer which should 
        generally prevent the DataLoader from reaching its size limit. 

        Another workaround that is implemented is catching the StopIteration 
        error while calling `next` and reinitialize the DataLoader.
        """
        return self.total_size


class Genomic_Structure_Dataset_Orca_valid(data.Dataset):
    """
    This class provides a Dataset interface for a Sampler or FileSampler. 
    `_SamplerDataset` is intended to be used with `SamplerDataLoader`.
    
    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.
    """
    def __init__(self, sampler,total_size,max_length):
        super(Genomic_Structure_Dataset_Orca_valid, self).__init__()
        self.sampler = sampler
        self.total_size=total_size
        self.max_length=max_length  
        self.sampler.mode = "validate"
        self.all_seqs=[]
        self.all_targets=[]
        self.dataloader = SamplerDataLoader(sampler,batch_size=1)
        i=0
        for sampled_data in self.dataloader:
            
            # sampled_data = self.sampler.sample(batch_size=1)
            
            # while np.isnan(sampled_data[1]).any():
            #     sampled_data = self.sampler.sample(batch_size=1)
            sequence=sampled_data[0]
            target=sampled_data[1]
            self.all_seqs.append(sequence)
            self.all_targets.append(target)
            i+=1
            if i==total_size:
                break
    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()
    
    def __getitem__(self, index):
        """
        Retrieve sample(s) from self.sampler. Only index length affects the 
        size the samples. The index values are not used.

        Parameters
        ----------
        index : int or any object with __len__ method implemented
            The size of index is used to determined size of the samples 
            to return.

        Returns
        ----------
        datatuple : tuple(numpy.ndarray, ...) or tuple(tuple(numpy.ndarray, ...), ...)
            A tuple containing the sampler.sample() output which can be a tuple 
            of arrays or a tuple of tuple of arrays (can be a mix of tuple and arrays). 
            The output dimension depends on the input of ` __getitem__`: if the
            index is an int the output is without the batch dimension. This fits
            the convention of most __getitem__ implementations and works with 
            DataLoader.
        """
        if isinstance(index, int):
            batch_size = 1
            reduce_dim = True
        else: 
            batch_size = len(index)
            reduce_dim = False

        x = self.all_seqs[index]
        y = self.all_targets[index]

        

        # seq_ids =torch.from_numpy(x).squeeze(0)
        seq_ids=torch.squeeze(x)
        #convert seq_ids to float32
        seq_ids = seq_ids.float()
        # target = torch.from_numpy(y).squeeze(0)
        target = torch.squeeze(y)
        target = target.float()

        #replace sample_data[0] with sequence

        #repeat until sampled_data is not nan

    
        
        return seq_ids, target

    def __len__(self):
        """
        Implementing __len__ is required by the DataLoader. So as a workaround,
        this returns `sys.maxsize` which is a large integer which should 
        generally prevent the DataLoader from reaching its size limit. 

        Another workaround that is implemented is catching the StopIteration 
        error while calling `next` and reinitialize the DataLoader.
        """
        return self.total_size




if __name__ == '__main__':
    """Quick test loading dataset.
    
    example
    python -m src.dataloaders.datasets.genomic_bench_dataset
    
    """

    max_length = 300  # max len of seq grabbed
    use_padding = True
    dest_path = "data/genomic_benchmark/"
    return_mask = True
    add_eos = True
    padding_side = 'right'    

    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length,
        add_special_tokens=False,
        padding_side=padding_side,
    )

    ds = GenomicBenchmarkDataset(
        max_length = max_length,
        use_padding = use_padding,
        split = 'train', # 
        tokenizer=tokenizer,
        tokenizer_name='char',
        dest_path=dest_path,
        return_mask=return_mask,
        add_eos=add_eos,
    )

    # it = iter(ds)
    # elem = next(it)
    # print('elem[0].shape', elem[0].shape)
    # print(elem)
    # breakpoint()