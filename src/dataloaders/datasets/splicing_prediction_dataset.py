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

import h5py
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded
from src.dataloaders.base import default_data_path
import pandas as pd
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



class SplicingPredictionDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name="human_nontata_promoters",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        l_output=None,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.l_output = l_output  # needed for decoder to grab
        
        # if not is_downloaded(dataset_name, cache_path=dest_path):
        #     print("{} in {} is not downloaded".format(dataset_name, dest_path))
        # else:
        #     print("already downloaded {}-{}".format(split, dataset_name))

        
        

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        #read .npy file
            
        self.all_seqs = np.load(os.path.join(dest_path,  split + "_seq.npy"), allow_pickle=True)
        self.all_labels = np.load(os.path.join(dest_path,  split + "_target.npy"), allow_pickle=True)
        #get [:,0] as sequence and [:,1] as label

        

        #splite array of all_seqs into 2d array of labels based on \n separator
        # seq_split = np.char.split(self.all_seqs, sep='\n')
        # self.all_seqs = [[label] for label in seq_split]
        
                
        #print all_seqs length with split type
        print("SplicingPredictionDataset: {} sequences loaded for split {}".format(len(self.all_seqs), split))


        
        

        
        assert len(self.all_seqs) == len(
            self.all_labels
        ), "Number of targets does not match number of sequences"
        
    def __len__(self):
        return len(self.all_labels)
    def genomic_to_one_hot(self,genomic_sequence):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        one_hot = np.zeros((len(genomic_sequence), 4))
        for i, base in enumerate(genomic_sequence):
            if base in mapping:
                one_hot[i, mapping[base]] = 1
            else:
                # 如果碱基不是A、C、G、T或N，可以选择将其编码为全零向量或者平均分配概率
                one_hot[i, :] = 0.25  # 或者使用 np.full((5,), 0.2) 平均分配概率
        return one_hot
    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()
    
    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)
        if self.tokenizer_name == 'spliceai':
            seq = self.genomic_to_one_hot(x)[:self.max_length]
            seq_ids = torch.from_numpy(seq).float()
        else:
            if(self.tokenizer_name == 'genslm' or self.tokenizer_name == 'bert'):
                x = self.group_by_kmer(x, 3)
            if(self.tokenizer_name=='evo'):
                seq =self.tokenizer(x)
            else:
                seq = self.tokenizer(x,
                    add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                    padding="max_length" if self.use_padding else "do_not_pad",
                    max_length=self.max_length,
                    truncation=True,
                )
            seq_ids = seq["input_ids"]  # get input_ids

            
            seq_ids = torch.LongTensor(seq_ids)  # convert to tensor

        # need to wrap in list
        #list y to tensor

        # y = [y]
        # target = torch.LongTensor(y)  # offset by 1, includes eos
        target = torch.FloatTensor(y)  # offset by 1, includes eos
        target = target[:self.l_output,:]

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target






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