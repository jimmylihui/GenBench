# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
from pathlib import Path
from typing import Any, List, Union
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from selene_utils2 import *
# import selene_sdk
# from selene_sdk.samplers.dataloader import SamplerDataLoader
from datasets import Dataset
#add path
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
# genomics datasets
from src.dataloaders.datasets.splicing_prediction_dataset import SplicingPredictionDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.datasets.genomic_bench_dataset import GenomicBenchmarkDataset
from src.dataloaders.datasets.drosophila_enhancer_activity_dataset import DrosophilaEnhancerActivityDataset
from src.dataloaders.datasets.nucleotide_transformer_dataset import NucleotideTransformerDataset
from src.dataloaders.datasets.chromatin_profile_dataset import ChromatinProfileDataset
from src.dataloaders.datasets.species_dataset import SpeciesDataset
from src.dataloaders.datasets.icl_genomics_dataset import ICLGenomicsDataset
from src.dataloaders.datasets.hg38_fixed_dataset import HG38FixedDataset
# from src.dataloaders.datasets.hg38_3_mer_tokenizer import DNATokenizer
from src.dataloaders.datasets.hg38_3_mer_tokenizer import ThreeMerTokenizer
from src.dataloaders.datasets.genslm_tokenizer import GenslmTokenizer
from src.dataloaders.datasets.genomic_bench_regression_dataset import GenomicBenchmarkDatasetRegression
from src.dataloaders.datasets.plant_genomic_bench_dataset import PlantGenomicBenchmarkDataset
from src.dataloaders.datasets.genomic_structure_dataset import Genomic_Structure_Dataset,Genomic_Structure_Dataset_valid
from src.utils.tokenizer_dna import DNATokenizer
from transformers import PreTrainedTokenizerFast
from src.dataloaders.datasets.genomic_structure_dataset_orca import Genomic_Structure_Dataset_Orca, Genomic_Structure_Dataset_Orca_valid
from src.dataloaders.datasets.genomic_long_range_benchmark import CagePredictionDataset,BulkRNAExpressionDataset,VariantEffectGeneExpression

# from src.utils.genslm.inference import GenSLM
"""

Dataloaders for genomics datasets, including pretraining and downstream tasks.  First works in HyenaDNA project, May 2023.

"""

class HG38(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    ###### very important to set this! ######
    _name_ = "hg38"  # this name is how the dataset config finds the right dataloader
    #########################################

    def __init__(self, bed_file, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, replace_N_token=False, pad_interval=False,l_output=5000,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.use_fixed_len_val = use_fixed_len_val
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval        
        self.l_output = l_output
        # handle if file paths are None (default paths)
        if self.bed_file is None:
            self.bed_file = default_data_path / self._name_ / 'human-sequences.bed'
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / 'hg38.ml.fa'

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            self.dataset_train.fasta.seqs.close()
            del self.dataset_train.fasta.seqs

        # delete old datasets to free memory
        if hasattr(self, 'dataset_test'):
            self.dataset_test.fasta.seqs.close()
            del self.dataset_test.fasta.seqs
    
        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(split=split,
                        bed_file=self.bed_file,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        shift_augs=None,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        replace_N_token=self.replace_N_token,
                        pad_interval=self.pad_interval)
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        if self.use_fixed_len_val:
            # we're placing the fixed test set in the val dataloader, for visualization!!!
            # that means we should track mode with test loss, not val loss

            # new option to use fixed val set
            print("Using fixed length val set!")
            # start end of chr14 and chrX grabbed from Enformer
            chr_ranges = {'chr14': [19726402, 106677047],
                            'chrX': [2825622, 144342320],
                            }

            self.dataset_val = HG38FixedDataset(
                chr_ranges=chr_ranges,
                fasta_file=self.fasta_file,
                max_length=self.max_length,
                pad_max_length=self.max_length,
                tokenizer=self.tokenizer,
                add_eos=True,
            )

        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet

class PlantGenomicBenchmark(HG38):
    _name_ = "plant_genomic_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry


        
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            PlantGenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class GenomicBenchmark(HG38):
    _name_ = "genomic_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            # if self.tokenizer.pad_token is None:
            #         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("**Using evo tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)


from datasets import load_dataset
class GenomicLongRangeBenchmark(HG38):
    _name_ = "genomic_long_range_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")

        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
    
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            # if self.tokenizer.pad_token is None:
            #         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("**Using evo tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'Enformer':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using Enformer tokenizer**")
        
        if self.dataset_name=='cage_prediction':
            self.dataset_train, self.dataset_val = [
                CagePredictionDataset(split=split,
                                    max_length=max_len,
                                    dataset_name=self.dataset_name,
                                    tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                    tokenizer_name=self.tokenizer_name,
                                    use_padding=self.use_padding,
                                    d_output=self.d_output,
                                    add_eos=self.add_eos,
                                    dest_path=self.dest_path,
                                    rc_aug=self.rc_aug,
                                    return_augs=False,
                                    return_mask=self.return_mask,
                )
                for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
                ]
        elif self.dataset_name=='bulk_rna_expression':
                self.dataset_train, self.dataset_val = [
                BulkRNAExpressionDataset(split=split,
                                    max_length=max_len,
                                    dataset_name=self.dataset_name,
                                    tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                    tokenizer_name=self.tokenizer_name,
                                    use_padding=self.use_padding,
                                    d_output=self.d_output,
                                    add_eos=self.add_eos,
                                    dest_path=self.dest_path,
                                    rc_aug=self.rc_aug,
                                    return_augs=False,
                                    return_mask=self.return_mask,
                )
                for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
            ]
        print('dataloader loading done')

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    
class EMP(HG38):
    _name_ = "EMP"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    


class promoter_prediction(HG38):
    _name_ = "promoter_prediction"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class GUE(HG38):
    _name_ = "GUE"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval) 

class mouse(HG38):
    _name_ = "mouse"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class prom(HG38):
    _name_ = "prom"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class splice(HG38):
    _name_ = "splice"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    
class splicing_prediction(HG38):
    _name_ = "splicing_prediction"
    l_output = 0  # need to set this for decoder to work correctly


    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,l_output=5000, tokenizer_path=None, *args, **kwargs):  

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.l_output=l_output
        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        elif self.tokenizer_name == 'spliceai':
            self.tokenizer=None
            print("**not Using tokenizer**")
        self.dataset_train, self.dataset_val = [
            SplicingPredictionDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
                                l_output=self.l_output,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    
class drosophila_enhancer_activity(HG38):
    _name_ = "drosophila_enhancer_activity"
    l_output = 0  # need to set this for decoder to work correctly


    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):  

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        elif self.tokenizer_name == 'deepstar':
            self.tokenizer=None
            print("**not Using tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            DrosophilaEnhancerActivityDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    
class tf(HG38):
    _name_ = "tf"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None, tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    
class virus(HG38):
    _name_ = "virus"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDataset(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class NucleotideTransformer(HG38):
    _name_ = "nucleotide_transformer"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, shuffle_eval=None, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_eval = shuffle_eval if shuffle_eval is not None else shuffle  # default is to use the same as train shuffle arg
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            NucleotideTransformerDataset(split=split,
                                max_length=max_len,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                dataset_name = self.dataset_name,
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval, shuffle=self.shuffle_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        # note: we're combining val/test into one
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval, shuffle=self.shuffle_eval)


class ChromatinProfile(HG38):
    _name_= 'chromatin_profile'
    l_output = 0  # need to set this for decoder to work correctly for seq level
    def __init__(self, data_path, ref_genome_path, ref_genome_version=None,
                 tokenizer_name=None, dataset_config_name=None, 
                 max_length=1000, d_output=2, rc_aug=False, add_eos=True, val_only=False,
                 batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None,
                 *args, **kwargs):
        
        self.data_path = data_path
        self.ref_genome_path = ref_genome_path
        self.ref_genome_version = ref_genome_version
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.add_eos = add_eos
        
        self.val_only=val_only
        
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.padding_side = 'left'

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.vocab_size = len(self.tokenizer)
        
        # Create all splits: torch datasets
        if self.val_only:
            splits=['val']*3
        else:
            splits=['train','val','test']
        self.dataset_train, self.dataset_val, self.dataset_test = [
            ChromatinProfileDataset(
                max_length=self.max_length,
                ref_genome_path = self.ref_genome_path,
                ref_genome_version = self.ref_genome_version,
                coords_target_path = f'{self.data_path}/{split}_{self.ref_genome_version}_coords_targets.csv',
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                use_padding=True,
            )
            for split in splits
        ]
        


class Species(HG38):
    _name_ = "species"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, species: list, species_dir: str, tokenizer_name=None, dataset_config_name=None, d_output=None, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, chromosome_weights='uniform', species_weights='uniform',
                total_size=None, task='species_classification', remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                tokenizer_path=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.species = species # list of species to load
        self.species_dir = species_dir
        self.chromosome_weights = chromosome_weights
        self.species_weights = species_weights
        self.total_size = total_size
        self.task = task
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test
        self.d_output = len(self.species)
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            for spec in list(self.dataset_train.fastas.keys()):
                for chromosome in list(self.dataset_train.fastas[spec].keys()):
                    self.dataset_train.fastas[spec][chromosome].close()
                    del self.dataset_train.fastas[spec][chromosome]
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SpeciesDataset(species=self.species,
                           species_dir=self.species_dir,
                           split=split,
                            max_length=max_len,
                            total_size=self.total_size * (1 if split == 'test' else (self.max_length_test + 2) // max_len), # See the same # of tokens every epoch across train/val/test
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            chromosome_weights=self.chromosome_weights,
                            species_weights=self.species_weights,
                            task=self.task,
                            remove_tail_ends=self.remove_tail_ends,
                            cutoff_train=self.cutoff_train,
                            cutoff_test=self.cutoff_test,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return
    

class GenomicBenchmark_regression(HG38):
    _name_ = "genomic_benchmark_regression"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False, total_size=None,
                fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size= total_size

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        
        # Create all splits: torch datasets (only train/test in this benchmark)
        self.dataset_train, self.dataset_val = [
            GenomicBenchmarkDatasetRegression(split=split,
                                max_length=max_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                rc_aug=self.rc_aug,
                                return_augs=False,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

class ICLGenomics(HG38):
    _name_ = "icl_genomics"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None, shots=1, label_to_token=None,
                add_eos=True, characters=None, padding_side='left', val_ratio=0.0005, val_split_seed=2357,
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=0,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None,
                use_shmem=True, *args, **kwargs):
        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.shots = shots  # num shots in ICL sample
        self.label_to_token = label_to_token  # this maps the label to a token in the vocab already, arbitrary
        self.add_eos = add_eos
        self.characters = list('ACTGN') if characters is None else characters
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.use_shmem = use_shmem
        # if self.use_shmem:
        #     assert cache_dir is not None

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=self.characters,
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )

        self.vocab_size = len(self.tokenizer)
        
        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val = [
            ICLGenomicsDataset(
                dataset_name=self.dataset_name,
                split=split,
                shots=self.shots,
                use_padding=self.use_padding,
                d_output=self.d_output,
                max_length=max_len,
                dest_path=self.dest_path,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                tokenizer_name=self.tokenizer_name,
                label_to_token=self.label_to_token,
                rc_aug=self.rc_aug,
                add_eos=self.add_eos,
            )
            for split, max_len in zip(['train', 'val'], [self.max_length, self.max_length_val])
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)


class HG38Fixed(HG38):
    _name_ = "hg38_fixed"

    """Just used for testing a fixed length, *non-overlapping* dataset for HG38."""

    def __init__(self, fasta_file=None, chr_ranges=None, pad_max_length=None, batch_size=32, 
                 max_length=None, num_workers=1, add_eos=True,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):
  
        self.fasta_file = fasta_file
        self.chr_ranges = chr_ranges
        self.max_length = max_length
        self.pad_max_length = pad_max_length
        self.add_eos = add_eos
        self.batch_size = batch_size
        self.batch_size_eval = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        if self.fasta_file is None:
            self.fasta_file = default_data_path / "hg38" / 'hg38.ml.fa'

        if self.chr_ranges is None:
            # start end of chr14 and chrX grabbed from Enformer
            self.chr_ranges = {'chr14': [19726402, 106677047],
                        'chrX': [2825622, 144342320],
                        }

    def setup(self, stage=None):

        # Create tokenizer
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length= self.max_length +  2,  # add 2 since default adds eos/eos tokens, crop later
            add_special_tokens=False,
        )

        # we only need one
        self.dataset_train = HG38FixedDataset(
            fasta_file=self.fasta_file,
            chr_ranges=self.chr_ranges,  # a dict of chr: (start, end) to use for test set
            max_length=self.max_length,
            pad_max_length=self.pad_max_length,
            tokenizer=tokenizer,
            add_eos=self.add_eos,
        )

        self.dataset_val = self.dataset_train
        self.dataset_test = self.dataset_train
    


class GenomicStructure(HG38):
    _name_ = "genomic_structure_dataset"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,total_size=None,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):
        self.max_length=max_length
        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.total_size=total_size  
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == '3_mer':
            self.tokenizer = ThreeMerTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            print("**Using 3-mer tokenizer**")
        
        elif self.tokenizer_name == 'hyena':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using hyena-dna tokenizer**")
        elif self.tokenizer_name == 'bert':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert tokenizer**")
        elif self.tokenizer_name == 'bert2':
            # self.tokenizer = DNATokenizer.from_pretrained(
            #     "dna3",
            #     do_lower_case=False,
            #     cache_dir='/usr/data/3-new-12w-0',
            # )
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using bert_2 tokenizer**")
        
        elif self.tokenizer_name=='NT':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using NT tokenizer**")
        elif self.tokenizer_name == 'genslm':
            model=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
            self.tokenizer=model.tokenizer
            print("**Using genslm tokenizer**")
            # self.tokenizer = GenslmTokenizer(
            #     characters=['A', 'C', 'G', 'T'],
            #     model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
            #     add_special_tokens=False,
            #     padding_side=self.padding_side,
            # )
            # print("**Using 3-mer tokenizer**")
        elif self.tokenizer_name == 'genalm':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            print("**Using genalm tokenizer**")
        elif self.tokenizer_name == 'evo':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using evo tokenizer**")
        
        elif self.tokenizer_name == 'mamba':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using mamba tokenizer**")
        elif self.tokenizer_name == 'CNN':
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,trust_remote_code=True)
            print("**Using CNN tokenizer**")
        if self.dataset_name=='h1esc':
            t = Genomic2DFeatures(
                [self.dest_path + "/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"],
                ["r1000"],
                (int(self.max_length/1000), int(self.max_length/1000)),
                cg=True,
            )
            sampler = RandomPositionsSamplerHiC(
                reference_sequence=MemmapGenome(
                    input_path=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    memmapfile=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                    blacklist_regions="hg38",
                ),
                target=t,
                # target_1d=MultibinGenomicFeatures(
                #     self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.gz",
                #     np.loadtxt(self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.features", str),
                #     1000,
                #     1000,
                #     (32, 10),
                #     mode="any",
                # ),
                features=["r1000"],
                test_holdout=["chr9", "chr10"],
                validation_holdout=["chr8"],
                sequence_length=self.max_length,
                position_resolution=1000,
                random_shift=100,
                random_strand=False,
                cross_chromosome=False,
            )
        elif self.dataset_name=='hff':
            t = Genomic2DFeatures(
                [self.dest_path + "/4DNFI643OYP9.rebinned.mcool::/resolutions/1000"],
                ["r1000"],
                (int(self.max_length/1000), int(self.max_length/1000)),
                cg=True,
            )
            sampler = RandomPositionsSamplerHiC(
                reference_sequence=MemmapGenome(
                    input_path=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    memmapfile=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                    blacklist_regions="hg38",
                ),
                target=t,
                # target_1d=MultibinGenomicFeatures(
                #     self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.gz",
                #     np.loadtxt(self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.features", str),
                #     1000,
                #     1000,
                #     (32, 10),
                #     mode="any",
                # ),
                features=["r1000"],
                test_holdout=["chr9", "chr10"],
                validation_holdout=["chr8"],
                sequence_length=self.max_length,
                position_resolution=1000,
                random_shift=100,
                random_strand=False,
                cross_chromosome=False,
            )
        # elif self.dataset_name=='hctnoc':
        #     t = Genomic2DFeatures(
        #         [self.dest_path + "/4DNFI643OYP9.rebinned.mcool::/resolutions/1000"],
        #         ["r1000"],
        #         (int(self.max_length/1000), int(self.max_length/1000)),
        #         cg=True,
        #     )
        #     sampler = RandomPositionsSamplerHiC(
        #         reference_sequence=MemmapGenome(
        #             input_path=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        #             memmapfile=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
        #             blacklist_regions="hg38",
        #         ),
        #         target=t,
        #         # target_1d=MultibinGenomicFeatures(
        #         #     self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.gz",
        #         #     np.loadtxt(self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.features", str),
        #         #     1000,
        #         #     1000,
        #         #     (32, 10),
        #         #     mode="any",
        #         # ),
        #         features=["r1000"],
        #         test_holdout=["chr9", "chr10"],
        #         validation_holdout=["chr8"],
        #         sequence_length=1000,
        #         position_resolution=1000,
        #         random_shift=100,
        #         random_strand=False,
        #         cross_chromosome=False,
        #     )
        sampler.mode = "train"
        self.dataset_train=Genomic_Structure_Dataset(sampler,self.total_size,self.tokenizer,self.tokenizer_name,self.max_length,self.return_mask,self.rc_aug)
        sampler.mode = "validate"
        self.dataset_val=Genomic_Structure_Dataset_valid(sampler,int(self.total_size/5),self.tokenizer,self.tokenizer_name,self.max_length,self.return_mask,self.rc_aug)
        

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    


class GenomicStructure_orca(HG38):
    _name_ = "genomic_structure_dataset_orca"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,total_size=None,
                fast_forward_epochs=None, fast_forward_batches=None,tokenizer_path=None, *args, **kwargs):
        self.max_length=max_length
        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path=tokenizer_path
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.total_size=total_size  
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

    
       
        if self.dataset_name=='h1esc':
            t = Genomic2DFeatures(
                [self.dest_path + "/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"],
                ["r1000"],
                (int(self.max_length/1000), int(self.max_length/1000)),
                cg=True,
            )
            sampler = RandomPositionsSamplerHiC(
                reference_sequence=MemmapGenome(
                    input_path=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    memmapfile=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                    blacklist_regions="hg38",
                ),
                target=t,
                # target_1d=MultibinGenomicFeatures(
                #     self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.gz",
                #     np.loadtxt(self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.features", str),
                #     1000,
                #     1000,
                #     (32, 10),
                #     mode="any",
                # ),
                features=["r1000"],
                test_holdout=["chr9", "chr10"],
                validation_holdout=["chr8"],
                sequence_length=self.max_length,
                position_resolution=1000,
                random_shift=100,
                random_strand=False,
                cross_chromosome=False,
            )
        elif self.dataset_name=='hff':
            t = Genomic2DFeatures(
                [self.dest_path + "/4DNFI643OYP9.rebinned.mcool::/resolutions/1000"],
                ["r1000"],
                (int(self.max_length/1000), int(self.max_length/1000)),
                cg=True,
            )
            sampler = RandomPositionsSamplerHiC(
                reference_sequence=MemmapGenome(
                    input_path=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    memmapfile=self.dest_path + "/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                    blacklist_regions="hg38",
                ),
                target=t,
                # target_1d=MultibinGenomicFeatures(
                #     self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.gz",
                #     np.loadtxt(self.dest_path + "/h1esc/h1esc.hg38.bed.sorted.features", str),
                #     1000,
                #     1000,
                #     (32, 10),
                #     mode="any",
                # ),
                features=["r1000"],
                test_holdout=["chr9", "chr10"],
                validation_holdout=["chr8"],
                sequence_length=self.max_length,
                position_resolution=1000,
                random_shift=100,
                random_strand=False,
                cross_chromosome=False,
            )
        sampler.mode = "train"
        self.dataset_train=Genomic_Structure_Dataset_Orca(sampler,self.total_size,self.max_length)
        sampler.mode = "validate"
        self.dataset_val=Genomic_Structure_Dataset_Orca_valid(sampler,int(self.total_size/5),self.max_length)
        

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    

if __name__ == '__main__':
    """Quick test using dataloader. Can't call from here though."""

    loader = GenomicStructure(
        dest_path='/usr/data/orca/resources',
        dataset_name='h1esc',
        tokenizer_name='char_level', max_length=2000
    )
    loader.setup()

    breakpoint()

    # it = iter(ds)
    # elem = next(it)
    # print(len(elem))
    # breakpoint()
