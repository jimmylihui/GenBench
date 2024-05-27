# Experiments

In this folder, we provide the scripts for the experiments conducted in the paper. The datasets used in the experiments include Chromatin_profile, drosophila_enhancer_activity, genomic_benchmark, genomic_structure, GUE, Promoter_prediction, Species, Splicing_prediction.

# How to run the experiments
Before running the experiments, please make sure you have downloaded the pretrained weight, we recommend you download the weight from following links and put them in the weight folder. Then edit the hyperparameters 'train.pretrained_model_path' and 'dataset.tokenizer_path' to the downloaded weight

- [hyena-dna](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen/tree/main)
- [DNABERT](https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view)
- [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- [The Nucleotide Transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- [GENA-LM](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-t2t)
- [Caduceus](https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16)


You can run the experiments by executing the corresponding scripts, we have specify the parameter to ensure you get the result in the paper. For example, to run the experiment on Chromatin_profile dataset with hyena-dna, you can enter the Chromatin_profile folder and run the script:.

```shell
bash chromatin_profile_hyena.sh
```

# Hyperparameters

To facilitate the modification of hyperparameters, we provide the important hyperparameters with descriptions.

- `experiment`: The default configuration of the experiment.
- `model.d_model`: The hidden size of the model.
- `model.layer._name_`: the name of the token mixing layer, such as 'hyena' and 'bert'.
- `train.pretrained_model_path`: the stored pretrained weight and model configuration. 
- `dataset.dataset_name`: the name of the sub dataset.
- `optimizer.lr`: the learning rate of the optimizer.
- `dataset`: the name of global dataset.
- `dataset.max_length`: the length of the input. If it is shorter than the maximum length of the sub dataset, the dataset will be truncated. If it is longer than the maximum length of the sub dataset, the dataset will be padded with 'N'.
- `wandb.mode`: the mode of the wandb logger, such as 'online' and 'offline'.
- `trainer.devices`: the number of GPUs used for training and inference.
- `dataset.batch_size`: the batch size for each device.
- `dataset.tokenizer_name`: the type of tokenizer.
- `train.global_batch_size`: the global bach size for each weight update. the framework will automatically split the global batch size into multiple sub-batches for each device with gradient accumulation.


# Result

For detail results, please refer to the paper.
