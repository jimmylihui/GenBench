cd ..
cd ..

python -m train experiment=hg38/cage_prediction \
    model.layer._name_=NT \
    model.d_model=1024 \
    task._name_=multilabel_regression \
    task.loss=poisson_loss \
    callbacks.early_stopping.patience=3 \
    callbacks.early_stopping.monitor=val/pearsonr_cage  \
    callbacks.early_stopping.mode=max \
    decoder._name_=sequence_cage \
    dataset.max_length=2048 \
    dataset.tokenizer_name=NT \
    optimizer.lr=3e-5 \
    dataset.return_mask=false \
    train.pretrained_model_path=weight/nt/nucleotide-transformer-v2-500m-multi-species \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=2 \
    train.global_batch_size=128 \
    wandb.id=cage_prediction_NT_2048 \
    dataset.dataset_name=cage_prediction 
    