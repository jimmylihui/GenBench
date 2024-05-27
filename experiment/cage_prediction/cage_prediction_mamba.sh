cd ..
cd ..

python -m train experiment=hg38/cage_prediction \
    model.layer._name_=mamba \
    model.d_model=256 \
    task._name_=multilabel_regression \
    task.loss=mse \
    task.metrics=pearsonr_cage \
    callbacks.early_stopping.patience=5 \
    callbacks.early_stopping.monitor=val/pearsonr_cage  \
    callbacks.early_stopping.mode=max \
    decoder._name_=sequence_cage \
    dataset.max_length=2048 \
    dataset.tokenizer_name=mamba \
    optimizer.lr=1e-4 \
    dataset.return_mask=false \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=30 \
    train.global_batch_size=128 \
    wandb.id=cage_prediction_hyena_2048 \
    dataset.dataset_name=cage_prediction 
    