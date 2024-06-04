cd ..
cd ..

python -m train experiment=hg38/Bulk_expression \
    model.layer._name_=hyena \
    dataset.dataset_name=bulk_rna_expression \
    model.d_model=256 \
    task._name_=multilabel_regression \
    task.loss=mse \
    callbacks.early_stopping.patience=5 \
    decoder._name_=sequence_bulk \
    dataset.max_length=2048 \
    dataset.tokenizer_name=hyena \
    dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
    optimizer.lr=3e-5 \
    dataset.return_mask=false \
    train.pretrained_model_path=weight/hyenadna/hyenadna-large-1m-seqlen \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=20 \
    train.global_batch_size=128 \
    wandb.id=Bulk_expression_hyena_2048 
    