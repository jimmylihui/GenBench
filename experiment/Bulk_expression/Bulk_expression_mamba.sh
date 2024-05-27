cd ..
cd ..

python -m train experiment=hg38/Bulk_expression \
    model.layer._name_=mamba \
    dataset.dataset_name=bulk_rna_expression \
    model.d_model=256 \
    task._name_=multilabel_regression \
    task.loss=mse \
    callbacks.early_stopping.patience=5 \
    decoder._name_=sequence_bulk \
    dataset.max_length=2048 \
    dataset.tokenizer_name=mamba \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/tokenizer/mamba \
    optimizer.lr=1e-4 \
    dataset.return_mask=false \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=30 \
    train.global_batch_size=128 \
    wandb.id=Bulk_expression_mamba_2048
    