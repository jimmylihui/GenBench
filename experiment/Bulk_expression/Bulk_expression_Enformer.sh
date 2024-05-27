cd ..
cd ..

python -m train experiment=hg38/Bulk_expression \
    model.layer._name_=Enformer \
    dataset.dataset_name=bulk_rna_expression \
    model.d_model=3072 \
    task._name_=multilabel_regression \
    task.loss=mse \
    callbacks.early_stopping.patience=3 \
    decoder._name_=sequence_bulk \
    dataset.max_length=2048 \
    dataset.tokenizer_name=Enformer \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/dnabert2 \
    optimizer.lr=1e-3 \
    dataset.return_mask=False \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/dnabert2 \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=10 \
    train.global_batch_size=128 \
    wandb.id=Bulk_expression_Enformer_2048
    