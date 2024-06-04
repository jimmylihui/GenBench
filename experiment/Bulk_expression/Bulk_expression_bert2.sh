cd ..
cd ..

python -m train experiment=hg38/Bulk_expression \
    model.layer._name_=bert2 \
    dataset.dataset_name=bulk_rna_expression \
    model.d_model=768 \
    task._name_=multilabel_regression \
    task.loss=mse \
    callbacks.early_stopping.patience=3 \
    decoder._name_=sequence_bulk \
    dataset.max_length=2048 \
    dataset.tokenizer_name=bert2 \
    dataset.tokenizer_path=weight/dnabert2 \
    optimizer.lr=1e-5 \
    dataset.return_mask=True \
    train.pretrained_model_path=weight/dnabert2 \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=5 \
    train.global_batch_size=128 \
    wandb.id=Bulk_expression_bert2_2048
    