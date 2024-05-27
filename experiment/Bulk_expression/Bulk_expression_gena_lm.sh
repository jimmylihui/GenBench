cd ..
cd ..

python -m train experiment=hg38/Bulk_expression \
    model.layer._name_=genalm \
    dataset.dataset_name=bulk_rna_expression \
    model.d_model=768 \
    task._name_=multilabel_regression \
    task.loss=mse \
    callbacks.early_stopping.patience=3 \
    decoder._name_=sequence_bulk \
    dataset.max_length=2048 \
    dataset.tokenizer_name=genalm \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
    optimizer.lr=1e-5 \
    dataset.return_mask=True \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=4 \
    train.global_batch_size=128 \
    wandb.id=Bulk_expression_genalm_2048
    