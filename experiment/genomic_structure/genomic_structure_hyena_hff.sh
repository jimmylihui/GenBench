




cd ..
cd ..

python -m train experiment=hg38/genomic_structure \
    model.d_model=256 \
    train.pretrained_model_path=weight/hyenadna/hyenadna-large-1m-seqlen \
    dataset.dataset_name=hff \
    optimizer.lr=1e-5 \
    wandb.mode=offline \
    dataset.tokenizer_name=hyena \
    model.layer._name_=hyena \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=4 \
    dataset.batch_size=3 \
    train.global_batch_size=15 \
    task.metrics=genomic_structure_hff_corr \
    wandb.id=Genomic_structure_hyenadna_hff-large-1m-seqlen_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.monitor=val/genomic_structure_hff_corr \
    callbacks.early_stopping.mode=max \
    train.monitor=val/genomic_structure_hff_corr \
    decoder._name_=sequence_structure \
    dataset.return_mask=False

