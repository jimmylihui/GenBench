




cd ..
cd ..

 
python -m train experiment=hg38/genomic_structure \
    model.d_model=768 \
    train.pretrained_model_path=weight/dnabert2 \
    dataset.dataset_name=hff \
    optimizer.lr=1e-5 \
    wandb.mode=offline \
    dataset.tokenizer_name=bert2 \
    model.layer._name_=bert2 \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=4 \
    dataset.batch_size=1 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_hff_bert2_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.monitor=val/genomic_structure_hff_corr \
    callbacks.early_stopping.mode=max \
    train.monitor=val/genomic_structure_corr \
    decoder._name_=sequence_structure \
    dataset.return_mask=True \
    dataset.dataset_name=hff

    


