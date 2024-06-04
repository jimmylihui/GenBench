





cd ..
cd ..
 
python -m train experiment=hg38/genomic_structure \
    model.d_model=768 \
    train.pretrained_model_path=weight/genalm/gena-lm-bigbird-base-t2t \
    dataset.dataset_name=hff \
    optimizer.lr=1e-5 \
    wandb.mode=offline \
    dataset.tokenizer_name=genalm \
    model.layer._name_=genalm \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=5 \
    dataset.batch_size=1 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_genalm_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.monitor=val/genomic_structure_hff_corr \
    callbacks.early_stopping.mode=max \
    dataset.return_mask=True \
    train.monitor=val/genomic_structure_hff_corr \
    decoder._name_=sequence_structure

    


