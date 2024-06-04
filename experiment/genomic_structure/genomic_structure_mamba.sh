




cd ..
cd ..
 
python -m train experiment=hg38/genomic_structure \
    model.d_model=256 \
    train.pretrained_model_path=weight/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
    dataset.dataset_name=h1sec \
    optimizer.lr=1e-4 \
    wandb.mode=offline \
    dataset.tokenizer_name=mamba \
    model.layer._name_=mamba \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=5 \
    dataset.batch_size=1 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_caduceus-ph_seqlen-131k_d_model-256_n_layer-16_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.monitor=val/genomic_structure_corr \
    callbacks.early_stopping.mode=max \
    train.monitor=val/genomic_structure_corr \
    decoder._name_=sequence_structure \
    dataset.return_mask=False

python -m train experiment=hg38/genomic_structure \
    model.d_model=256 \
    train.pretrained_model_path=weight/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
    dataset.dataset_name=h1sec \
    optimizer.lr=1e-4 \
    wandb.mode=offline \
    dataset.tokenizer_name=mamba \
    model.layer._name_=mamba \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=5 \
    dataset.batch_size=1 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_caduceus-ps_seqlen-131k_d_model-256_n_layer-16_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.monitor=val/genomic_structure_corr \
    callbacks.early_stopping.mode=max \
    train.monitor=val/genomic_structure_corr \
    decoder._name_=sequence_structure \
    dataset.return_mask=False

    


