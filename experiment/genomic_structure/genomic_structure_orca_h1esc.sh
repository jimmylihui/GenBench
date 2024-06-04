





cd ..
cd ..
 
python -m train experiment=hg38/genomic_structure_orca \
    model.d_model=768 \
    train.pretrained_model_path=orca \
    dataset.dataset_name=h1esc \
    optimizer.lr=2e-3 \
    optimizer=adamw \
    wandb.mode=offline \
    dataset.tokenizer_name=orca \
    model.layer._name_=orca \
    dataset.max_length=6000 \
    dataset.total_size=1000 \
    trainer.devices=1 \
    dataset.batch_size=15 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_h1esc_orca_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=100 \
    task.loss=genomic_structure_h1esc_loss \
    task.metrics=genomic_structure_h1esc_loss \
    callbacks.early_stopping.monitor=val/genomic_structure_h1esc_loss \
    callbacks.early_stopping.mode=min \
    train.monitor=val/genomic_structure_h1esc_loss \
    callbacks.model_checkpoint.mode=min \
    optimizer=sgd \
    dataset.return_mask=False 


    


