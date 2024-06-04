




cd ..
cd ..


for length in 512 1000 2000 3000; do
    python -m train experiment=hg38/species \
        model.d_model=768 \
        train.pretrained_model_path=/weight/genalm/gena-lm-bigbird-base-t2t \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_path=/weight/genalm/gena-lm-bigbird-base-t2t \
        dataset.tokenizer_name=genalm \
        model.layer._name_=genalm \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=4 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_genalm_$length \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10
done   


