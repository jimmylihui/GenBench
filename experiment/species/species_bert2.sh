



cd ..
cd ..




 
for length in 512 1000 2000 3000; do
    python -m train experiment=hg38/species \
        model.d_model=768 \
        train.pretrained_model_path=/weight/dnabert2 \
        optimizer.lr=6e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=bert2 \
        model.layer._name_=bert2 \
        dataset.tokenizer_path=/weight/dnabert2 \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_bert2_$length \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10
done
    


