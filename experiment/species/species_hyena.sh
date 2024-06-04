



cd ..
cd ..



for length in 512 1000 2000 3000 30000; do
    python -m train experiment=hg38/species \
        model.d_model=256 \
        train.pretrained_model_path=/weight/hyenadna/hyenadna-large-1m-seqlen \
        optimizer.lr=6e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=/weight/hyenadna/hyenadna-large-1m-seqlen \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=12 \
        train.global_batch_size=120 \
        wandb.id=species_hyenadna-large-1m-seqlen_{$length} \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10
    
    python -m train experiment=hg38/species \
        model.d_model=256 \
        train.pretrained_model_path=/weight/hyenadna/hyenadna-medium-450k-seqlen \
        optimizer.lr=6e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=/weight/hyenadna/hyenadna-medium-450k-seqlen \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=12 \
        train.global_batch_size=120 \
        wandb.id=species_hyenadna-medium-450k-seqlen_{$length} \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10

    
    python -m train experiment=hg38/species \
        model.d_model=256 \
        train.pretrained_model_path=/weight/hyenadna/hyenadna-medium-160k-seqlen \
        optimizer.lr=6e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=/weight/hyenadna/hyenadna-medium-160k-seqlen \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=12 \
        train.global_batch_size=120 \
        wandb.id=species_hyenadna-medium-160k-seqlen_{$length} \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10
    
    python -m train experiment=hg38/species \
        model.d_model=256 \
        train.pretrained_model_path=/weight/hyenadna/hyenadna-small-32k-seqlen \
        optimizer.lr=6e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=/weight/hyenadna/hyenadna-small-32k-seqlen \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=12 \
        train.global_batch_size=120 \
        wandb.id=species_hyenadna-small-32k-seqlen_{$length} \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10
    
done
    


