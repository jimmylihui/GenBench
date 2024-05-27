


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200

cd ..
cd ..

 
python -m train experiment=hg38/chromatin_profile \
        model.d_model=256 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
        optimizer.lr=6e-4 \
        wandb.mode=offline \
        dataset.max_length=512 \
        trainer.devices=5 \
        dataset.batch_size=200 \
        wandb.id=chromatin_profile_hyenadna-large-1m-seqlen_512 \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=250 \
        dataset.tokenizer_name=hyena \
        callbacks.early_stopping.monitor="test/f1_macro" \
        callbacks.model_checkpoint.monitor="test/f1_macro" \
        callbacks.model_checkpoint.filename="test/f1_macro" 

    


