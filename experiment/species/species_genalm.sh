


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200

cd ..
cd ..


for length in 512 1000 2000 3000; do
    python -m train experiment=hg38/species \
        model.d_model=768 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
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


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=100000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_250k callbacks.early_stopping.patience=100 trainer.max_epochs=200


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-4 wandb.mode=offline dataset.max_length=450000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_450k callbacks.early_stopping.patience=100 trainer.max_epochs=200