


# python -m train experiment=hg38/species  model.d_model=1024 model.layer._name_=NT train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species optimizer.lr=1e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_nucleotide-transformer-v2-500m-multi-species-seqlen_1k callbacks.early_stopping.patience=10 trainer.max_epochs=200 dataset.tokenizer_name=NT


cd ..
cd ..

for length in 512 1000 2000 3000; do
    python -m train experiment=hg38/species \
        model.d_model=1024 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
        model.layer._name_=NT \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_NT_{$length}_nucleotide-transformer-v2-500m-multi-species \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10

    python -m train experiment=hg38/species \
        model.d_model=768 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
        model.layer._name_=NT \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_NT_{$length}_nucleotide-transformer-v2-250m-multi-species \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10

    python -m train experiment=hg38/species \
        model.d_model=512 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
        model.layer._name_=NT \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_NT_{$length}_nucleotide-transformer-v2-100m-multi-species \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10

    python -m train experiment=hg38/species \
        model.d_model=512 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
        dataset.dataset_name=$dataset \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        dataset.tokenizer_name=NT \
        dataet.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
        model.layer._name_=NT \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.batch_size=1 \
        train.global_batch_size=120 \
        wandb.id=species_NT_{$length}_nucleotide-transformer-v2-50m-multi-species \
        callbacks.early_stopping.patience=100 \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=10

done
    


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=100000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_250k callbacks.early_stopping.patience=100 trainer.max_epochs=200


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-4 wandb.mode=offline dataset.max_length=450000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_450k callbacks.early_stopping.patience=100 trainer.max_epochs=200