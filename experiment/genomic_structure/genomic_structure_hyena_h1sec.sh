


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200

cd ..
cd ..

python -m train experiment=hg38/genomic_structure \
    model.d_model=256 \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
    dataset.dataset_name=h1esc \
    optimizer.lr=1e-4 \
    wandb.mode=offline \
    dataset.tokenizer_name=hyena \
    model.layer._name_=hyena \
    dataset.max_length=6000 \
    dataset.total_size=10000 \
    trainer.devices=4 \
    dataset.batch_size=5 \
    train.global_batch_size=15 \
    wandb.id=Genomic_structure_h1esc_hyenadna-large-1m-seqlen_6k \
    callbacks.early_stopping.patience=10 \
    trainer.max_epochs=1000 \
    task.loss=genomic_structure_h1esc_loss \
    task.metrics=genomic_structure_h1esc_corr \
    callbacks.early_stopping.monitor=val/genomic_structure_h1esc_corr \
    callbacks.early_stopping.mode=max \
    train.monitor=val/genomic_structure_h1esc_corr \
    callbacks.model_checkpoint.mode=max \
    decoder._name_=sequence_structure \
    dataset.return_mask=False \
    dataset.dataset_name=h1esc 

# python -m train experiment=hg38/genomic_structure \
#     model.d_model=256 \
#     train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
#     dataset.dataset_name=h1sec \
#     optimizer.lr=1e-5 \
#     wandb.mode=offline \
#     dataset.tokenizer_name=hyena \
#     model.layer._name_=hyena \
#     dataset.max_length=6000 \
#     dataset.total_size=1000 \
#     trainer.devices=5 \
#     dataset.batch_size=1 \
#     train.global_batch_size=15 \
#     wandb.id=Genomic_structure_hyenadna-medium-450k-seqlen_6k \
#     callbacks.early_stopping.patience=10 \
#     trainer.max_epochs=100 \
#     callbacks.early_stopping.monitor=val/genomic_structure_corr \
#     callbacks.early_stopping.mode=max \
#     train.monitor=val/genomic_structure_corr \
#     decoder._name_=sequence_structure

# python -m train experiment=hg38/genomic_structure \
#     model.d_model=256 \
#     train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
#     dataset.dataset_name=h1sec \
#     optimizer.lr=1e-5 \
#     wandb.mode=offline \
#     dataset.tokenizer_name=hyena \
#     model.layer._name_=hyena \
#     dataset.max_length=6000 \
#     dataset.total_size=1000 \
#     trainer.devices=5 \
#     dataset.batch_size=1 \
#     train.global_batch_size=15 \
#     wandb.id=Genomic_structure_hyenadna-medium-160k-seqlen_6k \
#     callbacks.early_stopping.patience=10 \
#     trainer.max_epochs=100 \
#     callbacks.early_stopping.monitor=val/genomic_structure_corr \
#     callbacks.early_stopping.mode=max \
#     train.monitor=val/genomic_structure_corr \
#     decoder._name_=sequence_structure

# python -m train experiment=hg38/genomic_structure \
#     model.d_model=256 \
#     train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
#     dataset.dataset_name=h1sec \
#     optimizer.lr=1e-5 \
#     wandb.mode=offline \
#     dataset.tokenizer_name=hyena \
#     model.layer._name_=hyena \
#     dataset.max_length=6000 \
#     dataset.total_size=1000 \
#     trainer.devices=5 \
#     dataset.batch_size=1 \
#     train.global_batch_size=15 \
#     wandb.id=Genomic_structure_hyenadna-hyenadna-small-32k-seqlen-6k \
#     callbacks.early_stopping.patience=10 \
#     trainer.max_epochs=100 \
#     callbacks.early_stopping.monitor=val/genomic_structure_corr \
#     callbacks.early_stopping.mode=max \
#     train.monitor=val/genomic_structure_corr \
#     decoder._name_=sequence_structure


    


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=100000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_250k callbacks.early_stopping.patience=100 trainer.max_epochs=200


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-4 wandb.mode=offline dataset.max_length=450000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_450k callbacks.early_stopping.patience=100 trainer.max_epochs=200