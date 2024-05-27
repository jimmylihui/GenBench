


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200


cd ..
cd ..

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=1024 \
        model.layer._name_=NT \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
        optimizer.lr=1e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_nucleotide-transformer-v2-500m-multi-species \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=768 \
        model.layer._name_=NT \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
        optimizer.lr=1e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_nucleotide-transformer-v2-250m-multi-species \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=512 \
        model.layer._name_=NT \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
        optimizer.lr=1e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_nucleotide-transformer-v2-100m-multi-species \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=512 \
        model.layer._name_=NT \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
        optimizer.lr=1e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_nucleotide-transformer-v2-50m-multi-species \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=NT \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"


    


