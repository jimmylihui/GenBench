


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200

cd ..
cd ..


 
python -m train experiment=hg38/chromatin_profile \
        model.d_model=768 \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/dnabert/dnabert3/3-new-12w-0 \
        optimizer.lr=3e-5 \
        wandb.mode=offline \
        model.layer._name_=bert \
        dataset.max_length=512 \
        trainer.devices=1 \
        dataset.batch_size=50 \
        wandb.id=chromatin_profile_bert-seqlen_512 \
        callbacks.early_stopping.patience=5 \
        trainer.max_epochs=100 \
        train.global_batch_size=250 \
        dataset.tokenizer_name=bert \
        callbacks.early_stopping.monitor="val/roc_auc_macro" \
        callbacks.model_checkpoint.monitor="val/roc_auc_macro" \
        callbacks.model_checkpoint.filename="val/roc_auc_macro" \

    


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=100000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_250k callbacks.early_stopping.patience=100 trainer.max_epochs=200


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-4 wandb.mode=offline dataset.max_length=450000 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=1 wandb.id=species_hyenadna-large-1m-seqlen_450k callbacks.early_stopping.patience=100 trainer.max_epochs=200