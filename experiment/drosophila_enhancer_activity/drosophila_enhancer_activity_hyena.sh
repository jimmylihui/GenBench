




cd ..
cd ..


python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        train.pretrained_model_path=weight/hyenadna/hyenadna-large-1m-seqlen \
        optimizer.lr=6e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=4 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_hyenadna-large-1m-seqlen \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        train.pretrained_model_path=weight/hyenadna/hyenadna-medium-450k-seqlen \
        optimizer.lr=6e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=4 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_hyenadna-medium-450k-seqlen \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        train.pretrained_model_path=weight/hyenadna/hyenadna-medium-160k-seqlen \
        optimizer.lr=6e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=4 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_hyenadna-medium-160k-seqlen \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        train.pretrained_model_path=weight/hyenadna/hyenadna-small-32k-seqlen \
        optimizer.lr=6e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=4 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_hyenadna-small-32k-seqlen \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=hyena \
        dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"


    


