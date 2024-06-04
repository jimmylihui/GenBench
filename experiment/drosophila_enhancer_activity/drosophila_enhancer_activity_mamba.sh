





cd ..
cd ..


python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        model.layer._name_=mamba \
        train.pretrained_model_path=weight/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        optimizer.lr=1e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=mamba \
        dataset.tokenizer_path=weight/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr" \
        dataset.return_mask=False

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        model.layer._name_=mamba \
        train.pretrained_model_path=weight/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
        optimizer.lr=1e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=mamba \
        dataset.tokenizer_path=weight/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr" \
        dataset.return_mask=False

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        model.layer._name_=mamba \
        train.pretrained_model_path=weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        optimizer.lr=1e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=mamba \
        dataset.tokenizer_path=weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr" \
        dataset.return_mask=False


python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        model.layer._name_=mamba \
        train.pretrained_model_path=weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        optimizer.lr=1e-4 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=mamba \
        dataset.tokenizer_path=weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr" \
        dataset.return_mask=False


    


