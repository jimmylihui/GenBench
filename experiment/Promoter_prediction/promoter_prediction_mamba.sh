cd ..
cd ..



for dataset in  0_txt; do
    for lr in 1e-4; do
        for max_length in 512 1000 2000 3000 30000; do
            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=mamba \
                    model.layer._name_=mamba \
                    dataset.tokenizer_path=/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=125 \
                    wandb.id=promoter_prediction_${dataset}_mamba_${max_length}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3 \
                    dataset.return_mask=False
            
            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=mamba \
                    model.layer._name_=mamba \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.tokenizer_path=/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=125 \
                    wandb.id=promoter_prediction_${dataset}_mamba_${max_length}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3 \
                    dataset.return_mask=False

            
        done
    done
done