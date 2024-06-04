
cd ..
cd ..

for dataset in  0_txt; do
    for lr in 1e-4; do
        for max_length in 512 1000 2000 3000 30000; do
            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=/weight/hyenadna/hyenadna-large-1m-seqlen \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=hyena \
                    optimizer.lr=$lr \
                    dataset.tokenizer_path=/weight/hyenadna/hyenadna-large-1m-seqlen \
                    dataset=promoter_prediction \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=10 \
                    train.global_batch_size=128 \
                    wandb.id=GUE_promoter_prediction_${dataset}_hyenadna-large-1m-seqlen_${max_length} \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=5 \
                    callbacks.early_stopping.mode=max 
            
            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=/weight/hyenadna/hyenadna-medium-450k-seqlen \
                    dataset.dataset_name=$dataset \
                    optimizer.lr=$lr \
                    dataset.tokenizer_name=hyena \
                    dataset.tokenizer_path=/weight/hyenadna/hyenadna-medium-450k-seqlen \
                    dataset=promoter_prediction \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=10 \
                    train.global_batch_size=128 \
                    wandb.id=GUE_promoter_prediction_${dataset}_hyenadna-medium-450k-seqlen_${max_length} \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=5 \
                    callbacks.early_stopping.mode=max 

            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=/weight/hyenadna/hyenadna-medium-160k-seqlen \
                    dataset.dataset_name=$dataset \
                    optimizer.lr=$lr \
                    dataset.tokenizer_name=hyena \
                    dataset.tokenizer_path=/weight/hyenadna/hyenadna-medium-160k-seqlen \
                    dataset=promoter_prediction \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=10 \
                    train.global_batch_size=128 \
                    wandb.id=GUE_promoter_prediction_${dataset}_hyenadna-medium-160k-seqlen_${max_length} \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=5 \
                    callbacks.early_stopping.mode=max 

            python -m train experiment=hg38/GUE \
                    model.d_model=256 \
                    train.pretrained_model_path=/weight/hyenadna/hyenadna-small-32k-seqlen \
                    dataset.dataset_name=$dataset \
                    optimizer.lr=$lr \
                    dataset.tokenizer_name=hyena \
                    dataset.tokenizer_path=weight/hyenadna/hyenadna-small-32k-seqlen \
                    dataset=promoter_prediction \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=10 \
                    train.global_batch_size=128 \
                    wandb.id=GUE_promoter_prediction_${dataset}_hyenadna-small-32k-seqlen_${max_length} \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=5 \
                    callbacks.early_stopping.mode=max 

        done
    done
done
