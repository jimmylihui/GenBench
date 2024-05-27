cd ..
cd ..

for dataset in  0_txt; do
    for lr in 1e-5; do
        for length in 512 1000 2000 3000; do
            python -m train experiment=hg38/GUE \
                    model.d_model=1024 \
                    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=NT \
                    model.layer._name_=NT \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.max_length=$length \
                    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=128 \
                    wandb.id=promoter_prediction_${dataset}_NT_${length}_nucleotide-transformer-v2-500m-multi-species \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3

            python -m train experiment=hg38/GUE \
                    model.d_model=768 \
                    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=NT \
                    model.layer._name_=NT \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                    dataset.max_length=$length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=128 \
                    wandb.id=promoter_prediction_${dataset}_NT_${length}_nucleotide-transformer-v2-250m-multi-species \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3
            
            python -m train experiment=hg38/GUE \
                    model.d_model=512 \
                    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=NT \
                    model.layer._name_=NT \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                    dataset.max_length=$length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=128 \
                    wandb.id=promoter_prediction_${dataset}_NT_${length}_nucleotide-transformer-v2-100m-multi-species \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3

            python -m train experiment=hg38/GUE \
                    model.d_model=512 \
                    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=NT \
                    model.layer._name_=NT \
                    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.max_length=$length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=128 \
                    wandb.id=promoter_prediction_${dataset}_NT_${length}_nucleotide-transformer-v2-50m-multi-species \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3
        done
    done
done