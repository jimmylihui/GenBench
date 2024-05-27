
cd ..
cd ..


for dataset in  0_txt; do
    for lr in 3e-5; do
        for max_length in 512 1000 2000 3000; do
            python -m train experiment=hg38/GUE \
                    model.d_model=768 \
                    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
                    dataset.dataset_name=$dataset \
                    dataset.tokenizer_name=genalm \
                    model.layer._name_=genalm \
                    optimizer.lr=$lr \
                    dataset=promoter_prediction \
                    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/tokenizer/gena-lm-bigbird-base-t2t \
                    dataset.max_length=$max_length \
                    trainer.devices=5 \
                    dataset.batch_size=1 \
                    train.global_batch_size=125 \
                    wandb.id=promoter_prediction_${dataset}_genalm_${max_length} \
                    wandb.mode=offline \
                    callbacks.early_stopping.patience=3
        done
    done
done