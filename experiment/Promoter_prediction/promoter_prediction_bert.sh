
cd ..
cd ..

for dataset in  0_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/dnabert/dnabert3/3-new-12w-0 \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=bert \
                model.layer._name_=bert \
                optimizer.lr=$lr \
                dataset=promoter_prediction \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/dnabert/dnabert3/3-new-12w-0 \
                dataset.max_length=512 \
                trainer.devices=5 \
                dataset.batch_size=14 \
                train.global_batch_size=128 \
                wandb.id=promoter_prediction_${dataset}_bert \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 
    done
done