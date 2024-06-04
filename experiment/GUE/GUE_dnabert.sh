


cd ..
cd ..

for dataset in  H3_txt H3K4me1_txt H3K4me2_txt H3K4me3_txt H3K9ac_txt H3K14ac_txt H3K36me3_txt H3K79me3_txt H4_txt H4ac_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE  \
        model.d_model=768 \
        model.layer._name_=bert \
        train.pretrained_model_path=weight/dnabert/dnabert3/3-new-12w-0 \
        dataset.dataset_name=$dataset \
        optimizer.lr=$lr \
        dataset=EMP \
        dataset.max_length=128 \
        wandb.mode=offline \
        trainer.devices=4 \
        dataset.batch_size=16 \
        dataset.tokenizer_name=bert \
        dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        train.global_batch_size=128 \
        callbacks.early_stopping.monitor="val/mcc" \
        callbacks.model_checkpoint.monitor="val/mcc" \
        callbacks.model_checkpoint.filename="val/mcc" \
        callbacks.early_stopping.patience=3 
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE \
            model.d_model=768 \
            model.layer._name_=bert \
            train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=mouse \
            dataset.max_length=30 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=bert \
            dataset.tokenizer_path=weight/dnabert/dnabert3/3-new-12w-0 \
            train.global_batch_size=128 \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 

    done
done

for dataset in  prom_300_all_txt prom_300_notata_txt prom_300_tata_txt prom_core_all_txt prom_core_notata_txt prom_core_tata_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE  \
        model.d_model=768 \
        model.layer._name_=bert \
        train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        dataset.dataset_name=$dataset \
        optimizer.lr=$lr \
        dataset=prom \
        dataset.max_length=70 \
        wandb.mode=offline \
        trainer.devices=4 \
        dataset.batch_size=16 \
        dataset.tokenizer_name=bert \
        dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        train.global_batch_size=128 \
        callbacks.early_stopping.monitor="val/mcc" \
        callbacks.model_checkpoint.monitor="val/mcc" \
        callbacks.model_checkpoint.filename="val/mcc" \
        callbacks.early_stopping.patience=3 
    done
done

for dataset in  reconstructed_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE  \
            model.d_model=768 \
            model.layer._name_=bert \
            train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=splice \
            dataset.max_length=80 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=bert \
            dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
            train.global_batch_size=128 \
            trainer.gradient_clip_val=1.0 \
            wandb.id=GUE_splice_${dataset}_bert \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE  \
        model.d_model=768 \
        model.layer._name_=bert \
        train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        dataset.dataset_name=$dataset \
        optimizer.lr=$lr \
        dataset=tf \
        dataset.max_length=30 \
        wandb.mode=offline \
        trainer.devices=4 \
        dataset.batch_size=16 \
        dataset.tokenizer_name=bert \
        dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        train.global_batch_size=128 \
        callbacks.early_stopping.monitor="val/mcc" \
        callbacks.model_checkpoint.monitor="val/mcc" \
        callbacks.model_checkpoint.filename="val/mcc" \
        callbacks.early_stopping.patience=3 
    done
done

for dataset in  covid_txt; do
    for lr in 3e-5; do
        python -m train experiment=hg38/GUE  \
        model.d_model=768 \
        model.layer._name_=bert \
        train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        dataset.dataset_name=$dataset \
        optimizer.lr=$lr \
        dataset=virus \
        dataset.max_length=256 \
        wandb.mode=offline \
        trainer.devices=4 \
        dataset.batch_size=16 \
        dataset.tokenizer_name=bert \
        dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
        train.global_batch_size=128 \
        callbacks.early_stopping.monitor="val/mcc" \
        callbacks.model_checkpoint.monitor="val/mcc" \
        callbacks.model_checkpoint.filename="val/mcc" \
        callbacks.early_stopping.patience=3 
    done
done