

#hyena-dna large on genomic benchmark datasets 
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=demo_coding_vs_intergenomic_seqs
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=demo_human_or_worm
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_enhancers_cohn
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_enhancers_ensembl
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_ensembl_regulatory
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_nontata_promoters
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_ocr_ensembl
cd ..
cd ..
#convert pervious code into for cycle to run all datasets and set learning rate from 1e-4 to 6e-4

for dataset in  H3_txt H3K4me1_txt H3K4me2_txt H3K4me3_txt H3K9ac_txt H3K14ac_txt H3K36me3_txt H3K79me3_txt H4_txt H4ac_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=EMP \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=EMP \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=EMP \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=EMP \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-4; do
       python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=mouse \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_mouse_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc" \
            callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=mouse \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_mouse_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc"\
            callbacks.model_checkpoint.filename="val/mcc"

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=mouse \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_mouse_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=mouse \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_mouse_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
    done
done

for dataset in  prom_300_all_txt prom_300_notata_txt prom_300_tata_txt prom_core_all_txt prom_core_notata_txt prom_core_tata_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=prom \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_prom_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc"\
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=prom \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_prom_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=prom \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_prom_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=prom \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_prom_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
    done
done

for dataset in  reconstructed_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=splice \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_splice_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=splice \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_splice_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=splice \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_splice_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=splice \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_splice_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=tf \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=tf \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=tf \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=tf \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_EMP_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
    done
done

for dataset in  covid_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=virus \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            train.global_batch_size=128 \
            wandb.id=GUE_virus_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=virus \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            dataset.tokenizer_name=mamba \
            train.global_batch_size=128 \
            wandb.id=GUE_virus_${dataset}_caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"

        python -m train experiment=hg38/GUE \
            model.d_model=256 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=virus \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_virus_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
        
        python -m train experiment=hg38/GUE \
            model.d_model=512 \
            model.layer._name_=mamba \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            dataset.dataset_name=$dataset \
            optimizer.lr=$lr \
            dataset=virus \
            dataset.max_length=128 \
            wandb.mode=offline \
            trainer.devices=4 \
            dataset.batch_size=28 \
            dataset.tokenizer_name=mamba \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            train.global_batch_size=128 \
            wandb.id=GUE_virus_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            callbacks.early_stopping.patience=3 \
            callbacks.early_stopping.mode=max \
            callbacks.early_stopping.monitor="val/mcc" \
            callbacks.model_checkpoint.monitor="val/mcc" \
            callbacks.model_checkpoint.filename="val/mcc"
    done
done