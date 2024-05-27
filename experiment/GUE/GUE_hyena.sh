

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

batch_size=32
for dataset in  H3_txt H3K4me1_txt H3K4me2_txt H3K4me3_txt H3K9ac_txt H3K14ac_txt H3K36me3_txt H3K79me3_txt H4_txt H4ac_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.tokenizer_name=hyena \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GUE_EMP_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.tokenizer_name=hyena \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 \
                
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=EMP \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=hyena \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.id=GUE_EMP_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=EMP \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GUE_EMP_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done


for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=mouse \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GUE_mouse_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=mouse \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                wandb.id=GUE_mouse_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=mouse \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.id=GUE_mouse_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=mouse \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.id=GUE_mouse_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=mouse \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GUE_mouse_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  prom_300_all_txt prom_300_notata_txt prom_300_tata_txt prom_core_all_txt prom_core_notata_txt prom_core_tata_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=prom \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GUE_prom_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=prom \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                wandb.id=GUE_prom_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=prom \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.id=GUE_prom_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=prom \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_hyenadna-small-32k-seqlen \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=prom \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  reconstructed_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=splice \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                datset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=splice \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=splice \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.id=GUE_splice_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=splice \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=splice \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GUE_splice_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done 
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=tf \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_hyenadna-large-1m-seqlen \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=tf \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_hyenadna-medium-450k-seqlen \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=tf \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_hyenadna-medium-160k-seqlen \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=tf \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.id=GUE_tf_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=tf \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GUE_tf_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  covid_txt; do
    for lr in 1e-4; do
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=virus \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GUE_virsu_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=virus \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-450k-seqlen \
                wandb.id=GUE_virus_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=virus \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.id=GUE_virus_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                allbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=256 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=virus \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.id=GUE_virus_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE  \
                model.d_model=128 \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset.tokenizer_name=hyena \
                dataset=virus \
                dataset.max_length=128 \
                trainer.devices=4 \
                dataset.batch_size=$batch_size \
                train.global_batch_size=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GUE_virus_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10 \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done