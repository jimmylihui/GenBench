

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
#dataset =EMP
for dataset in  H3_txt H3K4me1_txt H3K4me2_txt H3K4me3_txt H3K9ac_txt H3K14ac_txt H3K36me3_txt H3K79me3_txt H4_txt H4ac_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=EMP \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_EMP_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
       
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=mouse \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_mouse_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=mouse \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_mouse_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=mouse \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_mouse_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=mouse \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_mouse_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
                
    done
done

for dataset in  prom_300_all_txt prom_300_notata_txt prom_300_tata_txt prom_core_all_txt prom_core_notata_txt prom_core_tata_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=prom \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=prom \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=prom \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=prom \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_prom_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  reconstructed_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=splice \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=splice \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=splice \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=splice \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_splice_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  0_txt 1_txt 2_txt 3_txt 4_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=tf \
                dataset.max_length=128 \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=tf \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=tf \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=tf \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_tf_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done

for dataset in  covid_txt; do
    for lr in 1e-5; do
        python -m train experiment=hg38/GUE \
                model.d_model=1024 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=virus \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_virus_${dataset}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=768 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=virus \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_virus_${dataset}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 

        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=virus \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_virus_${dataset}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
        
        python -m train experiment=hg38/GUE \
                model.d_model=512 \
                model.layer._name_=NT \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                dataset=virus \
                dataset.max_length=128 \
                wandb.mode=offline \
                trainer.devices=4 \
                dataset.batch_size=14 \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                train.global_batch_size=128 \
                wandb.id=GUE_virus_${dataset}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.monitor="val/mcc" \
                callbacks.model_checkpoint.monitor="val/mcc" \
                callbacks.model_checkpoint.filename="val/mcc" \
                callbacks.early_stopping.patience=3 
    done
done