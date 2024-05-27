

#hyena-dna large on genomic benchmark datasets 
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=demo_coding_vs_intergenomic_seqs
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=demo_human_or_worm
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_enhancers_cohn
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_enhancers_ensembl
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_ensembl_regulatory
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_nontata_promoters
# python -m train experiment=hg38/genomic_benchmark_hyena_small  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=human_ocr_ensembl

#convert pervious code into for cycle to run all datasets and set learning rate from 1e-4 to 6e-4

cd ..
cd ..
# for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
#     for lr in 1e-4; do
#         python -m train experiment=hg38/genomic_benchmark_mamba \
#                 model.d_model=256 \
#                 dataset.tokenizer_name=mamba \
#                 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
#                 dataset.dataset_name=$dataset \
#                 optimizer.lr=$lr \
#                 trainer.devices=4 \
#                 dataset.batch_size=28 \
#                 train.global_batch_size=128 \
#                 wandb.id=GenomicBenchmark_${dataset}_caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
#                 wandb.mode=offline \
#                 callbacks.early_stopping.patience=3 \
#                 callbacks.early_stopping.mode=max \
#                 callbacks.early_stopping.patience=3
#     done
# done


# for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
#     for lr in 1e-4; do
#         python -m train experiment=hg38/genomic_benchmark_mamba \
#                 model.d_model=256 \
#                 dataset.tokenizer_name=mamba \
#                 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
#                 dataset.dataset_name=$dataset \
#                 optimizer.lr=$lr \
#                 trainer.devices=4 \
#                 dataset.batch_size=28 \
#                 train.global_batch_size=128 \
#                 wandb.id=GenomicBenchmark_${dataset}_caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
#                 wandb.mode=offline \
#                 callbacks.early_stopping.patience=3 \
#                 callbacks.early_stopping.mode=max \
#                 callbacks.early_stopping.patience=3 
#     done
# done



for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-5; do
        python -m train experiment=hg38/genomic_benchmark_mamba \
                model.d_model=512 \
                dataset.tokenizer_name=mamba \
                train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                trainer.devices=4 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                wandb.id=GenomicBenchmark_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                wandb.mode=offline \
                dataset.tokenizer_path=liuzicheng/ljh/hyena-dna/weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                callbacks.early_stopping.patience=3 \
                callbacks.early_stopping.mode=max \
                callbacks.early_stopping.patience=3
    done
done




