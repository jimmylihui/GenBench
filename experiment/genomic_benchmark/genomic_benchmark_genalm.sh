

#hyena-dna large on genomic benchmark datasets 
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=demo_coding_vs_intergenomic_seqs train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=demo_human_or_worm train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_enhancers_cohn train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_enhancers_ensembl train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_ensembl_regulatory train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_nontata_promoters train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_ocr_ensembl train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t model.d_model: 1024

# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=demo_coding_vs_intergenomic_seqs train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=demo_human_or_worm train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_enhancers_cohn train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_enhancers_ensembl train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_ensembl_regulatory train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_nontata_promoters train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
# python -m train experiment=hg38/genomic_benchmark_Genalm  dataset.dataset_name=human_ocr_ensembl train.pretrained_model_path: /liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t model.d_model: 768
cd ..
cd ..
#convert pervious code into for cycle to run all datasets 
for dataset in demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
#     python -m train experiment=hg38/genomic_benchmark_Genalm \
#             dataset.dataset_name=$dataset \
#             train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t \
#             model.d_model=1024 \
#             trainer.devices=5 \
#             wandb.mode=offline \
#             dataset.batch_size=14 \
#             train.global_batch_size=128 \
#             wandb.id=GenomicBenchmark_${dataset}_gena-lm-bert-large-t2t \
#             optimizer.lr=1e-5 


#     python -m train experiment=hg38/genomic_benchmark_Genalm  \
#             dataset.dataset_name=$dataset \
#             train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t  \
#             model.d_model=768 \
#             trainer.devices=4 \
#             wandb.mode=offline \
#             dataset.batch_size=14 \
#             train.global_batch_size=128 \
#             wandb.id=GenomicBenchmark_${dataset}_gena-lm-bert-base-t2t \
#             optimizer.lr=1e-5 \
#             callbacks.early_stopping.patience=3
    
    python -m train experiment=hg38/genomic_benchmark_Genalm  \
            dataset.dataset_name=$dataset \
            train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t  \
            model.d_model=768 \
            trainer.devices=4 \
            wandb.mode=offline \
            dataset.batch_size=14 \
            train.global_batch_size=128 \
            dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
            wandb.id=GenomicBenchmark_${dataset}_gena-lm-bigbird-base-t2t \
            optimizer.lr=5e-5 \
            callbacks.early_stopping.patience=3 
done
