

#hyena-dna large on genomic benchmark datasets 
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=demo_coding_vs_intergenomic_seqs
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=demo_human_or_worm
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=human_enhancers_cohn
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=human_enhancers_ensembl
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=human_ensembl_regulatory
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=human_nontata_promoters
# python -m train experiment=hg38/genomic_benchmark_hyena_BERT  dataset.dataset_name=human_ocr_ensembl

#convert pervious code into for cycle to run all datasets 
cd ..
cd ..
for dataset in human_ensembl_regulatory human_nontata_promoters,human_ocr_ensembl; do
    # python -m train experiment=hg38/genomic_benchmark_NT  \
    # wandb.mode=offline \
    # model.d_model=1024 \
    # dataset.dataset_name=$dataset \
    # train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-500m-multi-species \
    # trainer.devices=5 \
    # dataset.batch_size=2 \
    # trainer.max_epochs=20 \
    # wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-500m-multi-species \
    # train.global_batch_size=128 

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=768 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-250m-multi-species \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-250m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=512 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-100m-multi-species \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-100m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=512 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/nt/nucleotide-transformer-v2-50m-multi-species \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-50m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3
done

