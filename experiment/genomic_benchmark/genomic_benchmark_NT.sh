


cd ..
cd ..
for dataset in human_ensembl_regulatory human_nontata_promoters,human_ocr_ensembl; do
    

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=768 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=weight/nt/nucleotide-transformer-v2-250m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-250m-multi-species \
    dataset.tokenizer_path=weight/nt/nucleotide-transformer-v2-250m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=512 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=weight/nt/nucleotide-transformer-v2-100m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    dataset.tokenizer_path=weight/nt/nucleotide-transformer-v2-100m-multi-species \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-100m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3

    python -m train experiment=hg38/genomic_benchmark_NT  \
    wandb.mode=offline \
    model.d_model=512 \
    dataset.dataset_name=$dataset \
    train.pretrained_model_path=weight/nt/nucleotide-transformer-v2-50m-multi-species \
    trainer.devices=5 \
    dataset.batch_size=14 \
    trainer.max_epochs=20 \
    dataset.tokenizer_path=weight/nt/nucleotide-transformer-v2-50m-multi-species \
    wandb.id=GenomicBenchmark_${dataset}_nucleotide-transformer-v2-50m-multi-species \
    train.global_batch_size=128 \
    callbacks.early_stopping.patience=3
done

