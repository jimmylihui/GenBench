


cd ..
cd ..
for dataset in  demo_coding_vs_intergenomic_seqs,demo_human_or_worm,dummy_mouse_enhancers_ensembl,human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters,human_ocr_ensembl; do
    python -m train experiment=hg38/genomic_benchmark_bert \
            wandb.mode=offline \
            dataset.dataset_name=$dataset \
            dataset.tokenizer_path=weight/dnabert \
            trainer.devices=5 \
            dataset.batch_size=28 \
            wandb.id=GenomicBenchmark_${dataset}_bert 
done
