

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
for dataset in demo_coding_vs_intergenomic_seqs,demo_human_or_worm,dummy_mouse_enhancers_ensembl,human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters,human_ocr_ensembl; do
    python -m train experiment=hg38/genomic_benchmark_BERT2 \
        wandb.mode=offline \
        dataset.dataset_name=$dataset \
        trainer.devices=5 \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/dnabert2 \
        dataset.batch_size=28 \
        trainer.max_epochs=20 
done
