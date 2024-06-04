


cd ..
cd ..
for dataset in demo_coding_vs_intergenomic_seqs,demo_human_or_worm,dummy_mouse_enhancers_ensembl,human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters,human_ocr_ensembl; do
    python -m train experiment=hg38/genomic_benchmark_BERT2 \
        wandb.mode=offline \
        dataset.dataset_name=$dataset \
        trainer.devices=5 \
        dataset.tokenizer_path=weight/dnabert2 \
        dataset.batch_size=28 \
        trainer.max_epochs=20 
done
