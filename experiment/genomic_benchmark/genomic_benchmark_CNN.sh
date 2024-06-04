

cd ..
cd ..
for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_CNN \
                model.d_model=256 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                trainer.devices=4 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_CNN \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done


