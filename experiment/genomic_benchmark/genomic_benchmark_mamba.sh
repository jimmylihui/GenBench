



cd ..
cd ..




for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-5; do
        python -m train experiment=hg38/genomic_benchmark_mamba \
                model.d_model=512 \
                dataset.tokenizer_name=mamba \
                train.pretrained_model_path=weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                trainer.devices=4 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                wandb.id=GenomicBenchmark_${dataset}_caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                wandb.mode=offline \
                dataset.tokenizer_path=weight/mamba/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
                callbacks.early_stopping.patience=3 \
                callbacks.early_stopping.mode=max \
                callbacks.early_stopping.patience=3
    done
done




