


cd ..
cd ..

for dataset in demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do

    
    python -m train experiment=hg38/genomic_benchmark_Genalm  \
            dataset.dataset_name=$dataset \
            train.pretrained_model_path=weight/genalm/gena-lm-bigbird-base-t2t  \
            model.d_model=768 \
            trainer.devices=4 \
            wandb.mode=offline \
            dataset.batch_size=14 \
            train.global_batch_size=128 \
            dataset.tokenizer_path=weight/genalm/gena-lm-bigbird-base-t2t \
            wandb.id=GenomicBenchmark_${dataset}_gena-lm-bigbird-base-t2t \
            optimizer.lr=5e-5 \
            callbacks.early_stopping.patience=3 
done
