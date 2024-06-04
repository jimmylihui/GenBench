


cd ..
cd ..
for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_hyena_small \
                model.d_model=256 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.dataset_name=$dataset \
                optimizer.lr=$lr \
                trainer.devices=5 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-large-1m-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_hyenadna-large-1m-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done

for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_hyena_small \
                model.d_model=256 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-medium-450k-seqlen \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=hyena \
                optimizer.lr=$lr \
                trainer.devices=5 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-medium-450k-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_hyenadna-medium-450k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done

for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_hyena_small \
                model.d_model=256 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-medium-160k-seqlen \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=hyena \
                optimizer.lr=$lr \
                trainer.devices=5 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-medium-160k-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_hyenadna-medium-160k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done

for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_hyena_small \
                model.d_model=256 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-small-32k-seqlen \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=hyena \
                optimizer.lr=$lr \
                trainer.devices=5 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-small-32k-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_hyenadna-small-32k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done

for dataset in dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl; do
    for lr in 1e-4; do
        python -m train experiment=hg38/genomic_benchmark_hyena_small \
                model.d_model=128 \
                train.pretrained_model_path=weight/hyenadna/hyenadna-tiny-1k-seqlen \
                dataset.dataset_name=$dataset \
                dataset.tokenizer_name=hyena \
                optimizer.lr=$lr \
                trainer.devices=5 \
                dataset.batch_size=28 \
                train.global_batch_size=128 \
                dataset.tokenizer_path=weight/hyenadna/hyenadna-tiny-1k-seqlen \
                wandb.id=GenomicBenchmark_${dataset}_hyenadna-tiny-1k-seqlen \
                wandb.mode=offline \
                callbacks.early_stopping.patience=10  
    done
done


