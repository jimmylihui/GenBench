

cd ..
cd ..
for length in 512 ; do
    python -m train experiment=hg38/species  \
        model.d_model=768 \
        model.layer._name_=bert \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/dnabert/dnabert3/3-new-12w-0 \
        optimizer.lr=8e-5 \
        wandb.mode=offline \
        dataset.max_length=$length \
        dataset.total_size=1000 \
        trainer.devices=5 \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/dnabert/dnabert3/tokenizer \
        dataset.batch_size=14 \
        train.global_batch_size=128 \
        wandb.id=species_bert_seqlen_$length \
        trainer.max_epochs=200 \
        dataset.tokenizer_name=bert

done

