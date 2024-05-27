


# python -m train experiment=hg38/species  model.d_model=256 train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/hyenadna/hyenadna-large-1m-seqlen dataset.dataset_name=$dataset optimizer.lr=6e-5 wandb.mode=offline dataset.max_length=1024 dataset.total_size=1000 trainer.devices=5 dataset.batch_size=8 wandb.id=species_hyenadna-large-1m-seqlen_1k callbacks.early_stopping.patience=100 trainer.max_epochs=200


cd ..
cd ..

# python -m train experiment=hg38/drosophila_enhancer_activity \
#         model.d_model=1024 \
#         model.layer._name_=genalm \
#         train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-large-t2t \
#         optimizer.lr=5e-5 \
#         wandb.mode=offline \
#         dataset.max_length=128 \
#         trainer.devices=5 \
#         dataset.batch_size=25 \
#         wandb.id=drosophila_enhancer_activity_genalm \
#         callbacks.early_stopping.patience=10 \
#         trainer.max_epochs=100 \
#         train.global_batch_size=125 \
#         dataset.tokenizer_name=genalm \
#         callbacks.early_stopping.monitor="val/pearsonr" \
#         callbacks.model_checkpoint.monitor="val/pearsonr"\
#         callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=768 \
        model.layer._name_=genalm \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t \
        optimizer.lr=5e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_gena-lm-bert-base-t2t \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=genalm \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bert-base-t2t \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=768 \
        model.layer._name_=genalm \
        train.pretrained_model_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
        optimizer.lr=5e-5 \
        wandb.mode=offline \
        dataset.max_length=128 \
        trainer.devices=5 \
        dataset.batch_size=25 \
        wandb.id=drosophila_enhancer_activity_gena-lm-bigbird-base-t2t \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=125 \
        dataset.tokenizer_name=genalm \
        dataset.tokenizer_path=/liuzicheng/ljh/hyena-dna/weight/genalm/gena-lm-bigbird-base-t2t \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"



    


