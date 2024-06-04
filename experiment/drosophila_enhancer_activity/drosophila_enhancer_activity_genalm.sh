





cd ..
cd ..



python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=768 \
        model.layer._name_=genalm \
        train.pretrained_model_path=weight/genalm/gena-lm-bert-base-t2t \
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
        dataset.tokenizer_path=weight/genalm/gena-lm-bert-base-t2t \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=768 \
        model.layer._name_=genalm \
        train.pretrained_model_path=weight/genalm/gena-lm-bigbird-base-t2t \
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
        dataset.tokenizer_path=weight/genalm/gena-lm-bigbird-base-t2t \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"



    


