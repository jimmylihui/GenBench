




cd ..
cd ..


for max_length in 300 600 900 1200 1500 3000; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=768 \
                train.pretrained_model_path=/weight/genalm/gena-lm-bigbird-base-t2t \
                optimizer.lr=1e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=genalm \
                dataset.tokenizer_path=/weight/genalm/gena-lm-bigbird-base-t2t \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_genalm_$max_length \
                callbacks.early_stopping.patience=5 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=genalm \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
done



    


