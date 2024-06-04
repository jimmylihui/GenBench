




cd ..
cd ..


for max_length in 300 600 900 1200 1500 3000; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=768 \
                train.pretrained_model_path=/hyena-dna/weight/dnabert2 \
                optimizer.lr=3e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=bert2 \
                dataset.tokenizer_path=/weight/dnabert2 \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_bert2_$max_length \
                callbacks.early_stopping.patience=5 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=bert2 \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
done



    


