




cd ..
cd ..


for max_length in 3000; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=768 \
                train.pretrained_model_path=none \
                optimizer.lr=1e-4 \
                wandb.mode=offline \
                dataset.tokenizer_name=spliceai \
                dataset.tokenizer_path=/weight/hyenadna/hyenadna-large-1m-seqlen \
                dataset.return_mask=false \
                model.layer._name_=spliceai \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=4 \
                dataset.batch_size=25 \
                decoder=id \
                wandb.id=splicing_prediction_spliceai__$max_length \
                callbacks.early_stopping.patience=10 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
done



    


