set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/home/ibdo/Desktop/NLP/
export TASK_DATA_PATH=/home/ibdo/Desktop/NLP/task_data

python3 -u run_classifier.py \
	           --init_checkpoint /home/ibdo/Desktop/NLP/0617-0621/ERNIE/checkpoints/step_1000 \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 2 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/chnsenticorp/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/chnsenticorp/dev.tsv \
                   --test_set /home/ibdo/Desktop/NLP/0617-0621/ERNIE/snownlp\ data/snow_ERNIE.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 10 \
                   --max_seq_len 256 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
