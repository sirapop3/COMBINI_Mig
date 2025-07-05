# #!/bin/bash

# #source /jet/home/ghong1/miniconda3/bin/activate PL-Marker
# #echo "Activated PL-Marker"

# PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
# export PYTHONPATH="$PROJECT_DIR/transformers/src:$PYTHONPATH"

# --use_ner_results: use the original entity type predicted by NER models
# DiMB-RE
# mkdir dimb-re_models
# split=dev
# split=test

# SET YOUR OWN OUTPUT DIR
# output_dir=/jet/home/ghong1/ocean_cis230030p/ghong1/PL-Marker/no-result
# output_dir=./output

# for seed in 0 1 2 3 4; do 
#     python3 run_re_trg_inserted.py \
#     --model_type bertsub \
#     --model_name_or_path /home/user/combiniiii/COMBINI_PL-Marker-main/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#     --do_lower_case \
#     --data_dir ./data \
#     --learning_rate 2e-5 --num_train_epochs 28 \
#     --per_gpu_train_batch_size 8 \
#     --per_gpu_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --max_seq_length 256 \
#     --max_pair_length 16 \
#     --save_steps 607 \
#     --eval_logsoftmax \
#     --seed $seed \
#     --output_dir $output_dir/RE/dimb-re_models_biomedbert_trg_test-${seed} \
#     --overwrite_output_dir \
#     --use_trigger \
#     --use_typemarker \
#     --lminit \
#     --eval_all_checkpoints \
#     --binary_cls \
#     --no_sym \
#     --do_train
#     # --do_eval \
#     # --evaluate_during_training \
#     # --no_test

#     # --dev_file $output_dir/NER/dimb-re_models_biomedbert_trg_dev-$seed/ent_pred_dev.json \
#     # --eval_unidirect \
# done;

# adjusted by mig

#!/bin/bash

#source /jet/home/ghong1/miniconda3/bin/activate PL-Marker
#echo "Activated PL-Marker"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
export PYTHONPATH="$PROJECT_DIR/transformers/src:$PYTHONPATH"

--use_ner_results: use the original entity type predicted by NER models
DiMB-RE
mkdir dimb-re_models
split=dev
split=test

SET YOUR OWN OUTPUT DIR
output_dir=/jet/home/ghong1/ocean_cis230030p/ghong1/PL-Marker/no-result
output_dir=./output

for seed in 0; do 
    python3 run_re_trg_marked.py \
    --model_type bertsub \
    --model_name_or_path /scratch/bdxz/sumnakkittikul/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --do_lower_case \
    --data_dir /projects/bdxz/sumnakkittikul/data \
    --learning_rate 2e-5 --num_train_epochs 1 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256 \
    --max_pair_length 16 \
    --save_steps 607 \
    --eval_logsoftmax \
    --seed $seed \
    --output_dir $output_dir/RE/dimb-re_models_biomedbert_trg_test-${seed} \
    --overwrite_output_dir \
    --use_trigger \
    --use_typemarker \
    --lminit \
    --eval_all_checkpoints \
    #--binary_cls \
    --no_sym \
    --do_train
    # --do_eval \
    # --evaluate_during_training \
    # --no_test

    # --dev_file $output_dir/NER/dimb-re_models_biomedbert_trg_dev-$seed/ent_pred_dev.json \
    # --eval_unidirect \
done;
