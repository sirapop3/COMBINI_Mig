#!/bin/bash
#SBATCH --job-name=train-combini
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdxz-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Load and activate conda env
module reset
module load anaconda3_gpu
eval "$(conda shell.bash hook)"
conda activate combini

# Add local transformers to Python path
export PYTHONPATH="$(pwd)/transformers/src:$PYTHONPATH"

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Node: $(hostname)"

# Run training
python3 run_re_trg_marked.py \
  --model_type bertsub \
  --model_name_or_path /scratch/bdxz/sumnakkittikul/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --do_lower_case \
  --data_dir /projects/bdxz/sumnakkittikul/data \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --max_pair_length 16 \
  --save_steps 607 \
  --eval_logsoftmax \
  --seed 0 \
  --output_dir /projects/bdxz/sumnakkittikul/output/RE/dimb-re_models_biomedbert_trg_test-0 \
  --overwrite_output_dir \
  --use_trigger \
  --use_typemarker \
  --lminit \
  --eval_all_checkpoints \
  --no_sym \
  --do_train
