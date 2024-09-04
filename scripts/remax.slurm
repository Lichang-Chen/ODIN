#!/bin/bash
#SBATCH -A xxx
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:8

GPFS=xxx
HOME_DIR=/home/lichang
DSP_DIR=xxx # for finding the dsp-rlhf.sqsh (docker container)
CONTAINER="${DSP_DIR}/chzhu+dsp-rlhf.sqsh"
WANDB=$WANDB_API_KEY
export TRANSFORMERS_CACHE=path/to/cache
export WANDB_START_METHOD="thread"
export CUDA_LAUNCH_BLOCKING=1
# export TOKENIZERS_PARALLELISM=True

BASE_PATH=xxx
RM_PATH=${1} #"Quality_Acc_Epoch_3_length1.0_abs_ortho0.0_LR1e-5"
ACTOR_LR=${2} #1e-7
KL_CTL=${3} #0.01
num_epochs=${4}
FLAGS=${5}

REWARD_MODEL_PATH="${BASE_PATH}/${RM_PATH}"
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
SEED=1234

DATASET="local/oasst_prompt"
rm_name="${RM_PATH##*/}" # Take the last "word" after "/"
EXP_NAME=baseline_7B_KL${KL_CTL}_LR${ACTOR_LR}_RM_${rm_name}
RESULTS="${GPFS}/remax-Jan17/${EXP_NAME}" # the finetued model (could be better if we also finetune this model on the oasst dataset)
WANDB_PROJECT="bs32-jan17-length1024"
mkdir -p ${RESULTS}

if [[ -f "${RESULTS}/actor" ]]; then
    ACTOR_NAME=${RESULTS}/actor
else
    ACTOR_NAME=lmsys/vicuna-7b-v1.5
fi

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo ${VAR} \
&& cd ${GPFS} \
&& wandb login ${WANDB} \
&& deepspeed --master_addr='localhost' --master_port 12348 main_remax_nccl_multi_epoch_shuffle.py \
   --algo "remax" \
   --data_path ${DATASET} \
   --data_split 0,0,10 \
   --actor_model_name_or_path ${ACTOR_NAME} \
   --reward_model_name_or_path ${REWARD_MODEL_PATH} \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs ${num_epochs} \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --disable_reward_dropout \
   --num_warmup_steps 0 \
   --kl_ctl ${KL_CTL} \
   --gamma 0.99 \
   --deepspeed \
   --offload \
   --offload_reward_model \
   --offload_reference_model \
   --actor_bf16 \
   --reward_bf16 \
   --seed ${SEED} \
   --actor_zero_stage ${ACTOR_ZERO_STAGE} \
   --reward_zero_stage ${REWARD_ZERO_STAGE} \
   --reference_zero_stage ${REFERENCE_ZERO_STAGE} \
   --enable_hybrid_engine \
   --output_dir ${RESULTS} \
   --wandb_exp_name ${EXP_NAME} \
   --wandb_project_name ${WANDB_PROJECT} \
   --save_answers \
   --use_multihead_rm --exit_duration_in_mins 220 ${FLAGS}
EOF

MOUNTS="--container-mounts=${HOME_DIR}:${HOME_DIR},${DSP_DIR}:${DSP_DIR}"

# sh -c "${cmd}"

mkdir -p ${RESULTS}/logs
OUTFILE="${RESULTS}/logs/slurm-%j-%n.out"
ERRFILE="${RESULTS}/logs/error-%j-%n.out"
srun -o $OUTFILE -e $ERRFILE -l --output=$RESULTS/logs/%x_%j_$DATETIME.log  \
    --container-image=${CONTAINER} ${MOUNTS} bash -c "${cmd}"

set +x