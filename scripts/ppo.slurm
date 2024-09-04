#!/bin/bash
#SBATCH -A xxx
#SBATCH -p xxx
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J "jobname"
#SBATCH --exclusive

GPFS=xxx
HOME_DIR=/home/lichang
DSP_DIR=path/to/docker_file # for finding the dsp-rlhf.sqsh (docker container)
CONTAINER="${DSP_DIR}/chzhu+dsp-rlhf.sqsh"
WANDB=$WANDB_API_KEY
export TRANSFORMERS_CACHE=xxx
# export TOKENIZERS_PARALLELISM=True

BASE_PATH=path/to/base_path
RM_PATH="vicuna-7b-LR1e-5_Dec3"
REWARD_MODEL_PATH="${BASE_PATH}/RM/${RM_PATH}"
EPOCH=1
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
SEED=1234
ACTOR_LR=5e-7
KL_CTL=0.10
ONLINE_SFT_WARMUP_STEP=100

# some NCCL settings
export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATASET="local/oasst_multi_turn"
EXP_NAME=ppo_7B_KL${KL_CTL}
RESULTS="${GPFS}/ppo/${EXP_NAME}" # the finetued model (could be better if we also finetune this model on the oasst dataset)
WANDB_PROJECT="ppo-dec2"
mkdir -p ${RESULTS}

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo ${VAR} \
&& cd ${GPFS} \
&& wandb login ${WANDB} \
&& deepspeed --master_addr='localhost' --master_port 12348 main_ppo.py \
   --algo "remax" \
   --data_path ${DATASET} \
   --data_split 0,0,10 \
   --actor_model_name_or_path ${RESULTS}/actor \
   --reward_model_name_or_path ${REWARD_MODEL_PATH} \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
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
   --reward_zero_stage ${REWARD_ZERO_STAGE} \
   --actor_zero_stage ${ACTOR_ZERO_STAGE} \
   --reference_zero_stage ${REFERENCE_ZERO_STAGE} \
   --enable_hybrid_engine \
   --output_dir ${RESULTS} \
   --wandb_exp_name ${EXP_NAME} \
   --wandb_project_name ${WANDB_PROJECT} \
   --save_answers \
   --global_step ${VAR} \
   --online_sft_warmup ${ONLINE_SFT_WARMUP_STEP} \
   --save_at_final
EOF

MOUNTS="--container-mounts=${HOME_DIR}:${HOME_DIR},${DSP_DIR}:${DSP_DIR}"

srun --container-image=${CONTAINER} ${MOUNTS} bash -c "${cmd}"
set +x