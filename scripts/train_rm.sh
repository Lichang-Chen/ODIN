export TRANSFORMERS_CACHE=path/to/cache
export TOKENIZERS_PARALLELISM=True

export WANDB_API_KEY=xxx
export WANDB_PROJECT="ODIN"
export GPFS=/nfshomes/bobchen/ODIN/training/step2-reward-modeling
#  The hyperparameters of the loss


RM_L1_REG=0.0
RM_NUM_HEADS=2
RM_ORTHO_REG=1.0 # The coefficient of the orthogonality loss
ATTRIBUTE_CORR=0.0
LENGTH_CORR=1.0 # The coefficient of the length correlation loss
EPOCH=3
EPSILON=-1
LR=2e-5 # 1e-5 3e-5

#OPTIONS for datasets: [Dahoas/full-hh-rlhf, stanfordnlp/SHP, openai/webgpt_comparisons, stanfordnlp/SHP, Dahoas/rm-static]
DATASET="local/oasst_multi_turn"
EXP_NAME=two_head_Epoch_${EPOCH}_length_${LENGTH_CORR}_abs_ortho_${RM_ORTHO_REG}_LR_${LR}
RESULTS="${GPFS}/ODIN/${EXP_NAME}"
OUTFILE="${RESULTS}/slurm-%j.out"
ERRFILE="${RESULTS}/error-%j.out"
ZERO_STAGE=3 # 1 for 7B model and 3 for 13B model
WANDB_PROJECT="ODIN"
mkdir -p ${RESULTS}

#0,10,0 for train, val, test use
echo "*******STARTING********" 
echo "---------------" 
cd ${GPFS} 
wandb login ${WANDB} 
deepspeed --master_addr='localhost' --master_port 12345 main.py \
   --data_path local/oasst_multi_turn \
   --data_split 0,10,0 \
   --model_name_or_path lmsys/vicuna-7b-v1.5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate ${LR} \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --enable_tensorboard \
   --zero_stage ${ZERO_STAGE} \
   --deepspeed \
   --offload \
   --use_two_head_rw \
   --normalized_proj \
   --correlation_with_length ${LENGTH_CORR} \
   --rm_ortho_reg ${RM_ORTHO_REG} \
   --rm_num_heads ${RM_NUM_HEADS} \
   --wandb_exp_name ${EXP_NAME} \
   --wandb_project_name ${WANDB_PROJECT} \
   --output_dir ${RESULTS}
