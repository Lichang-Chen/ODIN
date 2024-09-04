#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import time
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import (
    SchedulerType,
    get_scheduler,
)

import wandb

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


# import socketserver
# socketserver.TCPServer.allow_reuse_address = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, AllGather, pearson_correlation
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        "--data_output_path",
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.99,
        help=
        "The beta2 of Adam. Default to 0.95, but may not be the best value.",
    )

    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default='step2_tensorboard')
    # Reward model arguments.
    parser.add_argument('--use_two_head_rw',
                        action='store_true',
                        help='Whether to use two_head_rw.')
    parser.add_argument('--use_decomposition',
                        action='store_true',
                        help='Whether to use reward decomposition.')
    parser.add_argument('--use_multihead',
                        action="store_true",
                        help='Whether to use multihead reward model to train.')
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument('--rm_num_heads', type=int, default=14,
                        help='Number of heads for the reward model.')
    parser.add_argument('--rm_l1_reg', type=float, default=0.,
                        help='L1 regularization strength for reward decomposition.')
    parser.add_argument('--rm_ortho_reg', type=float, default=0.,
                        help='Orthogonal regularization on the reward model head weights.')
    parser.add_argument('--normalized_proj', default=False, action='store_true',
                        help='Whether to normalize the projection head.')
    parser.add_argument('--greedy_epsilon', type=float, default=0.,
                        help='The epsilon for epsilon-greedy.')
    parser.add_argument("--debug", 
                        action='store_true', 
                        help="when it is in debug mode, we remove the first evaluation process to save time.")
    parser.add_argument('--attribute_corr', type=float, default=0.3,
                        help='The attribute correlation.')
    parser.add_argument('--correlation_with_length', type=float, default=0.1,
                        help='The attribute correlation.')
    parser.add_argument('--wandb_exp_name', type=str, help='the wandb experiment name')
    parser.add_argument('--wandb_project_name', type=str, default='rw-dcp-3', help='the wandb experiment name')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def save_model(rm_model, tokenizer, args):
    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

def main():
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl') # comment this when doing the evaluation
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step2_model")
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    dcomp_config = {
        'rm_num_heads': args.rm_num_heads,
        'rm_l1_reg': args.rm_l1_reg,
        'rm_ortho_reg': args.rm_ortho_reg,
        'normalized_proj': args.normalized_proj,
        'epsilon': args.greedy_epsilon,
        'attribute_corr': args.attribute_corr,
        'correlation_with_length': args.correlation_with_length,
    }
    # using wandb to login
    if args.global_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project_name,
            name=args.wandb_exp_name,
            config=dcomp_config,
        )

    # craete the dataset first for debugging
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len
        )
    # breakpoint()
    
    assert (args.use_decomposition == True and args.use_multihead == True) is False
    rm_model = create_critic_model(
        args.model_name_or_path,
        tokenizer,
        ds_config,
        args.num_padding_at_beginning,
        disable_dropout=args.disable_dropout,
        use_two_head_rw=args.use_two_head_rw,
        use_decomposition=args.use_decomposition,
        use_multihead=args.use_multihead,
        dcomp_config=dcomp_config
        )
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)
    # DataLoaders creation for the attributes
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            chosen = outputs["chosen_quality_reward"]
            rejected = outputs["rejected_quality_reward"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_quality_reward"].mean().float()

            # if step == 99:  # For faster evaluation and debugging
                # break
        acc = correct_predictions / total_predictions # use the quality reward to calculat the accuracy
        scores = scores / (step + 1)
        # try:
        acc = get_all_reduce_mean(acc).item()
        scores = get_all_reduce_mean(scores).item()
        # except:
        #     pass
        return scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, args.adam_beta2))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    
    print_rank_0(f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****", args.global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}", args.global_rank)

    baseline_acc = acc
    # baseline_acc = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            # the training will only backward the loss item
            # calculate the loss of the length here.
            local_length_list = outputs['seqlens'].contiguous()
            local_length_reward_list = outputs['length_reward'].contiguous()
            local_rewards_list = outputs['rewards'].contiguous()

            all_length_list = [torch.zeros_like(local_length_list) for _ in range(torch.distributed.get_world_size())]
            all_length_reward_list = [torch.zeros_like(local_length_reward_list) for _ in range(torch.distributed.get_world_size())]
            all_rewards_list = [torch.zeros_like(local_rewards_list) for _ in range(torch.distributed.get_world_size())]
            #TODO: debug the AllGather function
            # all_length_list = AllGather.apply(all_length_list, local_length_list)
            # all_length_reward_list = AllGather.apply(all_length_reward_list, local_length_reward_list)

            # collect the data from all GPUs
            dist.all_gather(all_length_list, local_length_list)
            dist.all_gather(all_length_reward_list, local_length_reward_list)
            dist.all_gather(all_rewards_list, local_rewards_list)

            all_length_reward_list[args.global_rank] = local_length_reward_list # backward the loss on the local device(local has the grad_fn)
            all_rewards_list[args.global_rank] = local_rewards_list
            torch.distributed.barrier() # synchronize the processes

            all_length_tensor = torch.cat([tsr.to(outputs['loss']) for tsr in all_length_list], dim=0)
            all_length_reward_tensor = torch.cat([tsr.to(outputs['loss']) for tsr in all_length_reward_list], dim=0)
            all_rewards_tensor = torch.cat([tsr.to(outputs['loss']) for tsr in all_rewards_list], dim=0)
            # all_length_tensor[torch.isinf(all_length_tensor)] = args.max_seq_len
            all_length_tensor = torch.cat(all_length_list, dim=0).to(outputs['loss'])
            all_length_reward_tensor = torch.cat(all_length_reward_list, dim=0).to(outputs['loss'])
            # calculate the correlation between all_length_tensor and all_rewards_tensor
            correlations = []
            if args.use_decomposition or args.use_multihead: # the code for using multihead > 2
                for i in range(all_rewards_tensor.size(1)):
                    rewards_matrix = torch.stack((all_length_tensor, all_rewards_tensor[:, i]))
                    rewards_corr_matrix = torch.corrcoef(rewards_matrix.to(dtype=torch.float32))
                    correlations.append(rewards_corr_matrix[0,1])
                correlations = torch.tensor(correlations).to(outputs['loss'])
                correlations[-2] = 0 # make sure the correlation with length_dim is zero
                other_dim_length_corr_loss = torch.sum(torch.abs(correlations))

                data_matrix = torch.stack((all_length_tensor, all_length_reward_tensor))
                corr_matrix = torch.corrcoef(data_matrix.to(dtype=torch.float32))
                length_loss = 1 - corr_matrix[0,1]
            elif args.use_two_head_rw:
                # two ways: only calculate the reward tensor
                rw_matrix = torch.stack((all_length_tensor, all_rewards_tensor[:, 0]))
                rw_corr_matrix = torch.corrcoef(rw_matrix.to(dtype=torch.float32))
                other_dim_length_corr_loss = torch.abs(rw_corr_matrix[0,1])

                data_matrix = torch.stack((all_length_tensor, all_length_reward_tensor))
                corr_matrix = torch.corrcoef(data_matrix.to(dtype=torch.float32))
                length_loss = 1 - corr_matrix[0,1]

                ## add ranking corr
                ranking_matrix = torch.stack((all_length_tensor, all_rewards_tensor[:, 0] + all_rewards_tensor[:, 1]))
                rewards_corr_matrix = torch.corrcoef(ranking_matrix.to(dtype=torch.float32))
                ranking_corr_loss = torch.abs(rewards_corr_matrix[0,1])

            if args.global_rank == 0:
                # device = outputs['loss'].device
                # all_length_tensor = torch.cat([tsr.to(outputs['loss']) for tsr in all_length_list], dim=0)
                # all_length_reward_tensor = torch.cat([tsr.to(outputs['loss']) for tsr in all_length_reward_list], dim=0)
                # all_length_tensor[torch.isinf(all_length_tensor)] = args.max_seq_len
                # all_length_tensor = torch.cat(all_length_list, dim=0).to(outputs['loss'])
                # all_length_reward_tensor = torch.cat(all_length_reward_list, dim=0).to(outputs['loss'])

                # data_matrix = torch.stack((all_length_tensor, all_length_reward_tensor))
                # corr_matrix = torch.corrcoef(data_matrix.to(dtype=torch.float32))
                # length_loss = 1 - corr_matrix[0,1]

                # print(f'All_length_tensor: {all_length_tensor}')
                # print(f'All_length_reward_tensor: {all_length_reward_tensor}')
                # print(f'Length Loss: {length_loss}')
                wandb.log(
                {
                    "ranking_corr_loss": ranking_corr_loss,
                    "other_dim_length_corr_loss": other_dim_length_corr_loss,
                    "length_loss": length_loss,
                    "loss": outputs['loss'],
                    "ortho_loss": outputs["ortho_loss"],
                    "attribute_loss": outputs["regression_loss"]
                }
                )
            loss = outputs["loss"]
            loss += args.correlation_with_length * length_loss
            loss += args.correlation_with_length * other_dim_length_corr_loss
            loss += args.correlation_with_length * ranking_corr_loss
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            # Evaluate reward_loss on the validation set.
            if (step + 1) % int(len(train_dataloader) // 5) == 0:
                print_rank_0(
                    f"Epoch {epoch + 1}/{args.num_train_epochs} Step {step + 1}/{len(train_dataloader)} with loss: {mean_loss / (step + 1)}, time used: {(time.time() - t0) / 60:.0f} minutes",
                    args.global_rank,
                )
                # Evaluate reward_loss on the validation set.
                print_rank_0(
                    f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(train_dataloader)} *****",
                    args.global_rank,
                )
                reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
                print_rank_0(
                    f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
                    args.global_rank)
                if acc > baseline_acc:
                    save_model(rm_model, tokenizer, args)
                    baseline_acc = acc
                    baseline_reward = reward_score
                    if args.global_rank == 0:
                        wandb.log({"Acc": acc, "Baseline_Acc": baseline_acc}, step=step)
        rm_model.tput_timer.update_epoch_count()
        print_rank_0(f"The selected reward model has chosen_last_score:{baseline_reward} and acc: {baseline_acc}", args.global_rank)


if __name__ == "__main__":
    main()
