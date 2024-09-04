import argparse
import json
import os
import sys
import time

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    SchedulerType,
    default_data_collator,
)

from perf import print_throughput_step3
from remax_trainer import (
    DeepSpeedReMaxTrainer,
    DeepSpeedReMaxTrainerUnsupervised,
)
from rlhf_engine import DeepSpeedRLHFEngine

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import (
    create_prompt_dataset,
    MiniDataset,
    DataCollatorRLHF,
    get_unsupervised_data,
)
from utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    save_zero_three_model,
    load_hf_tokenizer,
)
from utils.module.lora import convert_lora_to_linear_layer

# from utils.perf import print_throughput_step3
import wandb
import torch.distributed as dist
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description="(Step 3) RLHF training arguments")

    parser.add_argument(
        "--algo",
        type=str,
        default="remax",
        choices=["remax"],
        help="Algorithm to use.",
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` "
        "will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsup_coef",
        type=float,
        default=27.8,
        help="""gamma in Equation 2 from InstructGPT paper""",
    )
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader and generation purpose.",
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the training dataloader and training purpose.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--generation_batches",
        type=int,
        default=1,
        help="Generate x batches to go to training mode.",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.",
    )
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_answer_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--actor_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action="store_true",
        help="Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed.",
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action="store_true",
        help="Unpin actor's parameters during generation. This makes generation slower but requires less memory.",
    )
    parser.add_argument(
        "--release_inference_cache",
        action="store_true",
        help="Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size.",
    )
    parser.add_argument(
        "--memory_efficient_linear",
        action="store_true",
        help="Memory efficient linear implementation for Zero Stage 3.",
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help="Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature.",
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help="Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.",
    )
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--offload_reference_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reference model.",
    )
    parser.add_argument(
        "--offload_reward_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reward model.",
    )
    parser.add_argument(
        "--actor_bf16", action="store_true", help="Enable bf16 for actor model"
    )
    parser.add_argument(
        "--reward_bf16", action="store_true", help="Enable bf16 for reward model"
    )
    parser.add_argument(
        "--actor_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model.",
    )
    parser.add_argument(
        "--reward_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Reward model.",
    )
    parser.add_argument(
        "--reference_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Reference model.",
    )
    parser.add_argument(
        "--actor_gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--disable_actor_dropout",
        action="store_true",
        help="Disable the dropout of the actor model.",
    )
    parser.add_argument(
        "--disable_reward_dropout",
        action="store_true",
        help="Disable the dropout of the reward model.",
    )
    parser.add_argument(
        "--use_quality_head",
        action="store_true",
        help="To determine if using the quality head for training.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for MC return estimate.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="kl",
        choices=["kl", "full_kl", "kl_onestep", "full_kl_onestep"],
        help="penalty type.",
    )
    parser.add_argument(
        "--kl_ctl", type=float, default=0.1, help="KL penalty coefficient."
    )
    parser.add_argument(
        "--target_kl", type=float, default=1.0, help="Target KL that to early stop."
    )
    parser.add_argument(
        "--save_at_final", action="store_true", help="Whether save at final step."
    )
    parser.add_argument(
        "--use_multihead_rm", action="store_true", help="Whether to use the multihead reward model."
    )
    parser.add_argument(
        "--normalize_kl",
        action="store_true",
        help="Whether to normalize kl divergence using baseline.",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.0,
        help="Entropy regularization coefficient.",
    )
    parser.add_argument(
        "--clip_reward_value", type=float, default=5.0, help="Reward clipping value."
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--actor_lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--actor_lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial actor LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema", action="store_true", help="Enable EMA checkpoint for the model."
    )
    ## Mixed Precision ZeRO++
    parser.add_argument(
        "--enable_mixed_precision_lora",
        action="store_true",
        help="Enable Mixed Precision ZeRO++ for training and generation.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step3_tensorboard")
    ## Print actor model answers during training
    parser.add_argument(
        "--print_answers",
        action="store_true",
        help="Print prompt and answers during training",
    )
    parser.add_argument(
        "--save_answers",
        action="store_true",
        help="Save prompt and answers during training",
    )
    ## Testing
    parser.add_argument(
        "--enable_test_mode",
        action="store_true",
        help="Enable a testing mode that terminates training based on args.test_stop_step",
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help="Training non-overflow step at which to terminate training during testing.",
    )
    parser.add_argument('--alpha', type=float, default=0.0, help="the coefficient for the length penalty: r(x) = f(x) - L * alpha")
    parser.add_argument('--wandb_exp_name', type=str, help='the wandb experiment name')
    parser.add_argument('--wandb_project_name', type=str, default='rw-dcp-3', help='the wandb experiment name')
    parser.add_argument('--exit_duration_in_mins', type=float, default=float('inf'))
    # parser.add_argument('--last_ckpt_step', type=int, default=0, help='Control the global step')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args


def get_dsp_save_path(args):
    return os.path.join(args.output_dir, 'dsp_state')


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    prompt_train_dataset, prompt_eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.data_seed,
        tokenizer,
        args.max_prompt_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        reload=True,
    )
    _, ppl_eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        1,
        args.data_seed,
        tokenizer,
        args.max_prompt_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        reload=True,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len, args.inference_tp_size)
    # Sampler params
    prompt_train_sampler = DistributedSampler(prompt_train_dataset)
    prompt_eval_sampler = DistributedSampler(prompt_eval_dataset)
    ppl_eval_sampler = DistributedSampler(ppl_eval_dataset)
    if unsupervised_training_enabled:
        unsupervised_train_sampler = DistributedSampler(unsupervised_train_dataset)

    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    prompt_eval_dataloader = DataLoader(
        prompt_eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    ppl_eval_dataloader = DataLoader(
        ppl_eval_dataset,
        collate_fn=default_data_collator,
        sampler=ppl_eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size,
        )
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader
        )  # basically a dummy dataloader

    num_update_steps_per_epoch = (
        min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
        * (args.per_device_generation_batch_size / args.per_device_training_batch_size)
        * args.ppo_epochs
        / args.gradient_accumulation_steps
    )
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return (
        prompt_train_dataloader,
        prompt_eval_dataloader,
        ppl_eval_dataloader,
        unsupervised_train_dataloader,
        num_total_iters,
    )


def save_model(rlhf_engine, tokenizer, args):
    if args.output_dir is not None:
        # save the optimizer and lr scheduler states
        dsp_out_dir = get_dsp_save_path(args)
        print(f'saving rlhf engine checkpoint to {dsp_out_dir}')
        rlhf_engine.save_optimization_states(dsp_out_dir)
        # if torch.distributed.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # torch.distributed.barrier() # remove

        print_rank_0("saving model ...", args.global_rank)
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor, tokenizer, args, sub_folder="actor")
            if args.enable_ema:
                save_hf_format(
                    rlhf_engine.actor_ema, tokenizer, args, sub_folder="actor_ema"
                )

        if args.actor_zero_stage == 3:
            save_zero_three_model(
                rlhf_engine.actor,
                global_rank=args.global_rank,
                save_dir=os.path.join(args.output_dir, "actor"),
                zero_stage=args.actor_zero_stage,
            )
            if args.enable_ema:
                save_zero_three_model(
                    rlhf_engine.actor_ema,
                    global_rank=args.global_rank,
                    save_dir=os.path.join(args.output_dir, "actor_ema"),
                    zero_stage=args.actor_zero_stage,
                )


def save_prompts_and_answers(prompts, answers, rewards, global_step, file_path):
    assert len(prompts) == len(answers), "Mismatched lengths!"
    assert file_path.endswith(".json")
    data = [
        {
            "id": i,
            "global_step": global_step,
            "prompt": prompts[i],
            "answer": answers[i],
            "reward": rewards[i],
        }
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Determine the next id value
        next_id = data[-1]["id"] + 1 if data else 0

        # Create new entries and append them to the data list
        new_entries = [
            {
                "id": next_id + i,
                "global_step": global_step,
                "prompt": prompts[i],
                "answer": answers[i],
                "reward": rewards[i],
            }
            for i in range(len(prompts))
        ]
        data.extend(new_entries)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)


def evaluation_by_reward(
    trainer, eval_dataloader, device, args, global_step, print_answers=False
):
    eval_reward = []
    eval_length = []
    eval_kl = []
    eval_entropy = []
    for step, batch_prompt in enumerate(eval_dataloader):
        batch_prompt = to_device(batch_prompt, device)

        exp = trainer.generate_experience(
            batch_prompt["prompt"],
            batch_prompt["prompt_att_mask"],
            step,
            print_answers,
            training_mode=False,
        )
        reward = exp["rewards"].mean()

        prompt_length = trainer.prompt_length
        start = prompt_length - 1
        action_mask = exp["attention_mask"]
        answer_length = action_mask[:, start:].sum(dim=-1).float().mean()

        if "full_kl" in exp:
            kl = (
                torch.sum(exp["full_kl"][:, start:] * action_mask[:, start:])
                / action_mask[:, start:].sum()
            )
        else:
            kl = (
                torch.sum(
                    (exp["logprobs"][:, start:] - exp["ref_logprobs"][:, start:])
                    * action_mask[:, start:-1]
                )
                / action_mask[:, start:-1].sum()
            )
        if "entropy" in exp:
            entropy = (
                torch.sum(exp["entropy"][:, start:] * action_mask[:, start:])
                / action_mask[:, start:].sum()
            )
        else:
            entropy = torch.zeros(1)

        eval_reward.append(reward.item())
        eval_length.append(answer_length.item())
        eval_kl.append(kl.item())
        eval_entropy.append(entropy.item())

        # save eval result
        if args.save_answers and step < 10:
            assert global_step is not None and args.output_dir is not None
            save_dir = os.path.join(args.output_dir, "evaluation")
            os.makedirs(save_dir, exist_ok=True)

            prompts = trainer.tokenizer.batch_decode(
                exp["input_ids"][:, :prompt_length], skip_special_tokens=True
            )
            answers = trainer.tokenizer.batch_decode(
                exp["input_ids"][:, prompt_length:], skip_special_tokens=True
            )
            rewards = [rew.item() for rew in exp["rewards"]]

            file_path = os.path.join(save_dir, f"rank_{args.local_rank}.json")
            save_prompts_and_answers(prompts, answers, rewards, global_step, file_path)

        if step == 19:
            break
    return (
        np.mean(eval_reward),
        np.mean(eval_length).astype(int),
        np.mean(eval_kl),
        np.mean(eval_entropy),
    )


def evaluation_by_ppl(trainer, eval_dataloader, device):
    tokenizer = trainer.tokenizer

    losses = 0
    for step, batch in enumerate(eval_dataloader):
        trainer.train()
        with torch.no_grad():
            batch = to_device(batch, device)

            output = trainer.actor_model(**batch)
            loss = output.loss
            losses += loss.float()
        if step == 99:
            break

    losses = losses / (step + 1)
    try:
        perplexity = torch.exp(losses)
    except OverflowError:
        perplexity = float("inf")
    try:
        perplexity = get_all_reduce_mean(perplexity).item()
    except:
        pass
    return perplexity


def main():
    args = parse_args()
    dist.init_process_group(backend='nccl')
    if args.enable_tensorboard:
        args.tensorboard_path = args.output_dir

    assert args.gradient_accumulation_steps == 1

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    if args.global_rank == 0:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_exp_name
        )

    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    if args.global_rank == 0:
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path, fast_tokenizer=True)

    (
        prompt_train_dataloader,
        prompt_eval_dataloader,
        ppl_eval_dataloader,
        unsupervised_train_dataloader,
        num_total_iters,
    ) = create_datasets(args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        reward_model_name_or_path=args.reward_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args
    )

    # debugging, remove later
    # save_model(rlhf_engine, tokenizer, args)

    last_ckpt_step = 0
    if os.path.exists(get_dsp_save_path(args)):
        rlhf_engine.load_optimization_states(get_dsp_save_path(args))
        last_ckpt_step = rlhf_engine.actor.global_steps
        print(f'loaded rlhf engine checkpoint from {get_dsp_save_path(args)}, starting from step {last_ckpt_step}')

    # Mixed Precision ZeRO++
    if args.enable_mixed_precision_lora:
        assert (
            args.actor_lora_dim > 0
        ), "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    args.end_of_conversation_token = tokenizer.eos_token  # "<|endoftext|>"

    if unsupervised_training_enabled:
        if args.algo == "remax":
            remax_trainer = DeepSpeedReMaxTrainerUnsupervised
        else:
            raise ValueError(args.algo)
    else:
        if args.algo == "remax":
            remax_trainer = DeepSpeedReMaxTrainer
        else:
            raise ValueError(args.algo)
    trainer = remax_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(
        args.generation_batches, args.per_device_training_batch_size
    )
    unsup_mini_dataset = MiniDataset(
        args.generation_batches, args.per_device_training_batch_size
    )

    # Train!
    global_step = 0
    print_rank_0("***** Running training *****", args.global_rank)
    if last_ckpt_step == 0:
        # Log the metrics at initialization
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {1}/{args.num_train_epochs} *****",
            args.global_rank,
        )
        perplexity = evaluation_by_ppl(trainer, ppl_eval_dataloader, device)
        eval_reward, eval_length, eval_kl, eval_entropy = evaluation_by_reward(
            trainer, prompt_eval_dataloader, device, args, 0, False
        )
        print_rank_0(
            f"eval reward: {eval_reward} | eval length: {eval_length} | eval kl: {eval_kl} | eval entropy: {eval_entropy} | eval ppl: {perplexity}",
            args.global_rank,
        )
        if torch.distributed.get_rank() == 0:
            wandb.log(
                    {
                    "eval_reward": eval_reward,
                    "eval_length": eval_length,
                    "eval_kl": eval_kl,
                    "eval_entropy": eval_entropy,
                    "eval_ppl": perplexity
                    },
                    step=0)

    best_eval_reward = None
    non_overflow_step_count = 0
    early_stopped = False
    should_exit = False
    eval_kl = -1 # need an initial value for early stopping conditions
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank,
        )

        # Set epoch for distributed sampler, to yield a different ordering of data indices for each epoch
        prompt_train_dataloader.sampler.set_epoch(epoch)

        for step, (batch_prompt, batch_unsupervised) in enumerate(
            zip(prompt_train_dataloader, unsupervised_train_dataloader)
        ):
            if global_step < last_ckpt_step: # recover the training step (set up the correct step)
                global_step += 1
                continue

            batch_prompt = to_device(batch_prompt, device)

            for _ in range(args.generation_batches):
                out = trainer.generate_experience(
                    batch_prompt["prompt"],
                    batch_prompt["prompt_att_mask"],
                    global_step,
                    print_answers=args.print_answers and global_step % 20 == 0,
                )

                if batch_unsupervised is not None:
                    batch_unsupervised = to_device(batch_unsupervised, device)
                    unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
                else:
                    unsup_dataset = unsup_mini_dataset.add(
                        [[None] * args.per_device_generation_batch_size]
                    )

                exp_dataset = exp_mini_dataset.add(out)

            training_start = time.time()
            assert exp_dataset is not None
            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, unsup_loss_sum = 0, 0
                average_reward = 0
                average_return = 0
                average_max_return = 0
                average_length = 0
                average_kl = 0
                average_max_kl = 0
                average_entropy = 0
                average_kl_ratio = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                # we manully implement gradient accumulation here
                for i, (exp_data, unsup_data) in enumerate(
                    zip(exp_dataset, unsup_dataset)
                ):
                    actor_loss, actor_return, kl_ratio = trainer.compute_loss(exp_data)

                    trainer.actor_model.backward(actor_loss / len(exp_dataset))

                    actor_loss_sum += actor_loss.item()
                    average_reward += exp_data["rewards"].mean()
                    average_return += actor_return.mean()
                    average_max_return += torch.abs(actor_return).max()
                    average_kl_ratio += kl_ratio

                    prompt_length = trainer.prompt_length
                    start = prompt_length - 1
                    action_mask = exp_data["attention_mask"]
                    answer_length = action_mask[:, start:].sum(dim=-1).float().mean()
                    average_length += answer_length

                    if "full_kl" in exp_data:
                        kl = (
                            torch.sum(
                                exp_data["full_kl"][:, start:] * action_mask[:, start:]
                            )
                            / action_mask[:, start:].sum()
                        )
                        max_kl = torch.max(
                            exp_data["full_kl"][:, start:] * action_mask[:, start:]
                        )
                    else:
                        kl = (
                            torch.sum(
                                (
                                    exp_data["logprobs"][:, start:]
                                    - exp_data["ref_logprobs"][:, start:]
                                )
                                * action_mask[:, start:-1]
                            )
                            / action_mask[:, start:-1].sum()
                        )
                        max_kl = torch.max(
                            (
                                exp_data["logprobs"][:, start:]
                                - exp_data["ref_logprobs"][:, start:]
                            )
                            * action_mask[:, start:-1]
                        )
                    if "entropy" in exp_data:
                        entropy = (
                            torch.sum(
                                exp_data["entropy"][:, start:] * action_mask[:, start:]
                            )
                            / action_mask[:, start:].sum()
                        )
                    else:
                        entropy = torch.zeros(1)
                    average_kl += kl
                    average_entropy += entropy
                    average_max_kl += max_kl

                    if unsupervised_training_enabled:
                        raise NotImplementedError
                        # unsup_loss = trainer.train_unsupervised(
                        #     unsup_data, args.unsup_coef)
                        # unsup_loss_sum += unsup_loss.item()

                    inner_iter += 1
                    if args.enable_ema:
                        raise NotImplementedError

                # ==== perform batch_update here (i.e., gradient checkpointing for on-policy)
                trainer.actor_model.step()
                # ====

                end = time.time()
                training_time = end - training_start
                e2e_time = (
                    training_time + trainer.generate_time * args.generation_batches
                )  # it is an approximation, we did not include, e.g., rw forward time etc

                print_rank_0(
                    f"Epoch: {epoch + 1}/{args.num_train_epochs} | Step: {step}/{len(prompt_train_dataloader)} |  Actor Loss: {actor_loss_sum / inner_iter} | Unsupervised Loss: {unsup_loss_sum / inner_iter}",
                    args.global_rank,
                )
                print_throughput_step3(
                    rlhf_engine.actor.module,
                    args,
                    e2e_time,
                    trainer.generate_time,
                    training_time,
                    args.global_rank,
                )
                average_reward = get_all_reduce_mean(average_reward).item()
                average_return = get_all_reduce_mean(average_return).item()
                average_length = get_all_reduce_mean(average_length).item()
                average_kl = get_all_reduce_mean(average_kl).item()
                average_entropy = get_all_reduce_mean(average_entropy).item()
                average_max_kl = get_all_reduce_mean(average_max_kl).item()
                average_max_return = get_all_reduce_mean(average_max_return).item()
                average_kl_ratio = get_all_reduce_mean(average_kl_ratio).item()
                print_rank_0(
                    f"Average reward score: {average_reward / inner_iter} Average return: {average_return / inner_iter} Average length: {average_length / inner_iter:.0f} Average kl: {average_kl / inner_iter} Average entropy: {average_entropy / inner_iter} Max kl: {average_max_kl / inner_iter} Max return: {average_max_return / inner_iter} KL ratio: {average_kl_ratio / inner_iter}",
                    args.global_rank,
                )
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank,
                )

                if torch.distributed.get_rank() == 0:
                    # store the intermediate results in the wandb
                    wandb.log(
                        {
                            "train_reward": average_reward / inner_iter,
                            "train_return": average_return / inner_iter,
                            "train_length": average_length / inner_iter,
                            "train_kl": average_kl / inner_iter,
                            "train_entropy": average_entropy / inner_iter,
                            "train_actor_loss": actor_loss,
                            "train_actor_loss_sum": actor_loss_sum,
                            "train_max_kl": average_max_kl / inner_iter,
                            "train_max_return": average_max_return / inner_iter,
                            "train_kl_ratio": average_kl_ratio / inner_iter,
                            "lr": rlhf_engine.actor.optimizer.param_groups[0]['lr']
                        },
                        step=epoch*len(prompt_train_dataloader) + global_step
                    )
            if (global_step + 1) % (100) == 0:
                print_rank_0(
                    f"***** Evaluating policy, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(prompt_train_dataloader)} *****",
                    args.global_rank,
                )
                perplexity = evaluation_by_ppl(trainer, ppl_eval_dataloader, device)
                eval_reward, eval_length, eval_kl, eval_entropy = evaluation_by_reward(
                    trainer, prompt_eval_dataloader, device, args, global_step, False
                )
                print_rank_0(
                    f"eval reward: {eval_reward} | eval length: {eval_length} | eval kl: {eval_kl} | eval entropy: {eval_entropy} | eval ppl: {perplexity}",
                    args.global_rank,
                )
                if torch.distributed.get_rank() == 0:
                    wandb.log(
                        {
                            "eval_reward": eval_reward,
                            "eval_length": eval_length,
                            "eval_kl": eval_kl,
                            "eval_entropy": eval_entropy,
                            "eval_ppl": perplexity
                        },
                        step=global_step
                    )

                if best_eval_reward is None:
                    best_eval_reward = eval_reward
                if eval_reward >= best_eval_reward:
                    best_eval_reward = eval_reward
                    save_model(rlhf_engine, tokenizer, args)

            global_step += 1
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow = trainer.get_overflow()

            if not actor_overflow:
                non_overflow_step_count += 1

            # if eval_kl >= args.target_kl:
            #     print_rank_0(
            #         f"**** Early stop at {global_step} due to KL = {eval_kl} > Target KL ({args.target_kl}) ****"
            #     )
            #     early_stopped = True
            #     break

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break
            # if dist.get_rank() == 0:
            #     import pdb; pdb.set_trace()
            # torch.distributed.barrier()

            print_rank_0(f"Timer's training time: {(time.time() - _TRAIN_START_TIME) / 60.0 }")
            if (time.time() - _TRAIN_START_TIME) / 60.0 > args.exit_duration_in_mins:
                print("Exiting and saving model...")
                save_model(rlhf_engine, tokenizer, args)
                should_exit = True
                break

        if args.enable_test_mode or should_exit:
            break

    # Final
    if not early_stopped or args.save_at_final:
        print_rank_0(f"***** Evaluating at final *****", args.global_rank)
        perplexity = evaluation_by_ppl(trainer, ppl_eval_dataloader, device)
        eval_reward, eval_length, eval_kl, eval_entropy = evaluation_by_reward(
            trainer, prompt_eval_dataloader, device, args, global_step, False
        )
        print_rank_0(
            f"eval reward: {eval_reward} | eval length: {eval_length} | eval kl: {eval_kl} | eval entropy: {eval_entropy} | eval ppl: {perplexity}",
            args.global_rank,
        )
        if torch.distributed.get_rank() == 0:
            wandb.log(
                {"eval_reward": eval_reward,
                   "eval_length": eval_length,
                   "eval_kl": eval_kl,
                   "eval_entropy": eval_entropy,
                   "eval_ppl": perplexity
                },
                step=global_step)
            # save the state of the optimizer here.



if __name__ == "__main__":
    main()
