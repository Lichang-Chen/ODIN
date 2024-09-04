import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from training.utils.utils import load_hf_tokenizer
from training.utils.model.model_utils import create_hf_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import json
import torch
import argparse
from tqdm import tqdm

def evaluation_on_lima(args):
    prompts = json.load(open("./lima.json"))
    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_finetune, fast_tokenizer=True)
    tokenizer.padding_side = "left"

    model_fintuned = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path_finetune, tokenizer, None
    )
    model_fintuned.to(device)

    answers = []
    batch_size = 10
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = [
            "Human: {}\n\nAssitant:".format(content[0])
            for content in prompts[i : i + batch_size]
        ]
        batch_inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        # use top_p=0.8, temperature=0.8 for generation
        outputs = model_fintuned.generate(
            **batch_inputs,
            max_length=1024,
            temperature=0.8,
            top_p=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
        prompt_length = batch_inputs["input_ids"].shape[1]
        answers.extend(
            tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        )

    assert len(prompts) == len(answers), "Mismatched lengths!"

    data = [
        {"id": i, "prompt": prompts[i][0], "answer": answers[i]}
        for i in range(len(prompts))
    ]

    if args.save_path is not None:
        with open(f"{args.save_path}", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    else:
        with open("lima_test.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate the finetuned models")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the generation results.",
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        default=None,
        help="The training_data_source_name.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    evaluation_on_lima(args)