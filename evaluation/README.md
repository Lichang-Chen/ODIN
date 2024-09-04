
# <p align="center">Evaluation</p>
The evaluation folder contains:
- The generation of the [LIMA](https://huggingface.co/datasets/GAIR/lima) test set.
- code for GPT-4 evaluation. (compare with baseline, can also be side-by-side comparison: gpt4-eval.py)
- code for merging two answers
- code for evaluation (all in one)


## Generation
```
% Generate Answers of the trained policy for Lima Test Set
python evaluation/lima_gen.py --model_name_or_path_finetune "path/to/finetuned/model" --save_path "path/to/save/generation/results"
```

## The GPT-4 evaluation
Since the GPT-4 eval has positional bias, thus we need to evaluate the generations from two models in both orders, i.e., place A after B and place A before B.  
```
python gpt4-eval.py -qa ${eval_file} -k1 key1 -k2 key2  --batch_size 5 --max_tokens 32 --output_dir "output"
python gpt4-eval.py -qa ${eval_file} -k1 key2 -k2 key1 --batch_size 5 --max_tokens 32 --output_dir "output"
```

## Merge two answers
```
python merge.py --file1 "file1" --file2 "file2" \
    --output_path eval_data/merged-v2/${MODEL}.json --files_path eval_data/generation/
```

## Evaluations (All in one)
```
bash evaluation/eval.sh
```






