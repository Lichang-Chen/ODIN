python evaluation/lima_gen.py --model_name_or_path_finetune "path/to/finetuned/model" --save_path "path/to/save/generation/results"

# merge answers
python merge.py --file1 "file1" --file2 "file2" \
    --output_path eval_data/merged-v2/${MODEL}.json --files_path eval_data/generation/

# # gpt-4 evaluate
for eval_file in "eval_data/merged-v2/${MODEL}.json"
do
    python gpt4-eval.py -qa ${eval_file} -k1 key1 -k2 key2  --batch_size 5 --max_tokens 32 --output_dir "output"
    python gpt4-eval.py -qa ${eval_file} -k1 key2 -k2 key1 --batch_size 5 --max_tokens 32 --output_dir "output"
done