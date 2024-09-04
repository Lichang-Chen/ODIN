# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class OASSTDATASET(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)
        self.dataset_name = "OASST"
        self.dataset_name_clean = "oasst"
        # Split the dataset into train (60%) and temp (20%)
        split_data = self.dataset["train"].train_test_split(test_size=0.2)
        self.train_set = split_data["train"]
        self.val_set = split_data['test']
        
    def get_train_data(self):
        return self.train_set

    def get_eval_data(self):
        return self.val_set

    def get_prompt(self, sample):
        return sample['context']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']



class OASST_MULTI_TURN(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = "oasst"
        self.dataset_name_clean = "oasst"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path+'/oasst_data/oasst_en_train_dialogue.json',
                                             "eval":
                                             chat_path+'/oasst_data/oasst_en_val_dialogue.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return sample['prompt'] + "\nAssistant: "
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return " " + sample['chosen']
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample['rejected'] is not None:
            return " " + sample['rejected']
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return " " + sample['prompt'] + "\nAssistant: " + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        if sample['prompt'] is not None and sample['rejected'] is not None:
            return " " + sample['prompt'] + "\nAssistant: " + sample['rejected']
        return None

class OASST_PROMPT(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = "oasst"
        self.dataset_name_clean = "oasst"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path+'/oasst_data/oasst_en_train_prompt.json',
                                             "eval":
                                             chat_path+'/oasst_data/oasst_en_val_prompt.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return sample['prompt'] + "\nAssistant: "
        return None

    # def get_prompt_and_rejected(self, sample):
    #     if sample['prompt'] is not None and sample['rejected'] is not None:
    #         return " " + sample['prompt'] + "\nAssistant: " + sample['rejected']
    #     return None
    
    # def get_chosen_attributes(self, sample):
    #     if sample['chosen_labels'] is not None:
    #         return sample['chosen_labels']
    #     return None