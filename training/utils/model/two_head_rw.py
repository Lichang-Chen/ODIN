import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TwoHeadRewardModel(nn.Module):
    def __init__(
            self, base_model, tokenizer, num_padding_at_beginning=0,
            rm_num_heads=1, rm_l1_reg=0., rm_ortho_reg=0.,
            correlation_with_length=0., attribute_corr=0.,
            normalized_proj=True, epsilon=-1, batch_size=1, UCB=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.rm_num_heads = rm_num_heads
        self.rm_l1_reg = rm_l1_reg
        self.ucb = UCB
        self.correlation_with_length = correlation_with_length
        self.attribute_corr = attribute_corr
        self.rm_ortho_reg = rm_ortho_reg
        self.counts = torch.zeros(batch_size, self.rm_num_heads)
        self.sum_rewards = torch.zeros(batch_size, self.rm_num_heads)

        self.mean = 353.03467029231814 # mean for openassistant dataset
        self.std =  323.78544100220984 # std for openassistant dataset
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    self.rm_num_heads,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, self.rm_num_heads, bias=False)

        self.normalized_proj = normalized_proj
        self.epsilon = epsilon

        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def proj_with_normalized_weight(self, x):
        w = self.v_head.weight - torch.mean(self.v_head.weight, dim=-1, keepdim=True)
        head_norms = torch.norm(w, dim=-1, keepdim=True)
        head_norms = torch.maximum(head_norms, torch.full_like(head_norms, 1e-8))
        w = w / head_norms
        # Return the weight for orthogonal regularizations
        return w, F.linear(x, w)

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                use_cache=False):
        # num_padding_at_beginning > 0 is only adopted by OPT models and we don't need it here.
        assert self.num_padding_at_beginning == 0
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        # Split the inputs and rewards into two parts, chosen and rejected
        seqlens = (input_ids != self.PAD_ID).sum(axis=-1)
        feats = hidden_states[torch.arange(hidden_states.shape[0]), seqlens-1]

        if self.normalized_proj:
            w_normalized, rewards = self.proj_with_normalized_weight(feats)
        else:
            rewards = self.v_head(feats).squeeze(-1)
            head_norms = torch.norm(self.v_head.weight, dim=-1, keepdim=True)
            norms = torch.maximum(head_norms, torch.full_like(head_norms, 1e-8))
            w_normalized = self.v_head.weight / norms

        # The easy version: only unsupervised learn two heads 
        # TODO: sum may have some issues
        chosen_rewards = rewards[:bs].sum(axis=-1)
        rejected_rewards = rewards[bs:].sum(axis=-1)
        head_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        if self.epsilon > 0:
            # Use epsilon-greedy as the main objective.
            # We assume each labeler picks the chosen response because it is better in certain aspect
            # Then in the greedy mode, we sample from the heads where the reward of chosen is higher than the rejected.
            # With probability epsilon, we randomly pick one head to optimize.
            if torch.rand(1).item() < self.epsilon:
                # Exploration mode
                head_idxes = torch.randint(0, self.rm_num_heads, (bs, 1)).to(rewards.device)
            else:
                # More significant differences can be more noticable
                probs = F.softmax(head_diff, dim=-1)
                head_idxes = torch.multinomial(probs, num_samples=1).to(rewards.device)
            # TODO: revise the function
            chosen_diffs = torch.gather(head_diff, dim=1, index=head_idxes)
            loss = -F.logsigmoid(chosen_diffs).mean()
        else:
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # Add orthogonal regularization on the projection layer weights
        prod = w_normalized @ w_normalized.T
        mean_corr = torch.abs(torch.triu(prod, diagonal=1)).sum() / (self.rm_num_heads * (self.rm_num_heads-1) / 2)
        ortho_loss = self.rm_ortho_reg * mean_corr
        loss += ortho_loss

        return {
            "seqlens": seqlens,
            "v_head": self.v_head(feats),
            "length_reward": rewards[:, -1], # we assume the last dim is the length reward
            "chosen_quality_reward": rewards[:bs, 0],
            "rejected_quality_reward": rewards[bs:, 0],
            "loss": loss,
            "rewards": rewards,
            "ortho_loss": mean_corr,
            "regression_loss": 0, 
            "chosen_mean_scores": chosen_rewards,
            "rejected_mean_scores": rejected_rewards,
            "chosen_rewards": chosen_rewards,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):
        """Function used in RL for value network and the reward model.
        return_value_only: if True, then return estimated rewards of all tokens for the value estimate.
        """
        # TODO: check the dimensions for multi-head rewards, add functions to do weighted sums
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        if self.rm_num_heads > 1:
            values = self.v_head(hidden_states)[:,:,0]
        else:
            values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }