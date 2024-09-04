import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MultiheadRewardModel(nn.Module):
    def __init__(
            self, base_model, tokenizer, num_padding_at_beginning=0,
            rm_num_heads=1, rm_l1_reg=0., rm_ortho_reg=0.,
            correlation_with_length=0., attribute_corr=0.,
            normalized_proj=False, epsilon=-1, alpha=0.0, use_quality_head=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.rm_num_heads = rm_num_heads
        self.rm_l1_reg = rm_l1_reg
        self.correlation_with_length = correlation_with_length
        self.attribute_corr = attribute_corr
        self.rm_ortho_reg = rm_ortho_reg
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
        self.use_quality_head = use_quality_head
        self.normalized_proj = normalized_proj
        self.epsilon = epsilon
        

        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

        self.alpha = alpha

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
        
        # using the last dim as the final reward for training the last dim in the multihead rw
        chosen_rewards = rewards[:bs][:,-1]
        rejected_rewards = rewards[bs:][:,-1]

        # we would like the final dimension denotes the general reward (change the objective here)
        # write a loss penalizing the labels
        loss = 0
        if self.epsilon >= 0:
            # Use epsilon-greedy as the main objective.
            # We assume each labeler picks the chosen response because it is better in certain aspect
            # Then in the greedy mode, we sample from the heads where the reward of chosen is higher than the rejected.
            # With probability epsilon, we randomly pick one head to optimize.
            head_diff = rewards[:bs] - rewards[bs:]
            if torch.rand(1).item() < self.epsilon:
                # Exploration mode
                head_idxes = torch.randint(0, self.rm_num_heads, (bs, 1)).to(rewards.device)
            else:
                # More significant differences can be more noticable
                probs = F.softmax(head_diff, dim=-1)
                head_idxes = torch.multinomial(probs, num_samples=1).to(rewards.device)
            chosen_diffs = torch.gather(head_diff, dim=1, index=head_idxes)
            loss = -F.logsigmoid(chosen_diffs).mean()
        else:
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        #     # Add L1 regularization on the rewards
        #     # We want the estimated reward of each example to be sparse.
        #     # We ditch this for the epsilon-greedy approach.
        #     l1_norm = rewards.abs().sum() / input_ids.shape[0]
        #     l1_loss = self.rm_l1_reg * l1_norm
        #     loss += l1_loss

        # Add orthogonal regularization on the projection layer weights (without ortho)
        prod = w_normalized @ w_normalized.T
        mean_corr = torch.abs(torch.triu(prod, diagonal=1)).sum() / (self.rm_num_heads * (self.rm_num_heads-1) / 2)
        ortho_loss = self.rm_ortho_reg * mean_corr
        loss += ortho_loss

        # if self.correlation_with_length > 0.0:
        #     # normalized_seqlens = ((seqlens.float() - self.mean) / self.std).float() # normalize the sequence length (x - mean) / std
        #     # correlation_with_length = (rewards[:, -2] * normalized_seqlens.unsqueeze(-1)).mean()
        #     combined = torch.stack([seqlens, rewards[:, -2]], dim=0)
        #     def pearson_correlation(x, y):
        #         mean_x = torch.mean(x)
        #         mean_y = torch.mean(y)
        #         num = torch.sum((x - mean_x) * (y - mean_y))
        #         den = torch.sqrt(torch.sum((x - mean_x) ** 2) * torch.sum((y - mean_y) ** 2))
        #         return num / (den + 1e-8)
        #     correlation_with_length = pearson_correlation(combined[0], combined[1])
        #     # correlation_with_length = torch.cov(combined)[0,1]
        #     # Regularization term
        #     loss_correlation = 1 - correlation_with_length # make sure it is in the [0, 1]
        #     length_loss = loss_correlation * self.correlation_with_length
        #     loss += length_loss

        regression_loss = 0.0
        if self.attribute_corr > 0.0 and (labels == -1.0).all():
            attribute_predictions = self.v_head(feats)
            labels = torch.reshape(labels, (2, -1)).to(dtype=torch.float16)
            regression_loss = F.mse_loss(attribute_predictions[:, :13], labels) # Attributes are 13-d, so the first 13-d of rewards related to each attribute , using the cross_entropy loss
            loss += self.attribute_corr * regression_loss
        # 100 samples, 10 heads, 100 * 10 reward predictions, 100 lengths, 10 dimensional vector, whether it's sparse
        # 100 x n attributes (potentially missing), 10 x n correlations, check sparsity of each n dimensional vector

        return {
            "seqlens": seqlens,
            "v_head": self.v_head(feats),
            "length_reward": rewards[:, -1],
            "loss": loss,
            "rewards": rewards,
            "ortho_loss": mean_corr,
            "regression_loss": regression_loss, 
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
        if self.use_quality_head:
            values = self.v_head(hidden_states)[:,:,0]
        else:
            values = self.v_head(hidden_states).sum(-1)
        # if return_value_only:
        #     return values
        # else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
        assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_end_scores = []  # we use this name for consistency with the original forward function
        response_lengths = []
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() + prompt_length if len(
                c_inds) > 0 else seq_len
            
            response_length = c_ind - prompt_length
            response_lengths.append(response_length)
            chosen_end_scores.append(value[c_ind - 1] - response_length * self.alpha)

        # Convert response_lengths to a tensor and ensure it's on the same device as values
        response_lengths_tensor = torch.tensor(response_lengths, device=values.device).unsqueeze(1)

        # Compute r(x) for each item in the batch
        r_x = values - response_lengths_tensor * self.alpha

        if return_value_only:
            return r_x
        else:
            return {
                "values": r_x,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }