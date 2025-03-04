from trl import SFTTrainer
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class ASFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        """Ensure `advantages` is retained in the dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Ensure `advantages` is retained in the dataloader"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
        )
        
    def _get_per_token_logps(self, model, input_ids, attention_mask, prediction_ids, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep).logits
        return selective_log_softmax(logits, prediction_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, attention_mask, = inputs['input_ids'], inputs['attention_mask']
        completion_ids = inputs['labels']
        # rotate the attention mask by 1
        completion_mask = torch.roll(attention_mask, shifts=1, dims=1)
        advantages = inputs['advantages']
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, prompt_ids, attention_mask, completion_ids, logits_to_keep)
        loss = -((per_token_logps * advantages) * completion_mask).sum() / completion_mask.sum()
        return loss

class QSFTTrainer(ASFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, attention_mask, = inputs['input_ids'], inputs['attention_mask']
        completion_ids = inputs['labels']
        # rotate the attention mask by 1
        completion_mask = torch.roll(attention_mask, shifts=1, dims=1)
        advantages = 1 - inputs['q_value']
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, prompt_ids, attention_mask, completion_ids, logits_to_keep)
        loss = -((per_token_logps * advantages) * completion_mask).sum() / completion_mask.sum()
        return loss

class NSFTTrainer(ASFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, attention_mask, = inputs['input_ids'], inputs['attention_mask']
        completion_ids = inputs['labels']
        # rotate the attention mask by 1
        completion_mask = torch.roll(attention_mask, shifts=1, dims=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, prompt_ids, attention_mask, completion_ids, logits_to_keep)
        loss = -(per_token_logps * completion_mask).sum() / completion_mask.sum()
        return loss