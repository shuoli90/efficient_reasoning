from trl import SFTTrainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficient_reasoning.utils import evaluate, Benchmark
from tqdm import tqdm

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


def _get_per_token_logps(model, input_ids, attention_mask, labels):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    return selective_log_softmax(logits, labels)


class ASFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        """Ensure `advantages` is retained in the dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Ensure `advantages` is retained in the dataloader"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        labels, completion_mask = inputs['labels'][:, 1:], inputs['completion_mask'][:, 1:]
        # labels = F.pad(labels, (0, 1), value=self.processing_class.pad_token_id)
        # shift_labels = labels[..., 1:].contiguous()
        labels = labels[:, 1:]
        advantages = inputs['advantages'][:, 1:]
        per_token_logps = _get_per_token_logps(model, input_ids, attention_mask, labels) * completion_mask
        loss = -(per_token_logps * advantages).sum() / completion_mask.sum()
        return loss
    
    def evaluate(
        self,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(self.eval_dataset), self.args.eval_batch_size), total=len(self.eval_dataset) // self.args.eval_batch_size):
                batch = self.eval_dataset[i:i+self.args.eval_batch_size]
                # problems = batch['problem']
                # targets = batch['answer']
                problems = batch['problem']
                targets = [item['answer'] for item in problems]
                problems = [item['problem'] for item in problems]
                outputs = self.processing_class(problems, return_tensors='pt', padding=True, padding_side='left')
                outputs = self.model.generate(
                    input_ids=outputs['input_ids'].to(self.model.device),
                    attention_mask=outputs['attention_mask'].to(self.model.device),
                    temperature=0.7,
                    top_p=0.95,
                    max_new_tokens=512,
                )
                responses = [self.processing_class.decode(output, skip_special_tokens=True) for output in outputs]
                rewards = evaluate('MATH-500', responses, targets)
                correct += sum(rewards)
        # log the accuracy
        accuracy = correct / len(self.eval_dataset)
        metrics = {f"{metric_key_prefix}_acc": accuracy}
        self.log(metrics)
        self.model.train()
        return accuracy
    

class QSFTTrainer(ASFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        labels, completion_mask = inputs['labels'][:, 1:], inputs['completion_mask'][:, 1:]
        advantages = 1-inputs['advantages'][:, 1:]
        per_token_logps = _get_per_token_logps(model, input_ids, attention_mask, labels) * completion_mask
        loss = -(per_token_logps * advantages).sum() / completion_mask.sum()
        return loss


class NSFTTrainer(ASFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        labels, completion_mask = inputs['labels'][:, 1:], inputs['completion_mask'][:, 1:]
        # labels = F.pad(labels, (0, 1), value=self.processing_class.pad_token_id)
        # shift_labels = labels[..., 1:].contiguous()
        labels = labels[:, 1:]
        per_token_logps = _get_per_token_logps(model, input_ids, attention_mask, labels) * completion_mask
        loss = -(per_token_logps).sum() / completion_mask.sum()
        return loss