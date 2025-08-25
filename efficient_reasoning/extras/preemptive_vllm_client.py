from collections import deque
from typing import List, Dict, Any, Sequence, Iterator
import itertools
from efficient_reasoning.extras.multi_vllm_client import MultiVLLMClient
from transformers import AutoTokenizer
from tqdm import tqdm 
from datasets import Dataset
from torch.utils.data import Sampler

class PreemptiveMultiVLLMClient:
    """
    Wraps MultiVLLMClient to pre‐sample completions for the next
    `gradient_accumulation_steps` macro‐batches, based on a known
    prompt list and sampler order.
    """
    def __init__(
        self,
        server_configs: List[Dict[str, Any]],
        gradient_accumulation_steps: int,
        train_dataset: Dataset,      # <-- take a Dataset
        sampler: Sampler,            # <-- take a Sampler
        batch_size: int,
        num_generations: int,
        num_iterations: int,
        per_device_train_batch_size: int,
        preemptive_steps: int = 0,
        iw: bool = False,
        iw_first_step: bool = True,
    ):
        self.iw = iw
        self.iw_first_step = iw_first_step
        # Underlying multi‐client
        self.multi_client = MultiVLLMClient(server_configs)
        
        # How many macro‐batches to pre‐fetch
        self.grad_accum_steps = gradient_accumulation_steps

        self.preemptive_steps = preemptive_steps
        
        # materialize the prompts column into a list of str
        self.train_prompts = train_dataset["prompt"]

        # materialize the sampler into a list of indices
        # print(f"Sampler: {list(sampler)[:300]}")
        list_temp = list(sampler)[::num_generations] # remove the extra num_generations of samples (can't use training dataset because of potentially shuffling)
        # print(f"Sampler after removing extra num_generations: {list_temp[:300]}")
        list_final = [x for idx, x in enumerate(list_temp) if (idx // (per_device_train_batch_size * gradient_accumulation_steps)) % num_iterations == 0] # remove the extra num_iterations of samples
        # print(f"Sampler after removing extra num_iterations: {list_final[:300]}")
        self.sampler_cycle = itertools.cycle(list_final)
        # print(f"Sampler after removing extra num_iterations: {list_final[:1200]}")
        # for _ in range(8000):
        #     next(self.sampler_cycle) 
        # print(list(sampler))
        # print(list(sampler)[::num_generations])
        self.batch_size = batch_size
        
        # Buffer to hold all pre‐fetched token‐ID lists
        self.buffer = deque()
        self.current_cycle_calls = 0
        self.current_sampling_kwargs: Dict[str, Any] = {}

    def _next_batch_prompts(self) -> List[str]:
        """Pull the next `batch_size` prompts from our sampler cycle."""
        idxs = [next(self.sampler_cycle) for _ in range(self.batch_size)]
        return [self.train_prompts[i] for i in idxs]

    def generate(self, prompts: List[str], n: int, read_from_file: bool = False,  eval_only: bool = False, **sampling_kwargs) -> List[List[int]]:
        """
        On the very first call of each gradient‐accumulation cycle (or if
        sampling hyperparameters change), grab
          `(gradient_accumulation_steps * batch_size)` prompts
        enqueued via our sampler, and do one big `multi_client.generate(...)`
        to get `n` completions for each of them.  Zipper those into `self.buffer`.
        
        On each call, slice out the next `batch_size * n` token‐ID lists
        and return them.
        """
        if self.iw:
            if self.iw_first_step:
                self.iw_first_step = False

                if not read_from_file:
                    all_samples = self.multi_client.generate(
                        self.train_prompts,
                        n=n,
                        **sampling_kwargs
                    )
                    
                    with open("buffer.txt", "w") as f:
                        for sample in all_samples:
                            f.write(f"{sample}\n")
                
                else: 
                    # Read from file
                    all_samples = []
                    with open("buffer.txt", "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                sample = eval(line)
                                all_samples.append(sample)

                if len(all_samples) != 12000 * n:
                    raise ValueError(
                        f"Expected {12000 * n} generation lists, got {len(all_samples)}"
                    )
                
                self.buffer.clear()
                self.buffer.extend(all_samples)

            slice_size = len(prompts) * n
            if len(self.buffer) < slice_size:
                raise RuntimeError(f"Buffer underflow: not enough samples: remaining buffer size {len(self.buffer)} < requested slice size {slice_size}")
            next_batch = [self.buffer.popleft() for _ in range(slice_size)]
            return next_batch

        # 1) Refill if at start of cycle, or if sampling settings have changed
        if self.current_cycle_calls == 0:
            self.current_sampling_kwargs = sampling_kwargs.copy()
            
            # Gather all prompts for the cycle
            all_prompts: List[str] = []
            if self.preemptive_steps > 0:
                for _ in range(self.preemptive_steps):
                    batch_prompts = self._next_batch_prompts()
                    all_prompts.extend(batch_prompts)
            else:
                for _ in range(self.grad_accum_steps):
                    batch_prompts = self._next_batch_prompts()
                    all_prompts.extend(batch_prompts)
            
            # Sanity check: the very first macro‐batch should match `prompts`
            if all_prompts[: len(prompts)] != prompts:
                print("Fetched prompts:", all_prompts[: len(prompts)])
                print("Provided prompts:", prompts)
                raise ValueError(
                    "Sampler out of sync: fetched prompts != provided prompts"
                )
            
            # One big call for everything
            all_samples = self.multi_client.generate(
                all_prompts,
                n=n,
                **sampling_kwargs
            )
            expected = len(all_prompts) * n
            if len(all_samples) != expected:
                raise ValueError(
                    f"Expected {expected} token lists, got {len(all_samples)}"
                )
            
            # Load into buffer
            self.buffer.clear()
            self.buffer.extend(all_samples)
            self.current_cycle_calls = 0

        # 2) Now slice out this step’s completions
        slice_size = len(prompts) * n
        if len(self.buffer) < slice_size:
            raise RuntimeError("Buffer underflow: not enough samples")
        next_batch = [self.buffer.popleft() for _ in range(slice_size)]
        self.current_cycle_calls += 1

        # 3) If we’ve completed the full micro‐batch cycle, reset
        if self.preemptive_steps > 0:
            if self.current_cycle_calls >= self.preemptive_steps:
                self.current_cycle_calls = 0
        else:
            if self.current_cycle_calls >= self.grad_accum_steps:
                self.current_cycle_calls = 0

        return next_batch

    def reset_buffer(self):
        """Clear out everything and restart the sampler iterator."""
        self.buffer.clear()
        self.current_cycle_calls = 0
        self.sampler_cycle = itertools.cycle(self.sampler_cycle)

    def update_named_param(self, name: str, weights: Any):
        """Broadcast a single parameter update."""
        return self.multi_client.update_named_param(name, weights)

    def update_model_params(self, model: Any):
        """Broadcast full model parameter update."""
        return self.multi_client.update_model_params(model)

    def reset_prefix_cache(self):
        """Reset the prefix cache on all servers."""
        return self.multi_client.reset_prefix_cache()


def main():
    # Load prompts from file
    data = []
    with open("/home/leoh/efficient_reasoning/data/MATH-500/train.jsonl") as f:
        for line in f:
            data.append(eval(line)["problem"])
    prompts_list = data[:100]

    server_configs = [
        {"host": "127.0.0.1", "server_port": 8000, "group_port": 51216},
        {"host": "127.0.0.1", "server_port": 8010, "group_port": 51217},
    ]
    batch_size = 2
    grad_accum_steps = 30  # prefetch entire epoch

    pre_client = PreemptiveMultiVLLMClient(
        server_configs,
        gradient_accumulation_steps=grad_accum_steps,
        train_prompts=prompts_list,
        sampler_indices=list(range(len(prompts_list))),
        batch_size=batch_size,
    )

    n = 8
    sampling_kwargs = {
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 32,
    }

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    for i in tqdm(range(0, len(prompts_list), batch_size), desc="Generating Preemptive"):
        prompts = prompts_list[i:i+batch_size]
        completions = pre_client.generate(prompts, n, **sampling_kwargs)
        breakpoint()

    print("Preemptive test completed.")

if __name__ == "__main__":
    main()
