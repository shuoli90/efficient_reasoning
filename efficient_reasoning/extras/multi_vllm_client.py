import threading
from typing import List, Dict, Any
import torch
from .vllm_client import VLLMClient
from transformers import AutoTokenizer  # Import the tokenizer

class MultiVLLMClient:
    def __init__(self, server_configs: List[Dict[str, Any]]):
        """
        Initialize multiple vLLM clients.
        Args:
            server_configs: List of dictionaries, each containing server settings.
                Example:
                [{"host": "127.0.0.1", "server_port": 8000, "group_port": 51216},
                 {"host": "127.0.0.1", "server_port": 8010, "group_port": 51217}]
        """
        self.clients = []
        for config in server_configs:
            client = VLLMClient(
                host=config.get("host", "0.0.0.0"),
                server_port=config.get("server_port", 8000),
                group_port=config.get("group_port", 51216),
                connection_timeout=config.get("connection_timeout", 0.0)
            )
            self.clients.append(client)
    
    def generate(self, prompts: List[str], n: int, **sampling_kwargs) -> List[int]:
        """
        For each prompt, split the total number of completions among all clients.
        Each client receives the entire list of prompts and is assigned a portion of completions.
        The final result is a flat list where the completions for prompt i occupy the slice
        [i * n, (i + 1) * n].
        """
        num_clients = len(self.clients)
        # Determine completions per client for each prompt.
        base = n // num_clients
        remainder = n % num_clients
        completions_per_client = [base + (1 if i < remainder else 0) for i in range(num_clients)]
        
        results_per_client = [None] * num_clients

        def worker(idx, client, prompts, n_client):
            # Modify the sampling parameters for this client to add a unique seed.
            local_sampling_kwargs = sampling_kwargs.copy()
            base_seed = local_sampling_kwargs.get("seed", 1234)
            local_sampling_kwargs["seed"] = base_seed + idx
            try:
                print(
                    f"Client {idx} ({client.host}:{client.server_port}) generating {n_client} completions per prompt with seed {local_sampling_kwargs['seed']}"
                )
                res = client.generate(prompts=prompts, n=n_client, **local_sampling_kwargs)
                results_per_client[idx] = res
            except Exception as e:
                raise ValueError(f"Generation failed for client {client.host}:{client.server_port}: {e}")

        threads = []
        for i, (client, n_client) in enumerate(zip(self.clients, completions_per_client)):
            t = threading.Thread(target=worker, args=(i, client, prompts, n_client))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        
        # Combine results across clients.
        final_results = []
        num_prompts = len(prompts)
        for i in range(num_prompts):
            combined = []
            for client_idx, n_client in enumerate(completions_per_client):
                client_results = results_per_client[client_idx]
                if client_results is None:
                    raise ValueError(f"Client {client_idx} did not return results.")
                start = i * n_client
                end = (i + 1) * n_client
                combined.extend(client_results[start:end])
            if len(combined) != n:
                raise ValueError(f"Expected {n} completions for prompt {i}, got {len(combined)}")
            final_results.extend(combined)
        
        return final_results

    def update_named_param(self, name: str, weights: torch.Tensor):
        threads = []
        def worker(client):
            try:
                client.update_named_param(name, weights)
            except Exception as e:
                print(f"Update failed for client {client.host}:{client.server_port}: {e}")
        for client in self.clients:
            t = threading.Thread(target=worker, args=(client,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def update_model_params(self, model: torch.nn.Module):
        threads = []
        def worker(client):
            try:
                client.update_model_params(model)
            except Exception as e:
                print(f"Update model params failed for client {client.host}:{client.server_port}: {e}")
        for client in self.clients:
            t = threading.Thread(target=worker, args=(client,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def reset_prefix_cache(self):
        threads = []
        def worker(client):
            try:
                client.reset_prefix_cache()
            except Exception as e:
                print(f"Reset prefix cache failed for client {client.host}:{client.server_port}: {e}")
        for client in self.clients:
            t = threading.Thread(target=worker, args=(client,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

def main():
    # Define server configurations.
    server_configs = [
        {"host": "127.0.0.1", "server_port": 8000, "group_port": 51216},
        {"host": "127.0.0.1", "server_port": 8010, "group_port": 51217},
    ]
    
    multi_client = MultiVLLMClient(server_configs)
    
    # Define test prompts.
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity simply."
    ]
    
    # Total completions per prompt.
    n = 7
    
    # Sampling parameters including a base seed.
    sampling_kwargs = {
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 32,
        "guided_decoding_regex": None,
        # "seed": 42  # Base seed provided here.
    }
    
    try:
        completions = multi_client.generate(prompts, n, **sampling_kwargs)
    except Exception as e:
        print("Error during generation:", e)
        return
    
    # Instantiate a tokenizer for decoding.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Decode completions. Each item in 'completions' is a list of token IDs.
    decoded_completions = [tokenizer.decode(comp, skip_special_tokens=True) for comp in completions]
    
    # Print results.
    num_prompts = len(prompts)
    print(f"\nFinal decoded completions (flat list):")
    for i in range(num_prompts):
        print(f"\nPrompt {i} ('{prompts[i]}') completions:")
        start = i * n
        end = (i + 1) * n
        for j, comp in enumerate(decoded_completions[start:end]):
            print(f"  Completion {j}: {comp}")

if __name__ == "__main__":
    main()
