import threading
from typing import List, Dict
import torch
from efficient_reasoning.extras.vllm_client import VLLMClient
from transformers import AutoTokenizer
from tqdm import tqdm 

class MultiVLLMClient:
    def __init__(self, server_configs: List[Dict]):
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
                connection_timeout=config.get("connection_timeout", 30.0)
            )
            self.clients.append(client)
    
    def generate(self, prompts: List[str], n: int, **sampling_kwargs) -> List[int]:
        """
        Partition prompts among clients and call generate concurrently.
        The final results will be a flat list where the completions for prompt i occupy the slice
        [i * n, (i + 1) * n].

        Args:
            prompts: List of prompts.
            n: Number of completions per prompt.
            sampling_kwargs: Other keyword arguments for sampling.

        Returns:
            A flat list of completions with total length = len(prompts) * n.
        """
        num_clients = len(self.clients)
        # Partition the prompts among the available clients using round-robin assignment.
        partitioned_prompts = [[] for _ in range(num_clients)]
        original_idx = [[] for _ in range(num_clients)]
        for idx, prompt in enumerate(prompts):
            client_idx = idx % num_clients  # assign prompt to client in round-robin fashion
            partitioned_prompts[client_idx].append(prompt)
            original_idx[client_idx].append(idx)
        # print(partitioned_prompts)
        # breakpoint()
        # Preallocate a results list with a length equal to number of prompts * completions per prompt.
        flat_results = [None] * (len(prompts) * n)
        
        def worker(client, sub_prompts, indices):
            try:
                # Call the client's generate API. It returns a flat list of completions where each prompt
                # in the sublist has exactly n completions (i.e. client_result[i*n : (i+1)*n] are the completions
                # for sub_prompts[i]).
                # print("Generating for client:", client.host, client.server_port)
                # print("Sub-prompts:", sub_prompts)
                # print("Indices:", indices)
                client_result = client.generate(prompts=sub_prompts, n=n, **sampling_kwargs)
            except Exception as e:
                client_result = None
                print(f"Generation failed for client {client.host}:{client.server_port}: {e}")
                return  # early return if generation fails
            
            # Place the completions into the correct location in flat_results.
            # For each prompt originally assigned to this client:
            for i, orig_index in enumerate(indices):
                # The expected slice in flat_results for prompt orig_index:
                start = orig_index * n
                end = (orig_index + 1) * n
                # Extract the corresponding completions from client_result.
                # Here, client_result is organized so that for prompt i in the sublist,
                # its completions are in the slice [i*n, (i+1)*n]
                flat_results[start:end] = client_result[i * n:(i + 1) * n]
        
        # Launch threads: one per client to work on its partition of prompts.
        threads = []
        for client, sub_prompts, indices in zip(self.clients, partitioned_prompts, original_idx):
            if sub_prompts:  # only create a thread if there is at least one prompt for this client
                t = threading.Thread(target=worker, args=(client, sub_prompts, indices))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        return flat_results

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Update a specific named parameter across all vLLM servers concurrently.
        
        Args:
            name: The parameter name.
            weights: The updated tensor.
        """
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
        """
        Update all model parameters across all vLLM servers concurrently.
        
        This method iterates over all named parameters in the model and dispatches an update call
        to each client.
        
        Args:
            model: The model whose parameters will be updated.
        """
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
        """
        Resets the prefix cache on all vLLM servers concurrently.
        """
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

# Example usage
def main():
    # Define server configurations.
    server_configs = [
        {"host": "127.0.0.1", "server_port": 8000, "group_port": 51216},
        {"host": "127.0.0.1", "server_port": 8010, "group_port": 51217},
    ]
    
    multi_client = MultiVLLMClient(server_configs)
    
    # Define test prompts.
    # prompts = [
    #     "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
    #     "If $5x - 3 = 12$, what is the value of $5x + 3$?"
    # ]

    data = []
    with open("/home/leoh/efficient_reasoning/data/MATH-500/train.jsonl") as f:
        for line in f:
            tmp = eval(line)
            data.append(tmp)

    prompts_list = []
    for index, dict in enumerate(data):
        prompts_list.append(dict["problem"])

    prompts_list= prompts_list[:100]

    print(f"Length of prompts: {len(prompts_list)}")
        
    # Total completions per prompt.
    n = 8
    
    # Sampling parameters including a base seed.
    sampling_kwargs = {
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 2048,
        "guided_decoding_regex": None,
        # "seed": 42  # Base seed provided here.
    }
    
    for i in tqdm(range(0, len(prompts_list), 2), desc="Generating completions"):
        prompts = prompts_list[i:i+2]
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
        breakpoint()


if __name__ == "__main__":
    main()
