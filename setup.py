from setuptools import setup, find_packages


setup(
    name='erl',
    version="0.0.1",
    packages=find_packages(include=["erl", "erl.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "flash-attn>=2.6.3",
        "transformers>=4.46.2",  # fixed the gradient accumulation bug, see: https://huggingface.co/blog/gradient_accumulation
        "datasets==2.21.0",
        "accelerate==0.34.2",
        "peft==0.12.0",
        "trl==0.9.6",
        "deepspeed==0.15.2",
        "vllm==0.6.4",
        "antlr4-python3-runtime==4.11.0",
        "openai",
        "numpy",
        "pandas",
        "python-dotenv",
        "requests",
        "tqdm",
        "joblib",
        "wandb",
        "treelib",
    ],
    extras_require={
        "develop": [
            "black>=24.10.0",
            "flake8>=7.1.1",
            "mypy>=1.12.0",
        ],
    },
)
