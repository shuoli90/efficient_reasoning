from setuptools import setup, find_packages


setup(
    name='efficient_reasoning',
    version="0.0.1",
    packages=find_packages(include=["efficient_reasoning", "efficient_reasoning.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "flash-attn",
        "transformers>=4.46.2",  # fixed the gradient accumulation bug, see: https://huggingface.co/blog/gradient_accumulation
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "deepspeed",
        "vllm==0.8.1",
        "antlr4-python3-runtime",
        "numpy",
        "requests",
        "tqdm",
        "wandb",
        "latex2sympy2",
        "math_verify",
    ],
    extras_require={
        "develop": [
            "black>=24.10.0",
            "flake8>=7.1.1",
            "mypy>=1.12.0",
        ],
    },
)
