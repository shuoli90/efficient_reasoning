from setuptools import setup, find_packages


setup(
    name='erl',
    version="0.0.1",
    packages=find_packages(include=["erl", "erl.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.46.0",
        "accelerate",
        "datasets",
        "rich",
        "vllm",
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
