# PPE

## Installation

Create and Activate a Conda Environment.

``` bash
conda create -n erl python=3.12
conda activate erl
```

Install from Source with all Dependencies.

``` bash
git clone https://github.com/shuoli90/efficient_reasoning.git
cd erl
make develop
```

## Usage

TBD

## Development

We use the `Makefile` as a command registry:

- `make format`: autoformat  this library with `black`
- `make lint`: perform static analysis of this library with `black` and `flake8`
- `make type`: run type checking using `mypy`
- `make test`: run automated tests (currently, there are no tests)
- `make check`: check assets for packaging

Make sure that `make lint`, `make type`, and `make test` pass before committing changes.

Make sure that `make check` passes before releasing a new version.
