# Contributing to Keras Benchmarks

Thank you for contributing benchmark runs and improvements to `keras-benchmarks`!

## Submitting Benchmark Results

Follow these steps to clone the repository, run a benchmark on your hardware, and open a pull request with your result file:

1. **Fork and clone the repository:**
   Fork [`keras-team/benchmarks`](https://github.com/keras-team/benchmarks) on GitHub, clone your fork locally, and add `upstream` so you can stay up to date:

   ```bash
   git clone https://github.com/<your-username>/benchmarks.git
   cd benchmarks
   git remote add upstream https://github.com/keras-team/benchmarks.git
   git checkout -b bench/<backend>-<model>
   ```

2. **Set up your Python environment:**
   Create a virtual environment and install dependencies with [`uv`](https://docs.astral.sh/uv/) (Python 3.11+ required):

   ```bash
   uv venv
   uv pip install -e ".[dev]"
   ```

   If your target Keras backend is not included in the `dev` extra (for example, `tensorflow`, `torch`, or `keras-mlx`), install it into the virtual environment as well:

   ```bash
   uv pip install torch
   ```

3. **Run the benchmark:**
   Run the benchmark CLI (use `--source community` or `--source partner` as appropriate). Each config runs in a fresh subprocess and writes a single JSON file under `results/<YYYY-MM>/`:

   ```bash
   uv run --no-sync keras-benchmarks \
     --workload inference \
     --backend jax \
     --model mlp \
     --compile-mode compiled \
     --source community
   ```

4. **Verify your results and checks:**
   Inspect the generated JSON file in `results/<YYYY-MM>/` to confirm hardware metadata and timings look accurate, then run the linter and tests:

   ```bash
   uv run --no-sync ruff check
   uv run --no-sync ruff format --check
   uv run --no-sync pytest
   ```

5. **Commit and open a pull request:**
   Stage only your newly generated result file (or code changes), commit, push to your fork, and open a PR against `keras-team/benchmarks`:

   ```bash
   git add results/
   git commit -m "Add <backend> <model> inference benchmark results"
   git push -u origin bench/<backend>-<model>
   ```

   In your PR description, briefly summarize your hardware (`device_name`, OS), backend version, and any relevant environment details.
