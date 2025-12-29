# MathDatasetBenchmarking

part of my work for CS224R Reinforcement Learning research project.
An experimental benchmark for evaluating large language models (LLMs) on mathematical reasoning datasets (GSM8K) using reinforcement learning (RL) driven configurations. This project specializes in systematically testing models such as Stanford's S1.1-1.5B and S1.1-3B on arithmetic and word problem benchmarks, focusing on token efficiency and answer extraction performance.

## Features

- **Reinforcement Learning Task:** Emphasizes benchmarking LLMs (1.5B & 3B variant models) on the openai/gsm8k math dataset under various token constraints, simulating RL or reasoning-budgeted setups.
- **Flexible Benchmark Configurations:** Scripts allow adjustments for model size, token limits, temperature, and sample counts.
- **Automated Parsing & Evaluation:** Sophisticated answer extraction patterns handle numerical formats typical in GSM8K and similar benchmarks.
- **Detailed Logging & Visualization:** Results are systematically saved for further analysis, and code supports plotting via matplotlib/seaborn.

## Main Scripts

- [`script1.5b.py`](https://github.com/TheClassicTechno/MathDatasetBenchmarking/blob/main/script1.5b.py): 
  Benchmarks the "simplescaling/s1.1-1.5B" model on the GSM8K dataset. Tests under several token budgets, parses model answers, and computes accuracy, token usage, and timing statistics.
- [`script3b.py`](https://github.com/TheClassicTechno/MathDatasetBenchmarking/blob/main/script3b.py): 
  Similar benchmarking pipeline for the "simplescaling/s1.1-3B" model with larger budgets and adjusted evaluation parameters.
- [`smalltoken1.5b.py`](https://github.com/TheClassicTechno/MathDatasetBenchmarking/blob/main/smalltoken1.5b.py): 
  Specialized for "simplescaling/s1.1-1.5B" under very small token budgets, to explore minimal information reasoning.

## Usage

1. **Install dependencies:**  
   Most scripts require Python with packages: `transformers`, `vllm`, `datasets`, `pandas`, `numpy`, `tqdm`, `matplotlib`, and `seaborn`.
   ```bash
   pip install transformers vllm datasets pandas numpy tqdm matplotlib seaborn
   ```

2. **Run a Benchmark Script:**  
   Each script can be directly executed. Adjust parameters inside the script (e.g., model, `max_samples`, token budgets, etc.) as needed.
   ```bash
   python script1.5b.py
   python script3b.py
   python smalltoken1.5b.py
   ```

3. **Analyze Results:**  
   Output directories specified in each script contain experiment results (e.g., accuracy, tokens used, timings) for further analysis and visualization.

## Project Structure

```
script1.5b.py         # Benchmark 1.5B model (standard token budgets)
script3b.py           # Benchmark 3B model (larger token budgets)
smalltoken1.5b.py     # 1.5B model with small/restricted token budgets
README.md             # This file
...


---

*For detailed code and result exploration, please visit the [GitHub repository](https://github.com/TheClassicTechno/MathDatasetBenchmarking).*
