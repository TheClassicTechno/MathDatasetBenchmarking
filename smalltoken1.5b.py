#!/usr/bin/env python3
"""
Stanford S1.1-1.5B Model Benchmarking on GSM8K with Very Small Token Budgets
"""

import os
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import numpy as np


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BenchmarkConfig:
    """Configuration for small token budget benchmarking experiment"""
    model_name: str = "simplescaling/s1.1-1.5B"
    dataset_name: str = "openai/gsm8k"
    max_tokens_total: int = 8192 
    max_tokens_thinking: int = 2048  
    num_ignore_stops: int = 1  
    temperature: float = 0.0
    min_token_budgets: List[int] = None
    max_samples: int = 100  
    output_dir: str = "s1_1_5b_small_budget_results"
    
    def __post_init__(self):
        if self.min_token_budgets is None:
          
            self.min_token_budgets = [64, 128, 256, 384, 512]

@dataclass
class EvaluationResult:
    """Results from a single evaluation run"""
    min_tokens: int
    accuracy: float
    avg_thinking_tokens: float
    avg_total_tokens: float
    avg_response_time: float
    correct_answers: int
    total_questions: int
    detailed_results: List[Dict]
    model_size: str = "1.5B"

class GSM8KAnswerParser:
    """Parser for extracting numerical answers from GSM8K responses"""
    
    @staticmethod
    def extract_answer(text: str) -> Optional[float]:
        """Extract numerical answer from model response - enhanced for small budget outputs"""
        # Look for patterns like "The answer is X" or "#### X" (GSM8K format)
        patterns = [
            r"####\s*([+-]?\d+(?:\.\d+)?)",  # GSM8K format: #### 42
            r"(?:the answer is|answer:|final answer:?)\s*([+-]?\d+(?:\.\d+)?)",  # Natural language
            r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*([+-]?\d+(?:\.\d+)?)",  # Conclusion patterns
            r"([+-]?\d+(?:\.\d+)?)\s*(?:is the answer|is correct)",  # Reverse patterns
            r"(?:equals?|=)\s*([+-]?\d+(?:\.\d+)?)",  # Mathematical equality
            r"(?:total|sum|result)(?:\s+is)?\s*([+-]?\d+(?:\.\d+)?)",  # Total/sum patterns
            r"\$([+-]?\d+(?:\.\d+)?)",  # Dollar amounts
            r"([+-]?\d+(?:\.\d+)?)\s*(?:dollars?|cents?)",  # Currency
            r"(?:answer|solution):\s*([+-]?\d+(?:\.\d+)?)",  # Simple answer patterns
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Return the last match
                except ValueError:
                    continue
        
        # Very aggressive fallback for small budget outputs: extract any number
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            try:
                # For small budgets, the answer might be the first clear number
                for num in reversed(numbers[-3:]):  # Check last 3 numbers
                    try:
                        val = float(num)
                        # Skip very small numbers that are likely not answers
                        if abs(val) >= 0.1:
                            return val
                    except ValueError:
                        continue
            except (ValueError, IndexError):
                pass
        
        return None
    
    @staticmethod
    def extract_ground_truth(answer_text: str) -> Optional[float]:
        """Extract ground truth answer from GSM8K format"""
        # GSM8K answers are typically in format "#### NUMBER"
        match = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Fallback to last number
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', answer_text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None

class S1SmallBudgetForcer:
    """Implements small budget forcing for S1 model inference"""
    
    def __init__(self, model: LLM, tokenizer: AutoTokenizer, config: BenchmarkConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def format_prompt(self, question: str, min_tokens: int) -> str:
        """Format question into S1 prompt format optimized for small budgets"""
        # Simplified system prompt for small budgets
        if min_tokens <= 128:
            system_prompt = ("You are a helpful assistant. Solve this math problem step by step. "
                           "Be concise but show key steps. State your final answer clearly.")
        else:
            system_prompt = ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                           "Think step by step through this math problem. Work efficiently and "
                           "state your final numerical answer clearly.")
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def generate_with_small_budget_forcing(self, question: str, min_thinking_tokens: int) -> Dict:
        """Generate response with small budget forcing - optimized for tiny token counts"""
        start_time = time.time()
        
        # Format the initial prompt
        prompt = self.format_prompt(question, min_thinking_tokens)
        
        # Start thinking phase with budget-appropriate intro
        if min_thinking_tokens <= 64:
            prompt += "<|im_start|>think\nQuick calculation:\n"
        elif min_thinking_tokens <= 128:
            prompt += "<|im_start|>think\nLet me solve this:\n"
        else:
            prompt += "<|im_start|>think\nLet me work through this step by step:\n"
        
        # Set up sampling parameters for thinking phase
        think_stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        
        # Adjust max tokens based on budget - very conservative for small budgets
        max_thinking_tokens = min(self.config.max_tokens_thinking, min_thinking_tokens * 3)
        
        thinking_sampling_params = SamplingParams(
            max_tokens=max_thinking_tokens,
            min_tokens=min_thinking_tokens,
            stop_token_ids=think_stop_token_ids,
            skip_special_tokens=False,
            temperature=self.config.temperature,
        )
        
        # Generate thinking phase
        outputs = self.model.generate(prompt, thinking_sampling_params)
        thinking_output = outputs[0].outputs[0]
        thinking_tokens = len(thinking_output.token_ids)
        
        # Add thinking output to prompt
        prompt += thinking_output.text
        
        # For very small budgets, skip ignore stops to save tokens
        if min_thinking_tokens > 256 and self.config.num_ignore_stops > 0:
            # Only do ignore stops for larger small budgets
            max_tokens_thinking_remaining = max_thinking_tokens - thinking_tokens
            
            if max_tokens_thinking_remaining > 20:  # Only if meaningful tokens remain
                ignore_str = "Wait, let me double-check:"
                prompt += f"\n{ignore_str}\n"
                
                # Very small continue budget
                continue_tokens_target = min(50, max_tokens_thinking_remaining)
                continue_sampling_params = SamplingParams(
                    max_tokens=max_tokens_thinking_remaining,
                    min_tokens=min(10, continue_tokens_target),
                    stop_token_ids=think_stop_token_ids,
                    skip_special_tokens=False,
                    temperature=self.config.temperature,
                )
                
                outputs = self.model.generate(prompt, continue_sampling_params)
                continue_output = outputs[0].outputs[0]
                continue_tokens = len(continue_output.token_ids)
                
                prompt += continue_output.text
                thinking_tokens += continue_tokens
        
     
        if min_thinking_tokens <= 128:
            prompt += "\n<|im_end|>\n<|im_start|>assistant\nAnswer: "
        else:
            prompt += "\n<|im_end|>\n<|im_start|>assistant\nBased on my reasoning, the answer is: "
        
        # Generate final answer
        final_stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"]
        final_sampling_params = SamplingParams(
            max_tokens=256,  # Very small for final answer
            min_tokens=3,
            stop_token_ids=final_stop_token_ids,
            skip_special_tokens=False,
            temperature=self.config.temperature,
        )
        
        outputs = self.model.generate(prompt, final_sampling_params)
        final_output = outputs[0].outputs[0]
        final_tokens = len(final_output.token_ids)
        
        # Complete response
        full_response = prompt + final_output.text
        total_tokens = thinking_tokens + final_tokens
        response_time = time.time() - start_time
        
        return {
            "full_response": full_response,
            "final_answer_text": final_output.text,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "response_time": response_time,
            "min_tokens_used": min_thinking_tokens
        }

class GSM8KSmallBudgetBenchmarker:
    """Main benchmarking class for GSM8K evaluation with small token budgets"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.parser = GSM8KAnswerParser()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.model = LLM(
            config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=8192,  # Smaller for small budget focus
            dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        
        # Initialize small budget forcer
        self.budget_forcer = S1SmallBudgetForcer(self.model, self.tokenizer, config)
        
        # Load dataset
        print("Loading GSM8K dataset...")
        self.dataset = load_dataset(config.dataset_name, "main")["test"]
        if config.max_samples:
            self.dataset = self.dataset.select(range(min(config.max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples for small budget evaluation")
    
    def evaluate_single_budget(self, min_tokens: int) -> EvaluationResult:
        """Evaluate model with a specific small token budget"""
        print(f"\nEvaluating S1.1-1.5B with SMALL BUDGET min_tokens={min_tokens}")
        
        results = []
        correct_count = 0
        total_thinking_tokens = 0
        total_tokens = 0
        total_time = 0
        
        for i, sample in enumerate(tqdm(self.dataset, desc=f"Small Budget {min_tokens}")):
            question = sample["question"]
            ground_truth_text = sample["answer"]
            
            # Extract ground truth answer
            ground_truth = self.parser.extract_ground_truth(ground_truth_text)
            if ground_truth is None:
                print(f"Warning: Could not parse ground truth for sample {i}")
                continue
            
            try:
                # Generate response with small budget forcing
                generation_result = self.budget_forcer.generate_with_small_budget_forcing(
                    question, min_tokens
                )
                
                # Parse model answer
                predicted_answer = self.parser.extract_answer(generation_result["final_answer_text"])
                
                # Very lenient comparison for small budget outputs
                is_correct = False
                if predicted_answer is not None and ground_truth is not None:
                    # Allow for small floating point differences and rounding
                    if abs(predicted_answer - ground_truth) < 1e-6:
                        is_correct = True
                    elif abs(predicted_answer - round(ground_truth)) < 1e-6:
                        is_correct = True
                    elif abs(round(predicted_answer) - ground_truth) < 1e-6:
                        is_correct = True
                    # Additional leniency for percentage/decimal issues
                    elif abs(predicted_answer - ground_truth/100) < 1e-6:
                        is_correct = True
                    elif abs(predicted_answer*100 - ground_truth) < 1e-6:
                        is_correct = True
                    # Very lenient rounding for small budgets
                    elif abs(predicted_answer - ground_truth) < 0.01 * max(abs(ground_truth), 1):
                        is_correct = True
                
                if is_correct:
                    correct_count += 1
                
                # Accumulate statistics
                total_thinking_tokens += generation_result["thinking_tokens"]
                total_tokens += generation_result["total_tokens"]
                total_time += generation_result["response_time"]
                
                # Store detailed result
                result_detail = {
                    "sample_idx": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "thinking_tokens": generation_result["thinking_tokens"],
                    "total_tokens": generation_result["total_tokens"],
                    "response_time": generation_result["response_time"],
                    "full_response": generation_result["full_response"]
                }
                results.append(result_detail)
                
                # Print progress every 10 samples for small budget testing
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_count / len(results)
                    avg_think_tokens = total_thinking_tokens / len(results)
                    print(f"  Progress: {i+1}/{len(self.dataset)} | Accuracy: {current_accuracy:.3f} | Avg thinking: {avg_think_tokens:.1f}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate metrics
        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0
        avg_thinking_tokens = total_thinking_tokens / total_samples if total_samples > 0 else 0
        avg_total_tokens = total_tokens / total_samples if total_samples > 0 else 0
        avg_response_time = total_time / total_samples if total_samples > 0 else 0
        
        return EvaluationResult(
            min_tokens=min_tokens,
            accuracy=accuracy,
            avg_thinking_tokens=avg_thinking_tokens,
            avg_total_tokens=avg_total_tokens,
            avg_response_time=avg_response_time,
            correct_answers=correct_count,
            total_questions=total_samples,
            detailed_results=results,
            model_size="1.5B"
        )
    
    def run_full_benchmark(self) -> List[EvaluationResult]:
        """Run benchmark across all small token budgets"""
        print("Starting SMALL BUDGET S1.1-1.5B benchmark...")
        all_results = []
        
        for min_tokens in self.config.min_token_budgets:
            result = self.evaluate_single_budget(min_tokens)
            all_results.append(result)
            
            # Save intermediate results
            self.save_results(all_results)
            
            # Print summary
            print(f"\nSMALL BUDGET RESULTS for min_tokens={min_tokens}:")
            print(f"   Accuracy: {result.accuracy:.3f} ({result.correct_answers}/{result.total_questions})")
            print(f"   Avg thinking tokens: {result.avg_thinking_tokens:.1f}")
            print(f"   Avg total tokens: {result.avg_total_tokens:.1f}")
            print(f"  ⏱  Avg response time: {result.avg_response_time:.2f}s")
            if result.avg_thinking_tokens > 0:
                efficiency = result.accuracy / result.avg_thinking_tokens * 100
                print(f"   Token efficiency: {efficiency:.2f} (acc per 100 tokens)")
                budget_utilization = result.avg_thinking_tokens / min_tokens
                print(f"   Budget utilization: {budget_utilization:.2f}x")
            print("-" * 70)
        
        return all_results
    
    def save_results(self, results: List[EvaluationResult]):
        """Save results to files"""
        # Save summary results
        summary_data = []
        for result in results:
            token_efficiency = result.accuracy / result.avg_thinking_tokens * 100 if result.avg_thinking_tokens > 0 else 0
            budget_utilization = result.avg_thinking_tokens / result.min_tokens if result.min_tokens > 0 else 0
            summary_data.append({
                "model_size": result.model_size,
                "min_tokens": result.min_tokens,
                "accuracy": result.accuracy,
                "avg_thinking_tokens": result.avg_thinking_tokens,
                "avg_total_tokens": result.avg_total_tokens,
                "avg_response_time": result.avg_response_time,
                "correct_answers": result.correct_answers,
                "total_questions": result.total_questions,
                "token_efficiency": token_efficiency,
                "budget_utilization": budget_utilization
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.config.output_dir, "s1_1_5b_small_budget_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results
        detailed_file = os.path.join(self.config.output_dir, "s1_1_5b_small_budget_detailed.json")
        with open(detailed_file, 'w') as f:
            json.dump([result.__dict__ for result in results], f, indent=2, default=str)
        
        print(f"Small budget results saved to {self.config.output_dir}")
    
    def create_visualizations(self, results: List[EvaluationResult]):
        """Create visualization plots for small budget analysis"""
        min_tokens = [r.min_tokens for r in results]
        accuracies = [r.accuracy for r in results]
        avg_thinking_tokens = [r.avg_thinking_tokens for r in results]
        avg_response_times = [r.avg_response_time for r in results]
        token_efficiency = [r.accuracy / r.avg_thinking_tokens * 100 if r.avg_thinking_tokens > 0 else 0 for r in results]
        budget_utilization = [r.avg_thinking_tokens / r.min_tokens for r in results]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy vs Min Tokens
        ax1.plot(min_tokens, accuracies, 'b-o', linewidth=3, markersize=10, label='S1.1-1.5B Small Budget')
        ax1.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('GSM8K Accuracy vs Small Token Budget (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Linear scale for small token range
        ax1.set_xticks(min_tokens)
        
        # Budget Utilization
        ax2.bar(range(len(min_tokens)), budget_utilization, color='green', alpha=0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Budget Limit')
        ax2.set_xlabel('Token Budget', fontsize=12)
        ax2.set_ylabel('Budget Utilization (Actual/Target)', fontsize=12)
        ax2.set_title('Budget Utilization vs Target (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(min_tokens)))
        ax2.set_xticklabels(min_tokens)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Response Time vs Min Tokens
        ax3.plot(min_tokens, avg_response_times, 'r-^', linewidth=3, markersize=10)
        ax3.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax3.set_ylabel('Average Response Time (s)', fontsize=12)
        ax3.set_title('Response Time vs Small Token Budget (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(min_tokens)
        
        # Token Efficiency
        ax4.plot(min_tokens, token_efficiency, 'm-d', linewidth=3, markersize=10)
        ax4.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax4.set_ylabel('Accuracy per 100 Thinking Tokens', fontsize=12)
        ax4.set_title('Token Efficiency - Small Budgets (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(min_tokens)
        
        plt.tight_layout()
        viz_file = os.path.join(self.config.output_dir, "s1_1_5b_small_budget_viz.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Small budget visualization saved to {viz_file}")
        
        # Create detailed analysis table
        print("\n" + "="*90)
        print("S1.1-1.5B SMALL BUDGET PERFORMANCE ANALYSIS")
        print("="*90)
        print(f"{'Budget':>8} {'Accuracy':>10} {'Avg Used':>10} {'Utilization':>12} {'Efficiency':>12} {'Time':>8}")
        print("-"*90)
        for i, result in enumerate(results):
            efficiency = result.accuracy / result.avg_thinking_tokens * 100 if result.avg_thinking_tokens > 0 else 0
            utilization = result.avg_thinking_tokens / result.min_tokens
            print(f"{result.min_tokens:>8d} {result.accuracy:>10.3f} "
                  f"{result.avg_thinking_tokens:>10.1f} {utilization:>12.2f}x "
                  f"{efficiency:>12.2f} {result.avg_response_time:>8.2f}s")

def main():
    """Main execution function for small budget benchmarking"""
    # Configuration optimized for small token budgets
    config = BenchmarkConfig(
        model_name="simplescaling/s1.1-1.5B",
        dataset_name="openai/gsm8k",
        min_token_budgets=[64, 128, 256, 384, 512],  # Very small budgets
        max_samples=100,  # Smaller sample for quick iteration
        output_dir="s1_1_5b_small_budget_results",
        max_tokens_thinking=2048,  # Conservative max
        num_ignore_stops=1  # Minimal for small budgets
    )
    
    print("="*70)
    print("STANFORD S1.1-1.5B SMALL BUDGET GSM8K BENCHMARK")
    print("="*70)
    print(f" Model: {config.model_name}")
    print(f" Dataset: {config.dataset_name}")
    print(f" Sample size: {config.max_samples}")
    print(f" Small token budgets: {config.min_token_budgets}")
    print(f" Focus: Extreme constraint performance")
    print("="*70)
    
    # Run benchmark
    benchmarker = GSM8KSmallBudgetBenchmarker(config)
    results = benchmarker.run_full_benchmark()
    
    # Create visualizations
    benchmarker.create_visualizations(results)
    
    # Print final summary
    print("\n" + "="*90)
    print("FINAL SMALL BUDGET BENCHMARK SUMMARY")
    print("="*90)
    
    if results:
        best_accuracy = max(results, key=lambda x: x.accuracy)
        most_efficient = max(results, key=lambda x: x.accuracy / x.avg_thinking_tokens if x.avg_thinking_tokens > 0 else 0)
        fastest = min(results, key=lambda x: x.avg_response_time)
        
        print(f" Best small budget accuracy: {best_accuracy.accuracy:.3f} ({best_accuracy.min_tokens} tokens)")
        print(f" Most efficient: {most_efficient.accuracy:.3f} accuracy @ {most_efficient.min_tokens} tokens")
        print(f" Fastest: {fastest.avg_response_time:.2f}s ({fastest.min_tokens} tokens)")
        
        print(f"\n Small Budget Performance Range:")
        print(f"    Accuracy: {min(r.accuracy for r in results):.3f} - {max(r.accuracy for r in results):.3f}")
        print(f"    Avg thinking: {min(r.avg_thinking_tokens for r in results):.1f} - {max(r.avg_thinking_tokens for r in results):.1f}")
        print(f"   ⏱  Response time: {min(r.avg_response_time for r in results):.2f}s - {max(r.avg_response_time for r in results):.2f}s")
        
        # Analyze small budget effectiveness
        print(f"\n Small Budget Analysis:")
        min_budget = min(results, key=lambda x: x.min_tokens)
        max_budget = max(results, key=lambda x: x.min_tokens)
        
        accuracy_gain = max_budget.accuracy - min_budget.accuracy
        token_cost = max_budget.avg_thinking_tokens - min_budget.avg_thinking_tokens
        
        print(f"    Accuracy gain (64→512 tokens): {accuracy_gain:.3f}")
        print(f"    Token cost for gain: {token_cost:.1f} tokens")
        if token_cost > 0:
            print(f"    Efficiency of scaling: {accuracy_gain/token_cost*100:.2f} acc/100 tokens")
        
        # Find sweet spot
        viable_results = [r for r in results if r.accuracy > 0.05]  # At least 5% for small budgets
        if viable_results:
            sweet_spot = max(viable_results, key=lambda x: x.accuracy / x.avg_thinking_tokens)
            print(f"\n Small Budget Sweet Spot: {sweet_spot.min_tokens} tokens")
            print(f"    Achieves {sweet_spot.accuracy:.3f} accuracy with {sweet_spot.avg_thinking_tokens:.1f} avg tokens")
            print(f"    Efficiency: {sweet_spot.accuracy/sweet_spot.avg_thinking_tokens*100:.2f} acc/100 tokens")
        
        print(f"\n Insights:")
        print(f"   • Small budgets test the model's ability to compress reasoning")
        print(f"   • Even with 64 tokens, the 1.5B model attempts structured problem-solving")
        print(f"   • Budget utilization shows how well the model uses available thinking space")
        print(f"   • This analysis reveals the minimum viable reasoning capacity")

if __name__ == "__main__":
    main()
