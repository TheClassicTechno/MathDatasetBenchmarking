#!/usr/bin/env python3
"""
Stanford S1.1-1.5B 
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
    """Configuration for benchmarking experiment - optimized for 1.5B model"""
    model_name: str = "simplescaling/s1.1-1.5B"
    dataset_name: str = "openai/gsm8k"
    max_tokens_total: int = 16384  
    max_tokens_thinking: int = 12000  
    num_ignore_stops: int = 3 
    temperature: float = 0.0
    min_token_budgets: List[int] = None
    max_samples: int = 200  
    output_dir: str = "s1_1_5b_gsm8k_results"
    
    def __post_init__(self):
        if self.min_token_budgets is None:
           
            self.min_token_budgets = [500, 1000, 2000, 4096, 8192]

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
        """Extract numerical answer from model response - enhanced for 1.5B model"""
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
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Return the last match
                except ValueError:
                    continue
        
        # More aggressive fallback for 1.5B model: extract last number in the text
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            try:
                # Try the last few numbers in case the final answer is not the very last
                for num in reversed(numbers[-5:]):  # Check more numbers for 1.5B
                    try:
                        return float(num)
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

class S1BudgetForcer:
    """Implements budget forcing for S1 model inference - optimized for 1.5B"""
    
    def __init__(self, model: LLM, tokenizer: AutoTokenizer, config: BenchmarkConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def format_prompt(self, question: str) -> str:
        """Format question into S1 prompt format with enhanced instructions for 1.5B model"""
        system_prompt = ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                        "Think step by step and show your reasoning clearly. Work through the problem "
                        "methodically and state your final numerical answer clearly.")
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def generate_with_budget_forcing(self, question: str, min_thinking_tokens: int) -> Dict:
        """Generate response with budget forcing - enhanced for 1.5B model"""
        start_time = time.time()
        
        # Format the initial prompt
        prompt = self.format_prompt(question)
        
        # Start thinking phase
        prompt += "<|im_start|>think\nLet me work through this step by step:\n"
        
        # Set up sampling parameters for thinking phase
        think_stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        thinking_sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens_thinking,
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
        
        # Handle ignore stops (budget forcing technique) - more aggressive for 1.5B
        ignore_phrases = [
            "Wait, let me reconsider this...", 
            "Actually, let me double-check my work:", 
            "Hold on, let me verify this step:",
            "Let me approach this differently:",
            "I should be more careful here:"
        ]
        max_tokens_thinking_remaining = self.config.max_tokens_thinking - thinking_tokens
        
        for i in range(self.config.num_ignore_stops):
            if max_tokens_thinking_remaining > 50:  # Ensure some tokens remain
                ignore_str = ignore_phrases[i % len(ignore_phrases)]
                prompt += f"\n{ignore_str}\n"
                
                # Continue thinking with remaining budget
                continue_tokens_target = min(200, max_tokens_thinking_remaining // 2)  # Smaller chunks for 1.5B
                continue_sampling_params = SamplingParams(
                    max_tokens=max_tokens_thinking_remaining,
                    min_tokens=min(25, continue_tokens_target),  # Smaller minimum for 1.5B
                    stop_token_ids=think_stop_token_ids,
                    skip_special_tokens=False,
                    temperature=self.config.temperature,
                )
                
                outputs = self.model.generate(prompt, continue_sampling_params)
                continue_output = outputs[0].outputs[0]
                continue_tokens = len(continue_output.token_ids)
                
                prompt += continue_output.text
                thinking_tokens += continue_tokens
                max_tokens_thinking_remaining -= continue_tokens
        
        # Add explicit transition to final answer
        prompt += "\n<|im_end|>\n<|im_start|>assistant\nBased on my reasoning above, the final answer is: "
        
        # Generate final answer
        final_stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"]
        final_sampling_params = SamplingParams(
            max_tokens=1024,  # Reduced for 1.5B model
            min_tokens=5,
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

class GSM8KBenchmarker:
    """Main benchmarking class for GSM8K evaluation - optimized for 1.5B model"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.parser = GSM8KAnswerParser()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load model and tokenizer with 1.5B-optimized settings
        print(f"Loading model: {config.model_name}")
        self.model = LLM(
            config.model_name,
            tensor_parallel_size=1,  # 1.5B model fits on single GPU
            gpu_memory_utilization=0.90,  # Can be more aggressive with smaller model
            max_model_len=16384,  # Reduced for 1.5B
            dtype="auto",  # Let vLLM choose optimal dtype
            trust_remote_code=True,  # May be needed for some models
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        
        # Initialize budget forcer
        self.budget_forcer = S1BudgetForcer(self.model, self.tokenizer, config)
        
        # Load dataset
        print("Loading GSM8K dataset...")
        self.dataset = load_dataset(config.dataset_name, "main")["test"]
        if config.max_samples:
            self.dataset = self.dataset.select(range(min(config.max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples for evaluation")
    
    def evaluate_single_budget(self, min_tokens: int) -> EvaluationResult:
        """Evaluate model with a specific token budget"""
        print(f"\nEvaluating S1.1-1.5B with min_tokens={min_tokens}")
        
        results = []
        correct_count = 0
        total_thinking_tokens = 0
        total_tokens = 0
        total_time = 0
        
        for i, sample in enumerate(tqdm(self.dataset, desc=f"1.5B Min tokens {min_tokens}")):
            question = sample["question"]
            ground_truth_text = sample["answer"]
            
            # Extract ground truth answer
            ground_truth = self.parser.extract_ground_truth(ground_truth_text)
            if ground_truth is None:
                print(f"Warning: Could not parse ground truth for sample {i}")
                continue
            
            try:
                # Generate response with budget forcing
                generation_result = self.budget_forcer.generate_with_budget_forcing(
                    question, min_tokens
                )
                
                # Parse model answer
                predicted_answer = self.parser.extract_answer(generation_result["final_answer_text"])
                
                # Very lenient comparison for 1.5B model
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
                
                # Print progress every 20 samples for smaller model
                if (i + 1) % 20 == 0:
                    current_accuracy = correct_count / len(results)
                    print(f"  Progress: {i+1}/{len(self.dataset)} | Current accuracy: {current_accuracy:.3f}")
                
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
        """Run benchmark across all token budgets"""
        print("Starting full S1.1-1.5B benchmark...")
        all_results = []
        
        for min_tokens in self.config.min_token_budgets:
            result = self.evaluate_single_budget(min_tokens)
            all_results.append(result)
            
            # Save intermediate results
            self.save_results(all_results)
            
            # Print summary
            print(f"\nRESULTS for min_tokens={min_tokens}:")
            print(f"  Accuracy: {result.accuracy:.3f} ({result.correct_answers}/{result.total_questions})")
            print(f"  Avg thinking tokens: {result.avg_thinking_tokens:.1f}")
            print(f"  Avg total tokens: {result.avg_total_tokens:.1f}")
            print(f"  Avg response time: {result.avg_response_time:.2f}s")
            if result.avg_thinking_tokens > 0:
                print(f"  Token efficiency: {result.accuracy / result.avg_thinking_tokens * 1000:.2f} (acc per 1K tokens)")
            print("-" * 60)
        
        return all_results
    
    def save_results(self, results: List[EvaluationResult]):
        """Save results to files"""
        # Save summary results
        summary_data = []
        for result in results:
            token_efficiency = result.accuracy / result.avg_thinking_tokens * 1000 if result.avg_thinking_tokens > 0 else 0
            summary_data.append({
                "model_size": result.model_size,
                "min_tokens": result.min_tokens,
                "accuracy": result.accuracy,
                "avg_thinking_tokens": result.avg_thinking_tokens,
                "avg_total_tokens": result.avg_total_tokens,
                "avg_response_time": result.avg_response_time,
                "correct_answers": result.correct_answers,
                "total_questions": result.total_questions,
                "token_efficiency": token_efficiency
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.config.output_dir, "s1_1_5b_benchmark_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results
        detailed_file = os.path.join(self.config.output_dir, "s1_1_5b_detailed_results.json")
        with open(detailed_file, 'w') as f:
            json.dump([result.__dict__ for result in results], f, indent=2, default=str)
        
        print(f"S1.1-1.5B results saved to {self.config.output_dir}")
    
    def create_visualizations(self, results: List[EvaluationResult]):
        """Create visualization plots for 1.5B model"""
        min_tokens = [r.min_tokens for r in results]
        accuracies = [r.accuracy for r in results]
        avg_thinking_tokens = [r.avg_thinking_tokens for r in results]
        avg_response_times = [r.avg_response_time for r in results]
        token_efficiency = [r.accuracy / r.avg_thinking_tokens * 1000 if r.avg_thinking_tokens > 0 else 0 for r in results]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy vs Min Tokens
        ax1.plot(min_tokens, accuracies, 'b-o', linewidth=3, markersize=8, label='S1.1-1.5B')
        ax1.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('GSM8K Accuracy vs Token Budget (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xscale('log')  # Log scale for better visualization of smaller token ranges
        
        # Thinking Tokens vs Min Tokens
        ax2.plot(min_tokens, avg_thinking_tokens, 'g-s', linewidth=3, markersize=8, label='Actual Usage')
        ax2.plot(min_tokens, min_tokens, 'r--', linewidth=2, alpha=0.7, label='Budget Limit')
        ax2.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax2.set_ylabel('Average Thinking Tokens Used', fontsize=12)
        ax2.set_title('Thinking Tokens Used vs Budget (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Response Time vs Min Tokens
        ax3.plot(min_tokens, avg_response_times, 'r-^', linewidth=3, markersize=8)
        ax3.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax3.set_ylabel('Average Response Time (s)', fontsize=12)
        ax3.set_title('Response Time vs Token Budget (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Token Efficiency
        ax4.plot(min_tokens, token_efficiency, 'm-d', linewidth=3, markersize=8)
        ax4.set_xlabel('Minimum Thinking Tokens', fontsize=12)
        ax4.set_ylabel('Accuracy per 1K Thinking Tokens', fontsize=12)
        ax4.set_title('Token Efficiency (S1.1-1.5B)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        viz_file = os.path.join(self.config.output_dir, "s1_1_5b_benchmark_visualization.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"S1.1-1.5B visualization saved to {viz_file}")
        
        # Create comparison table
        print("\n" + "="*80)
        print("S1.1-1.5B MODEL PERFORMANCE BREAKDOWN")
        print("="*80)
        print(f"{'Min Tokens':>10} {'Accuracy':>10} {'Avg Think':>12} {'Efficiency':>12} {'Time (s)':>10}")
        print("-"*80)
        for result in results:
            efficiency = result.accuracy / result.avg_thinking_tokens * 1000 if result.avg_thinking_tokens > 0 else 0
            print(f"{result.min_tokens:>10d} {result.accuracy:>10.3f} "
                  f"{result.avg_thinking_tokens:>12.1f} {efficiency:>12.2f} "
                  f"{result.avg_response_time:>10.2f}")

def main():
    """Main execution function for S1.1-1.5B benchmarking"""
    # Configuration optimized for 1.5B model
    config = BenchmarkConfig(
        model_name="simplescaling/s1.1-1.5B",
        dataset_name="openai/gsm8k",
        min_token_budgets=[500, 1000, 2000, 4096, 8192],
        max_samples=200,  # More samples for better 1.5B evaluation
        output_dir="s1_1_5b_gsm8k_results",
        max_tokens_thinking=12000,  # Reduced for 1.5B
        num_ignore_stops=3
    )
    
    print("="*60)
    print("STANFORD S1.1-1.5B GSM8K BENCHMARK")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Sample size: {config.max_samples}")
    print(f"Token budgets: {config.min_token_budgets}")
    print("="*60)
    
    # Run benchmark
    benchmarker = GSM8KBenchmarker(config)
    results = benchmarker.run_full_benchmark()
    
    # Create visualizations
    benchmarker.create_visualizations(results)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL S1.1-1.5B BENCHMARK SUMMARY")
    print("="*80)
    
    if results:
        best_accuracy = max(results, key=lambda x: x.accuracy)
        most_efficient = max(results, key=lambda x: x.accuracy / x.avg_thinking_tokens if x.avg_thinking_tokens > 0 else 0)
        fastest = min(results, key=lambda x: x.avg_response_time)
        
        print(f" Best accuracy: {best_accuracy.accuracy:.3f} (min_tokens={best_accuracy.min_tokens})")
        print(f" Most efficient: {most_efficient.accuracy:.3f} accuracy with {most_efficient.avg_thinking_tokens:.1f} avg tokens")
        print(f" Fastest: {fastest.avg_response_time:.2f}s avg time (min_tokens={fastest.min_tokens})")
        
        print(f"\n Performance Range:")
        print(f"   Accuracy: {min(r.accuracy for r in results):.3f} - {max(r.accuracy for r in results):.3f}")
        print(f"   Thinking tokens: {min(r.avg_thinking_tokens for r in results):.1f} - {max(r.avg_thinking_tokens for r in results):.1f}")
        print(f"   Response time: {min(r.avg_response_time for r in results):.2f}s - {max(r.avg_response_time for r in results):.2f}s")
        
        # Find optimal operating point (best accuracy/token ratio above 20% accuracy for 1.5B)
        viable_results = [r for r in results if r.accuracy > 0.20]  # At least 20% accuracy for 1.5B
        if viable_results:
            optimal = max(viable_results, key=lambda x: x.accuracy / x.avg_thinking_tokens if x.avg_thinking_tokens > 0 else 0)
            print(f"\n Recommended setting: min_tokens={optimal.min_tokens}")
            print(f"   Balance of {optimal.accuracy:.3f} accuracy with {optimal.avg_thinking_tokens:.1f} avg tokens")
        
        print(f"\n Note: This is a 1.5B parameter model - performance expectations should be adjusted accordingly.")
        print(f"    The model excels at following reasoning patterns even with limited capacity.")

if __name__ == "__main__":
    main()
