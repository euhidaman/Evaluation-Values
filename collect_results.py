#!/usr/bin/env python3
"""
Aggregate and save evaluation results in JSON format for BabyLM 2025
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import re

def extract_scores_from_output(output_text):
    """Extract numerical scores from evaluation output"""
    scores = {}

    # Common score patterns
    patterns = {
        'accuracy': r'(?:accuracy|acc)[:=]\s*([\d.]+)',
        'f1': r'f1[:=]\s*([\d.]+)',
        'score': r'score[:=]\s*([\d.]+)',
        'blimp_score': r'blimp.*?score[:=]\s*([\d.]+)',
        'ewok_score': r'ewok.*?score[:=]\s*([\d.]+)',
        'reading_score': r'reading.*?score[:=]\s*([\d.]+)',
        'eye_tracking': r'eye tracking.*?score[:=]\s*([\d.]+)',
        'self_paced': r'self.*?paced.*?score[:=]\s*([\d.]+)',
    }

    for metric, pattern in patterns.items():
        matches = re.findall(pattern, output_text.lower())
        if matches:
            try:
                scores[metric] = float(matches[-1])  # Take the last match
            except ValueError:
                continue

    return scores

def collect_evaluation_results():
    """Collect all evaluation results from the results directory"""

    results_dir = Path("./results")
    if not results_dir.exists():
        print("No results directory found")
        return {}

    all_results = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        all_results[model_name] = {}

        for track_dir in model_dir.iterdir():
            if not track_dir.is_dir():
                continue

            track_name = track_dir.name
            all_results[model_name][track_name] = {
                'zero_shot': {},
                'finetuning': {},
                'reading': {},
                'metadata': {
                    'evaluation_date': datetime.now().isoformat(),
                    'track': track_name,
                    'model': model_name
                }
            }

            # Look for result files
            for result_file in track_dir.glob("*"):
                if result_file.is_file():
                    try:
                        if result_file.suffix == '.json':
                            with open(result_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                task_name = result_file.stem
                                all_results[model_name][track_name]['zero_shot'][task_name] = data
                        elif result_file.suffix in ['.txt', '.log']:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                scores = extract_scores_from_output(content)
                                if scores:
                                    task_name = result_file.stem
                                    all_results[model_name][track_name]['zero_shot'][task_name] = scores
                    except Exception as e:
                        print(f"Error reading {result_file}: {e}")
                        continue

    return all_results

def create_results_summary(results):
    """Create a summary of all results"""

    summary = {
        'evaluation_summary': {
            'total_models': len(results),
            'models_evaluated': list(results.keys()),
            'evaluation_date': datetime.now().isoformat(),
            'tracks': set()
        },
        'model_results': results,
        'score_summary': {}
    }

    # Collect all tracks
    for model_data in results.values():
        summary['evaluation_summary']['tracks'].update(model_data.keys())

    summary['evaluation_summary']['tracks'] = list(summary['evaluation_summary']['tracks'])

    # Create score summary
    for model_name, model_data in results.items():
        summary['score_summary'][model_name] = {}

        for track_name, track_data in model_data.items():
            summary['score_summary'][model_name][track_name] = {
                'zero_shot_tasks': len(track_data.get('zero_shot', {})),
                'finetuning_tasks': len(track_data.get('finetuning', {})),
                'has_reading_results': bool(track_data.get('reading', {}))
            }

    return summary

def main():
    """Main function to collect and save results"""

    print("BabyLM 2025 Results Aggregator")
    print("=" * 40)

    # Collect results
    print("Collecting evaluation results...")
    results = collect_evaluation_results()

    if not results:
        print("No evaluation results found.")
        print("Make sure you have run the evaluation scripts first.")
        return

    # Create summary
    print("Creating results summary...")
    summary = create_results_summary(results)

    # Save results
    results_file = Path("evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Results saved to: {results_file}")

    # Print summary
    print(f"\nResults Summary:")
    print(f"- Models evaluated: {summary['evaluation_summary']['total_models']}")
    print(f"- Models: {', '.join(summary['evaluation_summary']['models_evaluated'])}")
    print(f"- Tracks: {', '.join(summary['evaluation_summary']['tracks'])}")

    for model_name, model_summary in summary['score_summary'].items():
        print(f"\n{model_name}:")
        for track_name, track_summary in model_summary.items():
            print(f"  {track_name}: {track_summary['zero_shot_tasks']} zero-shot tasks, "
                  f"{track_summary['finetuning_tasks']} finetuning tasks")

    print(f"\nDetailed results are available in: {results_file}")

if __name__ == "__main__":
    main()
