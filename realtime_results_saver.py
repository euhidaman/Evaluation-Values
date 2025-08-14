#!/usr/bin/env python3
"""
Real-time results saver for BabyLM 2025 evaluations.
This saves results immediately after each evaluation task completes.
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class EvaluationResultsSaver:
    def __init__(self, results_file: str = "evaluation_results.json"):
        self.results_file = Path(results_file)
        self.results = self._load_existing_results()

    def _load_existing_results(self) -> Dict[str, Any]:
        """Load existing results file or create new structure"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing results file: {e}")
                print("Creating new results structure...")

        # Create new results structure
        return {
            "evaluation_summary": {
                "total_models": 0,
                "models_evaluated": [],
                "tracks": [],
                "last_updated": datetime.now().isoformat(),
                "evaluation_status": "in_progress"
            },
            "model_results": {},
            "evaluation_log": []
        }

    def _save_results(self):
        """Save current results to file"""
        self.results["evaluation_summary"]["last_updated"] = datetime.now().isoformat()

        try:
            # Write to temporary file first, then rename for atomic operation
            temp_file = self.results_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.results_file)
            print(f"✓ Results saved to {self.results_file}")

        except IOError as e:
            print(f"Error saving results: {e}")

    def start_model_evaluation(self, model_name: str, track: str):
        """Initialize model evaluation structure"""
        if model_name not in self.results["model_results"]:
            self.results["model_results"][model_name] = {}
            if model_name not in self.results["evaluation_summary"]["models_evaluated"]:
                self.results["evaluation_summary"]["models_evaluated"].append(model_name)
                self.results["evaluation_summary"]["total_models"] += 1

        if track not in self.results["model_results"][model_name]:
            self.results["model_results"][model_name][track] = {
                "status": "in_progress",
                "start_time": datetime.now().isoformat(),
                "zero_shot": {},
                "finetuning": {},
                "reading": {},
                "metadata": {
                    "track": track,
                    "model": model_name,
                    "backend": None
                }
            }

            if track not in self.results["evaluation_summary"]["tracks"]:
                self.results["evaluation_summary"]["tracks"].append(track)

        # Log the start
        self.results["evaluation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "start_evaluation",
            "model": model_name,
            "track": track
        })

        self._save_results()

    def save_task_result(self, model_name: str, track: str, task_type: str, task_name: str,
                        result_data: Dict[str, Any], output_text: str = ""):
        """Save results for a specific task"""

        # Ensure model/track structure exists
        if (model_name not in self.results["model_results"] or
            track not in self.results["model_results"][model_name]):
            self.start_model_evaluation(model_name, track)

        # Extract scores from output text
        scores = self._extract_scores_from_output(output_text)

        # Combine structured data with extracted scores
        final_result = {
            "completed_at": datetime.now().isoformat(),
            "scores": scores,
            "raw_data": result_data,
            "status": "completed"
        }

        # Save to appropriate section
        if task_type == "zero_shot":
            self.results["model_results"][model_name][track]["zero_shot"][task_name] = final_result
        elif task_type == "finetuning":
            self.results["model_results"][model_name][track]["finetuning"][task_name] = final_result
        elif task_type == "reading":
            self.results["model_results"][model_name][track]["reading"][task_name] = final_result

        # Log the task completion
        self.results["evaluation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "task_completed",
            "model": model_name,
            "track": track,
            "task_type": task_type,
            "task_name": task_name,
            "scores": scores
        })

        self._save_results()
        print(f"✓ Saved {task_type} results for {task_name} ({model_name}/{track})")

    def complete_model_evaluation(self, model_name: str, track: str, backend: str = None):
        """Mark model evaluation as complete"""
        if (model_name in self.results["model_results"] and
            track in self.results["model_results"][model_name]):

            self.results["model_results"][model_name][track]["status"] = "completed"
            self.results["model_results"][model_name][track]["end_time"] = datetime.now().isoformat()

            if backend:
                self.results["model_results"][model_name][track]["metadata"]["backend"] = backend

            # Log completion
            self.results["evaluation_log"].append({
                "timestamp": datetime.now().isoformat(),
                "event": "evaluation_completed",
                "model": model_name,
                "track": track
            })

            self._save_results()
            print(f"✓ Completed evaluation for {model_name} on {track} track")

    def mark_evaluation_complete(self):
        """Mark entire evaluation suite as complete"""
        self.results["evaluation_summary"]["evaluation_status"] = "completed"
        self.results["evaluation_summary"]["completion_time"] = datetime.now().isoformat()

        self.results["evaluation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "all_evaluations_completed"
        })

        self._save_results()
        print("✓ All evaluations completed!")

    def _extract_scores_from_output(self, output_text: str) -> Dict[str, float]:
        """Extract numerical scores from evaluation output"""
        scores = {}

        if not output_text:
            return scores

        # Common score patterns
        patterns = {
            'accuracy': r'(?:accuracy|acc)[:=\s]+([\d.]+)',
            'f1': r'f1[:=\s]+([\d.]+)',
            'score': r'(?:^|\s)score[:=\s]+([\d.]+)',
            'blimp_score': r'blimp.*?score[:=\s]+([\d.]+)',
            'ewok_score': r'ewok.*?score[:=\s]+([\d.]+)',
            'reading_score': r'reading.*?score[:=\s]+([\d.]+)',
            'eye_tracking': r'eye\s+tracking.*?score[:=\s]+([\d.]+)',
            'self_paced': r'self.*?paced.*?score[:=\s]+([\d.]+)',
            'loss': r'loss[:=\s]+([\d.]+)',
            'perplexity': r'perplexity[:=\s]+([\d.]+)',
            'mcc': r'mcc[:=\s]+([\d.]+)',
            'precision': r'precision[:=\s]+([\d.]+)',
            'recall': r'recall[:=\s]+([\d.]+)'
        }

        for metric, pattern in patterns.items():
            matches = re.findall(pattern, output_text.lower(), re.MULTILINE)
            if matches:
                try:
                    # Take the last match (most recent score)
                    scores[metric] = float(matches[-1])
                except ValueError:
                    continue

        return scores

    def get_summary(self) -> Dict[str, Any]:
        """Get current evaluation summary"""
        completed_models = 0
        total_tasks = 0
        completed_tasks = 0

        for model_name, model_data in self.results["model_results"].items():
            model_completed = True
            for track_name, track_data in model_data.items():
                if track_data.get("status") != "completed":
                    model_completed = False

                # Count tasks
                total_tasks += len(track_data.get("zero_shot", {}))
                total_tasks += len(track_data.get("finetuning", {}))
                total_tasks += len(track_data.get("reading", {}))

                # Count completed tasks
                for task_data in track_data.get("zero_shot", {}).values():
                    if task_data.get("status") == "completed":
                        completed_tasks += 1
                for task_data in track_data.get("finetuning", {}).values():
                    if task_data.get("status") == "completed":
                        completed_tasks += 1
                for task_data in track_data.get("reading", {}).values():
                    if task_data.get("status") == "completed":
                        completed_tasks += 1

            if model_completed:
                completed_models += 1

        return {
            "total_models": self.results["evaluation_summary"]["total_models"],
            "completed_models": completed_models,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress_percentage": (completed_tasks / max(total_tasks, 1)) * 100,
            "last_updated": self.results["evaluation_summary"]["last_updated"]
        }

# Global instance
results_saver = EvaluationResultsSaver()

def save_task_result(model_name: str, track: str, task_type: str, task_name: str,
                    result_data: Dict[str, Any] = None, output_text: str = ""):
    """Convenience function to save task results"""
    if result_data is None:
        result_data = {}

    results_saver.save_task_result(model_name, track, task_type, task_name, result_data, output_text)

def start_model_evaluation(model_name: str, track: str):
    """Convenience function to start model evaluation"""
    results_saver.start_model_evaluation(model_name, track)

def complete_model_evaluation(model_name: str, track: str, backend: str = None):
    """Convenience function to complete model evaluation"""
    results_saver.complete_model_evaluation(model_name, track, backend)

def mark_evaluation_complete():
    """Convenience function to mark all evaluations complete"""
    results_saver.mark_evaluation_complete()

def get_summary():
    """Convenience function to get evaluation summary"""
    return results_saver.get_summary()

if __name__ == "__main__":
    # Print current summary
    summary = get_summary()
    print("Evaluation Summary:")
    print(f"Models: {summary['completed_models']}/{summary['total_models']}")
    print(f"Tasks: {summary['completed_tasks']}/{summary['total_tasks']}")
    print(f"Progress: {summary['progress_percentage']:.1f}%")
    print(f"Last updated: {summary['last_updated']}")
