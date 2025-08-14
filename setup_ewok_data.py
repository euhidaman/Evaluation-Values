#!/usr/bin/env python3
"""
Fixed EWoK data download script with proper encoding handling for Windows.
"""

import json
import os
import sys
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

def setup_ewok_data():
    """Download and set up EWoK data with proper encoding"""

    print("Setting up EWoK evaluation data...")

    # Create directories
    ewok_full_dir = Path("../evaluation_data/full_eval/ewok_filtered")
    ewok_fast_dir = Path("../evaluation_data/fast_eval/ewok_fast")

    ewok_full_dir.mkdir(parents=True, exist_ok=True)
    ewok_fast_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load EWoK dataset from Hugging Face
        print("Downloading EWoK-core 1.0 dataset...")
        dataset = load_dataset("ewok-core/ewok-core-1.0", split="test", trust_remote_code=True)

        print(f"Loaded {len(dataset)} examples from EWoK dataset")

        # Group by domain
        items_per_domain = defaultdict(list)

        for example in dataset:
            domain = example.get("Domain", "unknown")
            items_per_domain[domain].append(example)

        print(f"Found {len(items_per_domain)} domains: {list(items_per_domain.keys())}")

        # Save data for each domain
        total_examples = 0
        for domain, items in items_per_domain.items():
            # Save to full_eval
            full_file = ewok_full_dir / f"{domain}.jsonl"
            with open(full_file, 'w', encoding='utf-8') as outfile:
                for item in items:
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_examples += 1

            # Also save a subset to fast_eval (first 20% of items)
            fast_items = items[:max(1, len(items) // 5)]
            fast_file = ewok_fast_dir / f"{domain}.jsonl"
            with open(fast_file, 'w', encoding='utf-8') as outfile:
                for item in fast_items:
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Successfully saved {total_examples} examples to EWoK data files")
        return True

    except Exception as e:
        print(f"Error setting up EWoK data: {e}")
        return False

def create_minimal_ewok_fallback(ewok_full_dir, ewok_fast_dir):
    """Create minimal EWoK test data as fallback"""

    # Minimal EWoK-style test data
    test_domains = {
        "animals": [
            {
                "Context1": "The cat sat on the mat",
                "Context2": "The dog ran in the park",
                "Target1": "cat",
                "Target2": "dog",
                "Domain": "animals",
                "uid": "ewok_animals_001"
            }
        ],
        "food": [
            {
                "Context1": "She ate an apple for lunch",
                "Context2": "He drank some water",
                "Target1": "apple",
                "Target2": "water",
                "Domain": "food",
                "uid": "ewok_food_001"
            }
        ]
    }

    for domain, examples in test_domains.items():
        # Save to both directories
        for directory in [ewok_full_dir, ewok_fast_dir]:
            domain_file = directory / f"{domain}.jsonl"
            with open(domain_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            print(f"✓ Created minimal {domain} data: {domain_file}")

def verify_ewok_setup():
    """Verify that EWoK data is properly set up"""

    ewok_full_dir = Path("../evaluation_data/full_eval/ewok_filtered")
    ewok_fast_dir = Path("../evaluation_data/fast_eval/ewok_fast")

    print("\nVerifying EWoK data setup:")

    for name, directory in [("Full", ewok_full_dir), ("Fast", ewok_fast_dir)]:
        if directory.exists():
            files = list(directory.glob("*.jsonl"))
            if files:
                print(f"✓ {name} EWoK data: {len(files)} domain files")
                for file in files:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        print(f"  - {file.name}: {lines} examples")
            else:
                print(f"✗ {name} EWoK data: No .jsonl files found")
        else:
            print(f"✗ {name} EWoK directory not found")

def main():
    print("EWoK Data Setup for BabyLM 2025")
    print("=" * 40)

    # Set up EWoK data
    success = setup_ewok_data()

    if success:
        # Verify setup
        verify_ewok_setup()

        print("\n✓ EWoK data setup completed!")
        print("\nTo run the full evaluation with EWoK data:")
        print("bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal")
        print("\nNote: Use 'strict' track to access the full EWoK data from full_eval directory")

    else:
        print("\n✗ EWoK data setup failed!")

if __name__ == "__main__":
    main()
