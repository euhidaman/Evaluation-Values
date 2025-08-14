#!/usr/bin/env python3
"""
Simple patch for cloud environment to handle empty evaluation results.
"""

import os
import sys

def create_cloud_patch():
    """Create a simple patch file for the cloud environment"""

    patch_content = '''
# Quick fix for KeyError: 'UID' issue
# Add this to the evaluation pipeline if needed:

# In process_results function, replace:
# average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())

# With:
if "UID" not in accuracy or not accuracy["UID"]:
    print(f"Warning: No results found for temperature {temp}. Setting accuracy to 0.")
    average_accuracies[temp] = 0.0
else:
    average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())
'''

    with open("evaluation_pipeline_patch.txt", "w") as f:
        f.write(patch_content)

    print("Created evaluation_pipeline_patch.txt with fix instructions")

if __name__ == "__main__":
    create_cloud_patch()
    print("All evaluation data is ready!")
    print("\nNext steps for cloud:")
    print("1. Copy this repository to cloud")
    print("2. Run: cd /workspace/Evaluation-Values")
    print("3. Apply patch if needed (instructions in evaluation_pipeline_patch.txt)")
    print("4. Run full evaluation: bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal")
