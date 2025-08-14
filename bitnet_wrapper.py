#!/usr/bin/env python3
"""
BitNet model wrapper that handles loading BitNet models with custom configuration
from the main directory without modifying the model folder.
"""

import os
import sys
import shutil
from pathlib import Path

def setup_bitnet_for_evaluation():
    """
    Setup BitNet model for evaluation by creating symbolic links or temporary copies
    in the model directory that point to the files in the main directory.
    """

    # Get paths
    current_dir = Path(__file__).parent.absolute()
    bitnet_model_dir = current_dir / "models" / "microsoft--bitnet-b1.58-2B-4T"

    # Source files in main directory
    config_source = current_dir / "configuration_bitnet.py"
    modeling_source = current_dir / "modeling_bitnet.py"

    # Target files in model directory
    config_target = bitnet_model_dir / "configuration_bitnet.py"
    modeling_target = bitnet_model_dir / "modeling_bitnet.py"

    # Check if model directory exists
    if not bitnet_model_dir.exists():
        print(f"BitNet model directory not found: {bitnet_model_dir}")
        return False

    # Check if source files exist
    if not config_source.exists() or not modeling_source.exists():
        print("BitNet configuration files not found in main directory")
        return False

    try:
        # Create symbolic links (preferred) or copy files (fallback)
        if os.name == 'nt':  # Windows
            # On Windows, copy files as symlinks may require admin privileges
            if not config_target.exists():
                shutil.copy2(config_source, config_target)
                print(f"Copied {config_source.name} to model directory")

            if not modeling_target.exists():
                shutil.copy2(modeling_source, modeling_target)
                print(f"Copied {modeling_source.name} to model directory")
        else:  # Unix/Linux
            # On Unix systems, create symbolic links
            if not config_target.exists():
                config_target.symlink_to(config_source)
                print(f"Created symlink for {config_source.name}")

            if not modeling_target.exists():
                modeling_target.symlink_to(modeling_source)
                print(f"Created symlink for {modeling_source.name}")

        return True

    except Exception as e:
        print(f"Error setting up BitNet files: {e}")
        return False

def cleanup_bitnet_files():
    """Clean up temporary BitNet files from model directory"""
    current_dir = Path(__file__).parent.absolute()
    bitnet_model_dir = current_dir / "models" / "microsoft--bitnet-b1.58-2B-4T"

    config_target = bitnet_model_dir / "configuration_bitnet.py"
    modeling_target = bitnet_model_dir / "modeling_bitnet.py"

    try:
        if config_target.exists():
            config_target.unlink()
            print("Removed configuration_bitnet.py from model directory")

        if modeling_target.exists():
            modeling_target.unlink()
            print("Removed modeling_bitnet.py from model directory")

    except Exception as e:
        print(f"Warning: Could not clean up BitNet files: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_bitnet_files()
    else:
        setup_bitnet_for_evaluation()
