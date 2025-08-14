#!/usr/bin/env python3
"""
Setup script for BitNet model evaluation.
This script modifies the Python path to include BitNet configuration files
without copying anything to the model directory.
"""

import os
import sys
from pathlib import Path

def setup_bitnet_path():
    """Add the current directory to Python path so BitNet modules can be imported"""
    current_dir = Path(__file__).parent.absolute()

    # Add current directory to Python path if not already there
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"Added {current_dir} to Python path for BitNet modules")

    # Verify BitNet files exist
    config_file = current_dir / "configuration_bitnet.py"
    modeling_file = current_dir / "modeling_bitnet.py"

    if not config_file.exists():
        print(f"Error: configuration_bitnet.py not found in {current_dir}")
        return False

    if not modeling_file.exists():
        print(f"Error: modeling_bitnet.py not found in {current_dir}")
        return False

    print("✓ BitNet configuration files found and path configured")
    return True

def create_bitnet_init():
    """Create __init__.py to make the directory a Python package"""
    current_dir = Path(__file__).parent.absolute()
    init_file = current_dir / "__init__.py"

    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('# BitNet evaluation package\n')
        print("Created __init__.py for BitNet package")

def main():
    """Main setup function"""
    print("Setting up BitNet evaluation environment...")

    # Create package structure
    create_bitnet_init()

    # Setup Python path
    success = setup_bitnet_path()

    if success:
        print("\n✓ BitNet setup completed successfully!")
        print("The evaluation pipeline can now find BitNet modules without copying files.")
    else:
        print("\n✗ BitNet setup failed!")
        return False

    return True

if __name__ == "__main__":
    main()
