"""Generate and convert Qwen3-VL from HuggingFace to MaxText Orbax format.

This script handles the complete workflow for both 2B and 8B versions:
1. Downloads/loads Qwen3-VL-{size}-Instruct from HuggingFace
2. Saves it locally for reference
3. Converts it to MaxText Orbax checkpoint format

Usage:
    python3 generate_hf_qwen3_vl_checkpoint.py [--size {2b|8b}] [--hf-dir HF_DIR] [--orbax-dir ORBAX_DIR]

Arguments:
    --size: Model size - 2b or 8b (default: 2b)
    --hf-dir: Directory to save HF checkpoint (default: tests/assets/qwen3_vl_{size}_hf)
    --orbax-dir: Directory to save Orbax checkpoint (default: tests/assets/qwen3_vl_{size}_orbax)

Examples:
    python3 generate_hf_qwen3_vl_checkpoint.py
    python3 generate_hf_qwen3_vl_checkpoint.py --size 8b
    python3 generate_hf_qwen3_vl_checkpoint.py --size 2b --orbax-dir /tmp/qwen3_vl_checkpoint
"""
import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText


def get_workspace_root() -> Path:
    """Get the MaxText workspace root directory."""
    current_file = Path(__file__).resolve()
    tools_dir = current_file.parent.parent
    workspace_root = tools_dir.parent
    return workspace_root


def download_and_save_hf_checkpoint(model_size: str, hf_dir: str) -> None:
    """Download Qwen3-VL and save to local directory.
    
    Args:
        model_size: Model size ('2b' or '8b').
        hf_dir: Directory to save the HuggingFace checkpoint.
    """
    # Map size to model ID
    size_upper = model_size.upper()
    model_id = f"Qwen/Qwen3-VL-{size_upper}-Instruct"
    
    print(f"Step 1/2: Downloading {model_id} from HuggingFace...")
    print(f"  Model: {model_id}")
    print(f"  Save directory: {hf_dir}\n")
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        os.makedirs(hf_dir, exist_ok=True)
        model.save_pretrained(hf_dir)
        print(f"✓ HuggingFace checkpoint saved to: {hf_dir}\n")
    except Exception as e:
        print(f"✗ Error downloading/saving HuggingFace checkpoint:")
        print(f"  {e}\n")
        raise


def convert_to_orbax(model_size: str, orbax_dir: str) -> None:
    """Convert HuggingFace checkpoint to MaxText Orbax format.
    
    Args:
        model_size: Model size ('2b' or '8b').
        orbax_dir: Directory to save the Orbax checkpoint.
    """
    workspace_root = get_workspace_root()
    os.chdir(workspace_root)
    
    size_upper = model_size.upper()
    model_id = f"Qwen/Qwen3-VL-{size_upper}-Instruct"
    model_name = f"qwen3-vl-{model_size}"
    
    print(f"Step 2/2: Converting to MaxText Orbax checkpoint format...")
    print(f"  Model: {model_id}")
    print(f"  Output directory: {orbax_dir}\n")
    
    # Convert orbax_dir to absolute path if relative
    if not os.path.isabs(orbax_dir):
        orbax_dir = os.path.join(workspace_root, orbax_dir)
    
    conversion_cmd = [
        "python3",
        "src/maxtext/checkpoint_conversion/to_maxtext.py",
        f"--hf_model_path={model_id}",
        "src/maxtext/configs/post_train/sft.yml",
        f"model_name={model_name}",
        f"base_output_directory={orbax_dir}",
        "run_name=",
        "packing=False",
        "enable_checkpointing=False",
        "scan_layers=False",
        "hardware=cpu",
        "skip_jax_distributed_system=True",
    ]
    
    try:
        result = subprocess.run(conversion_cmd, cwd=workspace_root, check=True)
        print(f"\n✓ Conversion completed successfully!")
        print(f"✓ MaxText Orbax checkpoint saved to: {orbax_dir}\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during conversion:")
        print(f"  Command: {' '.join(conversion_cmd)}")
        print(f"  Exit code: {e.returncode}\n")
        raise


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and convert Qwen3-VL to MaxText Orbax format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_hf_qwen3_vl_checkpoint.py
  python3 generate_hf_qwen3_vl_checkpoint.py --size 8b
  python3 generate_hf_qwen3_vl_checkpoint.py --size 2b --orbax-dir /tmp/qwen3_vl_checkpoint
        """
    )
    
    parser.add_argument(
        "--size",
        choices=["2b", "8b"],
        default="2b",
        help="Model size: 2b or 8b (default: 2b)",
    )
    parser.add_argument(
        "--hf-dir",
        default=None,
        help="Directory to save HuggingFace checkpoint (default: tests/assets/qwen3_vl_{size}_hf)",
    )
    parser.add_argument(
        "--orbax-dir",
        default=None,
        help="Directory to save Orbax checkpoint (default: tests/assets/qwen3_vl_{size}_orbax)",
    )
    
    args = parser.parse_args()
    
    # Set default directories based on model size if not provided
    hf_dir = args.hf_dir or f"tests/assets/qwen3_vl_{args.size}_hf"
    orbax_dir = args.orbax_dir or f"tests/assets/qwen3_vl_{args.size}_orbax"
    
    print("=" * 80)
    print(f"Qwen3-VL-{args.size.upper()} HuggingFace to MaxText Orbax Conversion")
    print("=" * 80)
    print()
    
    try:
        # Step 1: Download and save HF checkpoint
        download_and_save_hf_checkpoint(args.size, hf_dir)
        
        # Step 2: Convert to Orbax format
        convert_to_orbax(args.size, orbax_dir)
        
        print("=" * 80)
        print("All steps completed successfully! ✓")
        print("=" * 80)
        
    except Exception as e:
        print("=" * 80)
        print(f"Conversion failed! ✗")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
