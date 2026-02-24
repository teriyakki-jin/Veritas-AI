"""
Model file management script.

Usage:
    python download_models.py --check
    python download_models.py --manifest
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MODEL_DIRS = {
    "liar": "models/liar_baseline",
    "fever": "models/fever_baseline",
    "fnn": "models/fakenewsnet_baseline",
}

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_models(args):
    """Check if model files exist."""
    print("Checking model files...")
    all_ok = True
    
    for name, dirt in MODEL_DIRS.items():
        print(f"\n[{name}] Checking {dirt}...")
        
        if not os.path.exists(dirt):
            print(f"  FAILED: Directory not found: {dirt}")
            all_ok = False
            continue
            
        # Check for model weights (support both formats)
        has_weights = False
        if os.path.exists(os.path.join(dirt, "model.safetensors")):
            print("  OK: model.safetensors found")
            has_weights = True
        elif os.path.exists(os.path.join(dirt, "pytorch_model.bin")):
            print("  OK: pytorch_model.bin found")
            has_weights = True
        else:
            print("  FAILED: No model weights found (model.safetensors or pytorch_model.bin)")
            all_ok = False
            
        # Check config
        if os.path.exists(os.path.join(dirt, "config.json")):
             print("  OK: config.json found")
        else:
             print("  FAILED: config.json missing")
             all_ok = False

    print("\n" + ("="*30))
    if all_ok:
        print("SUCCESS: All required model files are present.")
    else:
        print("ERROR: Some model files are missing.")
        # Don't exit(1) if just checking, allow user to resolve
        sys.exit(1)

def generate_manifest(args):
    """Generate SHA256 manifest of model files."""
    print("Generating model manifest...")
    manifest = {}
    
    for name, dirt in MODEL_DIRS.items():
        if not os.path.exists(dirt):
            print(f"Skipping {name} (directory not found)")
            continue
        
        print(f"Processing {name}...")
        files = []
        for root, _, filenames in os.walk(dirt):
            for fname in filenames:
                fpath = os.path.join(root, fname)
                # Skip manifest itself if generated inside
                if fname == "manifest.json":
                    continue
                    
                size = os.path.getsize(fpath)
                sha = calculate_sha256(fpath)
                
                rel_path = os.path.relpath(fpath, os.getcwd())
                files.append({
                    "path": rel_path,
                    "size": size,
                    "sha256": sha
                })
        manifest[name] = files
        
    out_path = "models/manifest.json"
    os.makedirs("models", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Model management script")
    parser.add_argument("--check", action="store_true", help="Check if model files exist")
    parser.add_argument("--manifest", action="store_true", help="Generate SHA256 manifest")
    
    args = parser.parse_args()
    
    if args.check:
        check_models(args)
    elif args.manifest:
        generate_manifest(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
