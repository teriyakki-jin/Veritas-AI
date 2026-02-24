from datasets import load_dataset
try:
    print("Inspecting MTEB FEVER dataset...")
    ds = load_dataset("mteb/fever", trust_remote_code=True)
    print("MTEB FEVER loaded!")
    print(ds)
    if 'train' in ds:
        print("Train sample:")
        print(ds['train'][0])
except Exception as e:
    print(f"Failed MTEB: {e}")

try:
    print("\nInspecting KILT FEVER dataset (kilt_tasks)...")
    # kilt_tasks includes 'fever'
    ds = load_dataset("kilt_tasks", "fever", trust_remote_code=True)
    print("KILT FEVER loaded!")
    print(ds)
    if 'train' in ds:
        print("Train sample:")
        print(ds['train'][0])
except Exception as e:
    print(f"Failed KILT: {e}")
