"""
Data loading test - FINAL CORRECT VERSION

API: create_dataset(dataset_type, config, split='train', transform=None)
     where config is a dictionary with dataset parameters

Run from project root:
    python test_data_loading_final.py
"""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("DATA LOADING TEST - Your Actual Tubelets")
print("=" * 70)

# ============================================================================
# TEST DATASET LOADING WITH CORRECT API
# ============================================================================

try:
    from datasets import create_dataset

    # The correct API requires a config dictionary
    print("\n[Step 1] Creating config for dataset...")

    config = {
        'train_csv': 'data/tubelets/train_index.csv',
        'val_csv': 'data/tubelets/val_index.csv',
        'test_csv': 'data/tubelets/test_index.csv',
        # May also need root or other params - we'll see
    }

    print(f"✅ Config created: {config}")

    # Load training dataset
    print("\n[Step 2] Loading training dataset...")
    train_dataset = create_dataset(
        dataset_type='tubelet',
        config=config,
        split='train',
        transform=None
    )

    print(f"✅ Training dataset loaded!")
    print(f"   Total samples: {len(train_dataset)}")

    # Load a sample
    print("\n[Step 3] Loading sample tubelet...")
    sample, label = train_dataset[0]

    print(f"✅ Sample loaded!")
    print(f"   Tubelet shape: {sample.shape}")
    print(f"   Label: {label}")
    print(f"   Data type: {sample.dtype}")
    print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")

    # Check shape format
    if sample.shape == (3, 112, 112, 3):
        print(f"   📝 Shape: (T=3, H=112, W=112, C=3) - Original format")
    elif sample.shape == (3, 3, 112, 112):
        print(f"   📝 Shape: (T=3, C=3, H=112, W=112) - PyTorch format")
    elif len(sample.shape) == 4:
        print(f"   📝 Shape: 4D tensor")

    # Test multiple samples
    print("\n[Step 4] Testing multiple samples...")
    for i in range(min(5, len(train_dataset))):
        s, l = train_dataset[i]
        print(f"   Sample {i}: shape={s.shape}, label={l}, dtype={s.dtype}")

    # Check label distribution
    print("\n[Step 5] Checking label distribution...")
    num_check = min(1000, len(train_dataset))
    labels = [train_dataset[i][1] for i in range(num_check)]
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"   Checked {num_check} samples:")
    print(f"   - Positive (drone): {pos} ({100*pos/len(labels):.1f}%)")
    print(f"   - Negative (no drone): {neg} ({100*neg/len(labels):.1f}%)")

    # Load validation dataset
    print("\n[Step 6] Loading validation dataset...")
    val_dataset = create_dataset(
        dataset_type='tubelet',
        config=config,
        split='val',
        transform=None
    )
    print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")

    # Load test dataset
    print("\n[Step 7] Loading test dataset...")
    test_dataset = create_dataset(
        dataset_type='tubelet',
        config=config,
        split='test',
        transform=None
    )
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Success!
    print("\n" + "=" * 70)
    print("🎉 DATA LOADING TEST PASSED!")
    print("=" * 70)
    print(f"""
Dataset Summary:
✅ Training:   {len(train_dataset):,} samples
✅ Validation: {len(val_dataset):,} samples  
✅ Test:       {len(test_dataset):,} samples

Sample info:
- Shape: {sample.shape}
- Type: {sample.dtype}
- Range: [{sample.min():.1f}, {sample.max():.1f}]

🎯 Your data is fully ready for training!

Next steps:
1. Run safety checker: python safety_checker.py
2. Copy test config: Copy test_experiment.yaml to configs/
3. Run 2-epoch training test

Next command:
    python safety_checker.py
""")

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("\nCheck that these files exist:")
    print("  - data/tubelets/train_index.csv")
    print("  - data/tubelets/val_index.csv")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

    print("\n💡 If you see 'KeyError' or 'missing key' errors,")
    print("   the config dict might need additional keys.")
    print("   Share the error and I'll fix it!")

print("=" * 70)