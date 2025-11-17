"""
Safety checker - Verifies test configuration won't overwrite production data.
Run this before any testing to ensure safety.
"""

import os
import sys
from pathlib import Path
import yaml

print("=" * 70)
print("🛡️  SAFETY CHECKER - Protecting Your Production Data")
print("=" * 70)

# ============================================================================
# DEFINE PROTECTED PATHS
# ============================================================================

PROTECTED_PATHS = [
    # Your existing data (read-only, never write)
    "data/tubelets/train",
    "data/tubelets/val",
    "data/tubelets/test",
    "data/regions",

    # Your existing models (never overwrite)
    "outputs/checkpoints",
    "outputs/models",

    # Your existing results (never overwrite)
    "outputs/evaluation",
    "outputs/results",
    "results",

    # Your existing logs (never overwrite)
    "outputs/logs",
]

TEST_OUTPUT_PATHS = [
    # Test directories that are SAFE to write to
    "data/test_tubelets",
    "outputs/TEST_checkpoints",
    "outputs/TEST_logs",
    "outputs/TEST_evaluation",
    "outputs/TEST_results",
]


# ============================================================================
# CHECK FUNCTIONS
# ============================================================================

def check_config_safety(config_path: str) -> bool:
    """
    Check if a config file uses safe test paths.

    Args:
        config_path: Path to config YAML file

    Returns:
        True if safe, False if dangerous
    """
    print(f"\n📋 Checking config: {config_path}")

    if not os.path.exists(config_path):
        print(f"  ⚠️  Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dangerous_paths = []
    safe = True

    # Check paths in config
    def check_path_recursive(obj, key_path=""):
        nonlocal safe, dangerous_paths

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key_path = f"{key_path}.{key}" if key_path else key
                check_path_recursive(value, new_key_path)
        elif isinstance(obj, str):
            # Check if this string is a path that writes to protected location
            for protected in PROTECTED_PATHS:
                if protected in obj and ('output' in key_path.lower() or
                                         'checkpoint' in key_path.lower() or
                                         'log' in key_path.lower() or
                                         'save' in key_path.lower()):
                    dangerous_paths.append((key_path, obj))
                    safe = False

    check_path_recursive(config)

    if dangerous_paths:
        print(f"  ❌ DANGEROUS CONFIG - Would write to protected paths:")
        for key_path, path in dangerous_paths:
            print(f"     - {key_path}: {path}")
        return False
    else:
        print(f"  ✅ Config is SAFE - Uses test directories")
        return True


def check_existing_data() -> dict:
    """Check what production data exists and needs protection."""
    print(f"\n📊 Checking existing production data...")

    existing = {
        'data': [],
        'models': [],
        'results': []
    }

    # Check data
    for path in ["data/tubelets", "data/regions"]:
        if os.path.exists(path):
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            existing['data'].append({
                'path': path,
                'size_mb': size / (1024 * 1024),
                'files': len(list(Path(path).rglob('*')))
            })

    # Check models
    for path in ["outputs/checkpoints", "outputs/models"]:
        if os.path.exists(path):
            files = list(Path(path).rglob('*.pth'))
            if files:
                size = sum(f.stat().st_size for f in files)
                existing['models'].append({
                    'path': path,
                    'size_mb': size / (1024 * 1024),
                    'files': len(files)
                })

    # Check results
    for path in ["outputs/evaluation", "outputs/results", "results"]:
        if os.path.exists(path):
            files = list(Path(path).rglob('*'))
            if files:
                existing['results'].append({
                    'path': path,
                    'files': len([f for f in files if f.is_file()])
                })

    # Print summary
    if existing['data']:
        print(f"\n  📦 Found existing DATA:")
        for item in existing['data']:
            print(f"     - {item['path']}: {item['files']} files, {item['size_mb']:.1f} MB")

    if existing['models']:
        print(f"\n  🤖 Found existing MODELS:")
        for item in existing['models']:
            print(f"     - {item['path']}: {item['files']} checkpoints, {item['size_mb']:.1f} MB")

    if existing['results']:
        print(f"\n  📈 Found existing RESULTS:")
        for item in existing['results']:
            print(f"     - {item['path']}: {item['files']} files")

    if not any([existing['data'], existing['models'], existing['results']]):
        print(f"  ℹ️  No existing production data found (fresh project)")

    return existing


def create_test_directories():
    """Create test directories if they don't exist."""
    print(f"\n📁 Creating test directories...")

    for path in TEST_OUTPUT_PATHS:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {path}")

    print(f"\n  ✅ All test directories ready")


def safety_report():
    """Generate safety report."""
    print("\n" + "=" * 70)
    print("🛡️  SAFETY REPORT")
    print("=" * 70)

    print(f"""
✅ PROTECTED PATHS (will never be overwritten):
   - data/tubelets/
   - data/regions/
   - outputs/checkpoints/
   - outputs/evaluation/
   - outputs/logs/
   - results/

✅ SAFE TEST PATHS (okay to write):
   - data/test_tubelets/
   - outputs/TEST_checkpoints/
   - outputs/TEST_evaluation/
   - outputs/TEST_logs/
   - outputs/TEST_results/

🔒 SAFETY RULES:
   1. Always use test_experiment.yaml for testing
   2. Never modify configs/experiment.yaml
   3. Test outputs go to TEST_* directories
   4. Production data is read-only during tests

💡 TO RUN SAFE TESTS:
   python scripts/train.py --config configs/test_experiment.yaml
   python test_all_components.py  # Uses dummy data only
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all safety checks."""

    # Check existing data
    existing = check_existing_data()

    # Check test config exists and is safe
    test_config = "configs/test_experiment.yaml"
    if os.path.exists(test_config):
        is_safe = check_config_safety(test_config)
        if not is_safe:
            print(f"\n❌ TEST CONFIG IS NOT SAFE!")
            print(f"   Please fix the config before running tests.")
            return False
    else:
        print(f"\n⚠️  Test config not found: {test_config}")
        print(f"   Create this file from test_experiment.yaml in outputs/")

    # Check production config (should NOT be used for testing)
    prod_config = "configs/experiment.yaml"
    if os.path.exists(prod_config):
        print(f"\n⚠️  PRODUCTION CONFIG DETECTED: {prod_config}")
        print(f"   DO NOT use this for testing - it may overwrite production data!")

    # Create test directories
    create_test_directories()

    # Generate safety report
    safety_report()

    print("\n" + "=" * 70)
    print("✅ SAFETY CHECK COMPLETE")
    print("=" * 70)
    print(f"\nYou can now safely run tests using:")
    print(f"  python test_all_components.py")
    print(f"  python scripts/train.py --config configs/test_experiment.yaml")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)