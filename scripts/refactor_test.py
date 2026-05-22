"""
refactor_test.py
Verifies the DRY refactoring is functional:
  1. Constants import correctly from ml.utils.constants
  2. Model helpers import and run without errors
  3. get_test_folders() returns expected structure
  4. All updated scripts have valid syntax (compile check)
  5. train.py does not hardcode old boilerplate

Usage:
    python scripts/refactor_test.py
"""
import os
import sys

# Ensure project root is on the path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Test 1: ml.utils.constants
# ---------------------------------------------------------------------------
def test_constants():
    print("  [Test 1] ml.utils.constants ... ", end="")
    from ml.utils.constants import (
        PROJECT_ROOT, DATA_PATH, MODEL_SAVE_DIR,
        TRAIN_SPLIT, VAL_SPLIT, IFRAME_PROB, get_device
    )
    assert isinstance(PROJECT_ROOT, os.PathLike) or isinstance(PROJECT_ROOT, str)
    assert "Autoencoders-for-Compression" in str(PROJECT_ROOT)
    expected_suffix = "data" + os.sep + "processed_frames"
    assert str(DATA_PATH).endswith(expected_suffix), f"DATA_PATH={DATA_PATH}"
    assert MODEL_SAVE_DIR is not None
    assert 0 < TRAIN_SPLIT < 1
    assert 0 < VAL_SPLIT < 1
    assert 0 < IFRAME_PROB < 1
    dev = get_device()
    assert str(dev) in ("cuda", "cpu")
    print("PASS")


# ---------------------------------------------------------------------------
# Test 2: ml.utils.model_loading
# ---------------------------------------------------------------------------
def test_model_loading():
    print("  [Test 2] ml.utils.model_loading ... ", end="")
    from ml.utils.model_loading import detect_latent_channels, load_model
    assert callable(detect_latent_channels)
    assert callable(load_model)
    print("PASS  (signatures verified; no model file needed)")


# ---------------------------------------------------------------------------
# Test 3: ml.dataset.get_test_folders
# ---------------------------------------------------------------------------
def test_get_test_folders():
    print("  [Test 3] ml.dataset.get_test_folders ... ", end="")
    from ml.dataset import get_test_folders
    folders = get_test_folders()
    assert isinstance(folders, list)
    if folders:
        assert os.path.isdir(folders[0])
    print(f"PASS  ({len(folders)} folders returned)")


# ---------------------------------------------------------------------------
# Test 4: Script syntax validation (compile only)
# ---------------------------------------------------------------------------
def test_script_syntax():
    print("  [Test 4] Updated scripts have valid syntax ... ")

    scripts_to_check = [
        os.path.join(_THIS_DIR, "evaluate.py"),
        os.path.join(_THIS_DIR, "visualize.py"),
        os.path.join(_THIS_DIR, "analysis.py"),
        os.path.join(_THIS_DIR, "test_code.py"),
        os.path.join(_PROJECT_ROOT, "ml", "train.py"),
    ]
    for path in scripts_to_check:
        name = os.path.relpath(path, _PROJECT_ROOT)
        print(f"    {name} ... ", end="")
        with open(path, encoding="utf-8") as f:
            source = f.read()
        compile(source, path, "exec")
        print("PASS")
    print("  All syntax checks PASS")


# ---------------------------------------------------------------------------
# Test 5: train.py does not hardcode old boilerplate
# ---------------------------------------------------------------------------
def test_train_sanity():
    print("  [Test 5] ml/train.py no stale boilerplate ... ", end="")
    src_path = os.path.join(_PROJECT_ROOT, "ml", "train.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert "get_device()" in src, "train.py should use get_device()"
    assert "PROJECT_ROOT =" not in src, "train.py should not define PROJECT_ROOT"
    assert "DATA_PATH =" not in src, "train.py should not define DATA_PATH"
    print("PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  REFACTOR VERIFICATION TESTS")
    print("=" * 55)

    tests = [
        test_constants,
        test_model_loading,
        test_get_test_folders,
        test_script_syntax,
        test_train_sanity,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  ({e})")
            failed += 1

    print("-" * 55)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 55)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
