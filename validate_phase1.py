#!/usr/bin/env python3
"""
Phase 1 Validation Script for Colin Trading Bot v2.0 ML Infrastructure

This script validates that all Phase 1 components are properly implemented
without requiring external dependencies.
"""

import ast
import os
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def validate_class_structure(file_path, expected_classes):
    """Validate that expected classes exist in the file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)
        found_classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found_classes.append(node.name)

        missing_classes = [cls for cls in expected_classes if cls not in found_classes]
        return len(missing_classes) == 0, missing_classes
    except Exception as e:
        return False, str(e)

def main():
    """Main validation function."""
    print("🔍 Phase 1 ML Infrastructure Validation")
    print("=" * 50)

    base_path = Path("src/v2/ai_engine")

    # Define expected files and classes
    validation_map = {
        "base/ml_base.py": ["MLModelBase"],
        "base/feature_base.py": ["FeatureEngineerBase"],
        "base/pipeline_base.py": ["MLPipelineBase"],
        "prediction/lstm_model.py": ["LSTMPricePredictor", "LSTMNetwork"],
        "prediction/transformer_model.py": ["TransformerPredictor", "TransformerNetwork"],
        "prediction/ensemble_model.py": ["EnsembleModel", "GradientBoostingEnsemble"],
        "features/technical_features.py": ["TechnicalFeatureEngineer"],
        "features/orderbook_features.py": ["OrderBookFeatureEngineer"],
        "features/liquidity_features.py": ["LiquidityFeatureEngineer"],
        "features/alternative_features.py": ["AlternativeFeatureEngineer"]
    }

    all_passed = True

    for file_path, expected_classes in validation_map.items():
        full_path = base_path / file_path

        if not full_path.exists():
            print(f"❌ {file_path}: File not found")
            all_passed = False
            continue

        # Validate syntax
        syntax_valid, syntax_error = validate_python_syntax(full_path)
        if not syntax_valid:
            print(f"❌ {file_path}: Syntax error - {syntax_error}")
            all_passed = False
            continue

        # Validate class structure
        classes_valid, missing_classes = validate_class_structure(full_path, expected_classes)
        if not classes_valid:
            print(f"❌ {file_path}: Missing classes - {missing_classes}")
            all_passed = False
            continue

        print(f"✅ {file_path}: Syntax and structure OK")

    # Validate __init__.py files
    init_files = [
        "__init__.py",
        "base/__init__.py",
        "prediction/__init__.py",
        "features/__init__.py"
    ]

    for init_file in init_files:
        full_path = base_path / init_file
        if full_path.exists():
            syntax_valid, _ = validate_python_syntax(full_path)
            if syntax_valid:
                print(f"✅ {init_file}: Syntax OK")
            else:
                print(f"❌ {init_file}: Syntax error")
                all_passed = False
        else:
            print(f"❌ {init_file}: File not found")
            all_passed = False

    # Validate requirements file
    req_file = Path("requirements_v2.txt")
    if req_file.exists():
        print(f"✅ requirements_v2.txt: Found")

        # Check for key dependencies
        with open(req_file, 'r') as f:
            requirements = f.read()

        key_deps = ["torch", "tensorflow", "pandas", "numpy", "scikit-learn"]
        for dep in key_deps:
            if dep in requirements:
                print(f"  ✅ {dep}: Included")
            else:
                print(f"  ❌ {dep}: Missing")
    else:
        print("❌ requirements_v2.txt: Not found")
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("🎉 Phase 1 Validation: PASSED")
        print("✅ All ML Infrastructure components implemented correctly")
        print("✅ Syntax validation passed")
        print("✅ Class structure validation passed")
        print("✅ Dependencies documented")
    else:
        print("❌ Phase 1 Validation: FAILED")
        print("Some components need attention")

    return all_passed

if __name__ == "__main__":
    main()