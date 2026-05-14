#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ApuShape Smoke Test Script

This script performs a basic smoke test of the ApuShape inference functionality.
It runs in CPU mode and tests the segmentation on sample images.

Usage:
    python .github/workflows/smoke_test.py
"""

import os
import sys
import glob
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable CUDA for CPU-only testing
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import cv2
        import numpy as np
        from PyQt5 import QtWidgets
        from backend.ApuFunction import FlexibleUNet_star
        from backend.stardist_pkg import dist_to_coord, non_maximum_suppression, polygons_to_label
        print("  All imports successful")
        return True
    except ImportError as e:
        print(f"  Import failed: {e}")
        return False


def test_model_creation():
    """Test that the model can be created."""
    print("Creating model...")
    try:
        from backend.ApuFunction import FlexibleUNet_star

        model = FlexibleUNet_star(
            in_channels=3,
            out_channels=33,  # 32 rays + 1 prob
            backbone='convnext_small',
            pretrained=False,  # No pretrained for faster testing
            n_rays=32,
            prob_out_channels=1
        )
        model.eval()
        print("  Model created successfully")
        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        return False


def test_model_inference():
    """Test model inference on a dummy image."""
    print("Testing model inference...")
    try:
        import torch
        from backend.ApuFunction import FlexibleUNet_star

        model = FlexibleUNet_star(
            in_channels=3,
            out_channels=33,
            backbone='convnext_small',
            pretrained=False,
            n_rays=32,
            prob_out_channels=1
        )
        model.eval()

        # Create dummy input (batch=1, channels=3, height=512, width=512)
        dummy_input = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            dist_output, prob_output = model(dummy_input)

        # Check output shapes
        assert dist_output.shape == (1, 32, 512, 512), f"Unexpected dist output shape: {dist_output.shape}"
        assert prob_output.shape == (1, 1, 512, 512), f"Unexpected prob output shape: {prob_output.shape}"

        print(f"  Inference successful - dist shape: {dist_output.shape}, prob shape: {prob_output.shape}")
        return True
    except Exception as e:
        print(f"  Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nms():
    """Test Non-Maximum Suppression."""
    print("Testing NMS...")
    try:
        import numpy as np
        from backend.stardist_pkg import non_maximum_suppression

        # Create dummy dist and prob maps
        dist = np.ones((128, 128, 32), dtype=np.float32) * 10
        prob = np.ones((128, 128), dtype=np.float32) * 0.8

        points, probi, disti, nms = non_maximum_suppression(
            dist, prob, prob_thresh=0.5, nms_thresh=0.4
        )

        print(f"  NMS successful - detected {len(points)} points")
        return True
    except Exception as e:
        print(f"  NMS failed: {e}")
        return False


def test_contour_save():
    """Test contour saving functionality."""
    print("Testing contour saving...")
    try:
        import json
        import tempfile

        # Create dummy contour data
        contour = [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]]

        geojson_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": "1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[contour[0] + [contour[0][0]]]]  # Close the polygon
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": "cell"
                }
            }]
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False, encoding='utf-8') as f:
            json.dump(geojson_data, f)
            temp_path = f.name

        # Verify file was created and can be read
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        os.unlink(temp_path)  # Clean up

        print("  Contour save/load successful")
        return True
    except Exception as e:
        print(f"  Contour save failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("ApuShape Smoke Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()

    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Model Inference Test", test_model_inference),
        ("NMS Test", test_nms),
        ("Contour Save Test", test_contour_save),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()