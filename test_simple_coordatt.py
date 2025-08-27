#!/usr/bin/env python3
"""
Simple test to verify that just the CoordAtt module works correctly.
"""

import torch
import custom_modules
from ultralytics import YOLO

def test_simple_integration():
    """Test minimal CoordAtt integration."""
    print("🧪 Testing simple CoordAtt integration...")
    
    # Test standalone CoordAtt
    coord_att = custom_modules.CoordAtt(256)
    x = torch.randn(1, 256, 64, 64)
    output = coord_att(x)
    print(f"✅ CoordAtt input: {x.shape}, output: {output.shape}")
    
    # Test with standard YOLO model
    print("🔄 Loading standard YOLO11...")
    std_model = YOLO('models/yolo11_surface_defect_p2.yaml')
    print("✅ Standard model loaded successfully")
    
    return True

if __name__ == "__main__":
    test_simple_integration()
    print("🎉 Simple integration test passed!")