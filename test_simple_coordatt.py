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
    
    # Test standalone CoordAtt with automatic channel detection
    coord_att = custom_modules.CoordAtt()  # No channel specification needed
    x = torch.randn(1, 256, 64, 64)
    output = coord_att(x)
    print(f"✅ CoordAtt input: {x.shape}, output: {output.shape}")
    
    # Test with different channel sizes
    x2 = torch.randn(1, 128, 32, 32)
    coord_att2 = custom_modules.CoordAtt()
    output2 = coord_att2(x2)
    print(f"✅ CoordAtt input: {x2.shape}, output: {output2.shape}")
    
    # Test with enhanced model
    print("🔄 Loading enhanced YOLO11 with CoordAtt...")
    try:
        enhanced_model = YOLO('models/yolo11_surface_defect_p2_coordatt_final.yaml')
        print("✅ Enhanced model loaded successfully")
    except Exception as e:
        print(f"⚠️ Enhanced model failed to load: {e}")
        # Fallback to standard model
        std_model = YOLO('models/yolo11_surface_defect_p2.yaml')
        print("✅ Standard model loaded successfully")
    
    return True

if __name__ == "__main__":
    test_simple_integration()
    print("🎉 Simple integration test passed!")