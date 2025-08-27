#!/usr/bin/env python3
"""
Test the fixed CoordAtt integration.
"""

import torch
import custom_modules_fixed
from ultralytics import YOLO
from pathlib import Path

def test_fixed_integration():
    """Test the fixed integration."""
    print("🧪 Testing fixed CoordAtt integration...")
    
    # Test standalone CoordAtt
    coord_att = custom_modules_fixed.CoordAtt()
    x = torch.randn(1, 256, 64, 64)
    output = coord_att(x)
    print(f"✅ Simple CoordAtt input: {x.shape}, output: {output.shape}")
    
    # Test enhanced model loading
    print("🔄 Loading enhanced YOLO11 with CoordAtt...")
    try:
        model_path = Path(__file__).parent / 'models' / 'yolo11_surface_defect_p2_coordatt_final.yaml'
        model = YOLO(str(model_path))
        print("✅ Enhanced model loaded successfully!")
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            output = model.model(dummy_input)
            print(f"✅ Forward pass successful! Output: {len(output)} tensors")
            
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_integration()
    if success:
        print("🎉 Fixed integration test passed!")
    else:
        print("❌ Integration test failed")