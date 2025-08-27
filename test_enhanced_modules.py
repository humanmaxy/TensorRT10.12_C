#!/usr/bin/env python3
"""
Test the enhanced modules integration.
"""

import torch
import custom_modules_fixed
import advanced_modules
from ultralytics import YOLO
from pathlib import Path

def test_enhanced_integration():
    """Test the enhanced modules."""
    print("🧪 Testing enhanced modules integration...")
    
    # Test individual modules
    print("\n📋 Testing individual modules:")
    
    # SE Attention (auto-detects channels)
    se_att = advanced_modules.SEAttention()
    x = torch.randn(1, 256, 64, 64)
    output = se_att(x)
    print(f"✅ SEAttention: {x.shape} -> {output.shape}")
    
    # CBAM (auto-detects channels)
    cbam = advanced_modules.CBAM()
    output = cbam(x)
    print(f"✅ CBAM: {x.shape} -> {output.shape}")
    
    # ECA (auto-detects channels)
    eca = advanced_modules.ECA()
    output = eca(x)
    print(f"✅ ECA: {x.shape} -> {output.shape}")
    
    # Test enhanced C3k2
    c3k2_enhanced = advanced_modules.C3k2_Enhanced(256, 512, n=2, attention_type='se')
    output = c3k2_enhanced(x)
    print(f"✅ C3k2_Enhanced: {x.shape} -> {output.shape}")
    
    # Test model loading
    print("\n🔄 Testing enhanced model loading...")
    try:
        # Test backbone enhanced version (more conservative)
        model_path = Path(__file__).parent / 'models' / 'yolo11_surface_defect_backbone_enhanced.yaml'
        model = YOLO(str(model_path))
        print("✅ Backbone enhanced model loaded successfully!")
        
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
    success = test_enhanced_integration()
    if success:
        print("🎉 Enhanced modules integration test passed!")
        print("\n📈 Ready for improved MAP training!")
        print("💡 Try: python train_surface_defect.py")
    else:
        print("❌ Integration test failed")