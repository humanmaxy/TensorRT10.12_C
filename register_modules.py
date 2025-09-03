"""
Module Registration for Ultralytics YOLO
确保自定义模块正确注册到ultralytics框架
"""

def register_all_modules():
    """注册所有自定义模块到ultralytics"""
    
    print("🔧 Starting module registration...")
    
    # 导入所有模块
    from snake_bifpn_modules import (
        SnakeDeformableConv, C3k2_SnakeDeformable,
        BiFPNLayer, BiFPNBlock, TripleBiFPN, MultiScaleBiFPN,
        MicroDefectAttention, EnhancedDetectHead, FastNormalizedFusion
    )
    
    from advanced_modules import (
        SEAttention, CBAM, ECA, SPP_Enhanced, FPN_Enhanced,
        RepVGGBlock, GhostConv_Enhanced, C3k2_Enhanced, ASFF
    )
    
    # 导入简单模块
    from simple_modules import C3k2
    
    # 尝试多种注册方式
    registration_success = False
    
    # 方法1: 注册到 ultralytics.nn.tasks
    try:
        import ultralytics.nn.tasks as tasks
        
        # Simple modules
        tasks.C3k2 = C3k2
        
        # Snake Deformable Conv modules
        tasks.SnakeDeformableConv = SnakeDeformableConv
        tasks.C3k2_SnakeDeformable = C3k2_SnakeDeformable
        
        # BiFPN modules
        tasks.BiFPNLayer = BiFPNLayer
        tasks.BiFPNBlock = BiFPNBlock
        tasks.TripleBiFPN = TripleBiFPN
        tasks.MultiScaleBiFPN = MultiScaleBiFPN
        tasks.FastNormalizedFusion = FastNormalizedFusion
        
        # Micro defect detection modules
        tasks.MicroDefectAttention = MicroDefectAttention
        tasks.EnhancedDetectHead = EnhancedDetectHead
        
        # Advanced attention modules
        tasks.SEAttention = SEAttention
        tasks.CBAM = CBAM
        tasks.ECA = ECA
        tasks.SPP_Enhanced = SPP_Enhanced
        tasks.FPN_Enhanced = FPN_Enhanced
        tasks.RepVGGBlock = RepVGGBlock
        tasks.GhostConv_Enhanced = GhostConv_Enhanced
        tasks.C3k2_Enhanced = C3k2_Enhanced
        tasks.ASFF = ASFF
        
        print("✅ Registered to ultralytics.nn.tasks")
        registration_success = True
        
    except Exception as e:
        print(f"⚠️ Failed to register to tasks: {e}")
    
    # 方法2: 注册到 ultralytics.nn.modules
    try:
        import ultralytics.nn.modules as modules
        
        # 添加所有模块到modules命名空间
        module_dict = {
            'C3k2': C3k2,
            'SnakeDeformableConv': SnakeDeformableConv,
            'C3k2_SnakeDeformable': C3k2_SnakeDeformable,
            'BiFPNLayer': BiFPNLayer,
            'BiFPNBlock': BiFPNBlock,
            'TripleBiFPN': TripleBiFPN,
            'MultiScaleBiFPN': MultiScaleBiFPN,
            'FastNormalizedFusion': FastNormalizedFusion,
            'MicroDefectAttention': MicroDefectAttention,
            'EnhancedDetectHead': EnhancedDetectHead,
            'SEAttention': SEAttention,
            'CBAM': CBAM,
            'ECA': ECA,
            'SPP_Enhanced': SPP_Enhanced,
            'FPN_Enhanced': FPN_Enhanced,
            'RepVGGBlock': RepVGGBlock,
            'GhostConv_Enhanced': GhostConv_Enhanced,
            'C3k2_Enhanced': C3k2_Enhanced,
            'ASFF': ASFF,
        }
        
        for name, module_class in module_dict.items():
            setattr(modules, name, module_class)
            
        print("✅ Registered to ultralytics.nn.modules")
        registration_success = True
        
    except Exception as e:
        print(f"⚠️ Failed to register to modules: {e}")
    
    # 方法3: 动态添加到 __all__ 列表
    try:
        import ultralytics.nn.tasks as tasks
        
        # 确保 tasks 模块有 __all__ 属性
        if not hasattr(tasks, '__all__'):
            tasks.__all__ = []
            
        # 添加所有模块名到 __all__ 列表
        custom_modules = [
            'SnakeDeformableConv', 'C3k2_SnakeDeformable',
            'BiFPNLayer', 'BiFPNBlock', 'TripleBiFPN', 'MultiScaleBiFPN',
            'MicroDefectAttention', 'EnhancedDetectHead',
            'SEAttention', 'CBAM', 'ECA', 'SPP_Enhanced', 'FPN_Enhanced',
            'RepVGGBlock', 'GhostConv_Enhanced', 'C3k2_Enhanced', 'ASFF'
        ]
        
        for module_name in custom_modules:
            if module_name not in tasks.__all__:
                tasks.__all__.append(module_name)
                
        print("✅ Added to __all__ list")
        registration_success = True
        
    except Exception as e:
        print(f"⚠️ Failed to add to __all__: {e}")
    
    # 方法4: 检查并报告注册状态
    try:
        import ultralytics.nn.tasks as tasks
        
        # 检查关键模块是否已注册
        key_modules = ['C3k2_SnakeDeformable', 'BiFPNBlock', 'MicroDefectAttention']
        registered_modules = []
        missing_modules = []
        
        for module_name in key_modules:
            if hasattr(tasks, module_name):
                registered_modules.append(module_name)
            else:
                missing_modules.append(module_name)
                
        if registered_modules:
            print(f"✅ Verified registered modules: {registered_modules}")
            
        if missing_modules:
            print(f"⚠️ Missing modules: {missing_modules}")
            
    except Exception as e:
        print(f"⚠️ Failed to verify registration: {e}")
    
    if registration_success:
        print("🎉 Module registration completed successfully!")
    else:
        print("❌ Module registration failed!")
        
    return registration_success


if __name__ == "__main__":
    register_all_modules()