#!/usr/bin/env python3

"""
Test script for Industrial Reconstruction C++ Extensions
"""

import numpy as np
import time
import sys
import os

# Add the parent directory to the path to import the extensions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_image_buffer():
    """Test the ImageBuffer C++ extension"""
    print("Testing ImageBuffer...")
    
    try:
        import industrial_reconstruction_cpp as cpp_ext
        
        # Create image buffer
        buffer = cpp_ext.ImageBuffer(max_size=5)
        
        # Create test images
        depth_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        
        # Test push operation
        buffer.push(depth_img, color_img, timestamp)
        print(f"  ✓ Push operation successful")
        
        # Test size
        assert buffer.size() == 1, f"Expected size 1, got {buffer.size()}"
        print(f"  ✓ Size check passed: {buffer.size()}")
        
        # Test pop operation
        success, popped_depth, popped_color, popped_timestamp = buffer.try_pop()
        assert success, "Pop operation failed"
        assert np.array_equal(depth_img, popped_depth), "Depth image mismatch"
        assert np.array_equal(color_img, popped_color), "Color image mismatch"
        print(f"  ✓ Pop operation successful")
        
        # Test empty buffer
        success, _, _, _ = buffer.try_pop()
        assert not success, "Expected pop to fail on empty buffer"
        print(f"  ✓ Empty buffer handling correct")
        
        print("  ✓ ImageBuffer test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ ImageBuffer test failed: {e}")
        return False

def test_pose_calculator():
    """Test the PoseCalculator C++ extension"""
    print("Testing PoseCalculator...")
    
    try:
        import industrial_reconstruction_cpp as cpp_ext
        
        # Create pose calculator
        calculator = cpp_ext.PoseCalculator()
        
        # Test transformation matrix creation
        translation = np.array([1.0, 2.0, 3.0])
        rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        
        transform = calculator.create_transformation_matrix(translation, rotation)
        assert transform.shape == (4, 4), f"Expected 4x4 matrix, got {transform.shape}"
        print(f"  ✓ Transformation matrix creation successful")
        
        # Test distance calculations
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([3.0, 4.0, 0.0])
        distance = calculator.calculate_translation_distance(pos1, pos2)
        expected_distance = 5.0  # sqrt(3^2 + 4^2)
        assert abs(distance - expected_distance) < 1e-6, f"Expected {expected_distance}, got {distance}"
        print(f"  ✓ Translation distance calculation correct: {distance}")
        
        # Test pose processing
        pose1 = cpp_ext.TransformData()
        pose1.translation = pos1
        pose1.rotation = rotation
        pose1.timestamp = 0.0
        
        pose2 = cpp_ext.TransformData()
        pose2.translation = pos2
        pose2.rotation = rotation
        pose2.timestamp = 1.0
        
        should_process = calculator.should_process_pose(pose1, pose2, 2.0, 0.1)
        assert should_process, "Should process pose with large translation"
        print(f"  ✓ Pose processing logic correct")
        
        print("  ✓ PoseCalculator test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PoseCalculator test failed: {e}")
        return False

def test_memory_manager():
    """Test the MemoryManager C++ extension"""
    print("Testing MemoryManager...")
    
    try:
        import industrial_reconstruction_cpp as cpp_ext
        
        # Get global memory manager
        manager = cpp_ext.GlobalMemoryManager.get_image_manager()
        
        # Test memory allocation
        initial_memory = manager.get_allocated_memory()
        print(f"  ✓ Initial allocated memory: {initial_memory / 1024} KB")
        
        # Test preallocation
        manager.preallocate_buffers(5, 480, 640)
        pool_memory = manager.get_pool_memory()
        print(f"  ✓ Pool memory after preallocation: {pool_memory / 1024} KB")
        
        # Test temporary buffer allocation
        temp_depth = manager.get_temp_depth_image()
        temp_color = manager.get_temp_color_image()
        
        assert temp_depth.shape == (480, 640), f"Expected depth shape (480, 640), got {temp_depth.shape}"
        assert temp_color.shape == (480, 640, 3), f"Expected color shape (480, 640, 3), got {temp_color.shape}"
        print(f"  ✓ Temporary buffer allocation successful")
        
        # Test buffer return
        manager.return_temp_image(temp_depth)
        manager.return_temp_image(temp_color)
        print(f"  ✓ Buffer return successful")
        
        # Print memory statistics
        manager.print_memory_stats()
        
        print("  ✓ MemoryManager test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ MemoryManager test failed: {e}")
        return False

def test_performance():
    """Test performance improvements"""
    print("Testing Performance...")
    
    try:
        import industrial_reconstruction_cpp as cpp_ext
        
        # Test image buffer performance
        buffer = cpp_ext.ImageBuffer(max_size=100)
        
        # Create test images
        depth_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Time push operations
        num_operations = 1000
        start_time = time.time()
        
        for i in range(num_operations):
            buffer.push(depth_img, color_img, time.time())
        
        push_time = time.time() - start_time
        print(f"  ✓ {num_operations} push operations took {push_time:.3f} seconds")
        print(f"  ✓ Average time per push: {push_time/num_operations*1000:.3f} ms")
        
        # Time pop operations
        start_time = time.time()
        
        for i in range(num_operations):
            success, _, _, _ = buffer.try_pop()
            if not success:
                break
        
        pop_time = time.time() - start_time
        print(f"  ✓ {num_operations} pop operations took {pop_time:.3f} seconds")
        print(f"  ✓ Average time per pop: {pop_time/num_operations*1000:.3f} ms")
        
        print("  ✓ Performance test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Industrial Reconstruction C++ Extensions Test Suite")
    print("=" * 50)
    
    tests = [
        test_image_buffer,
        test_pose_calculator,
        test_memory_manager,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! C++ extensions are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
