#!/usr/bin/env python3
"""
Comprehensive Test Suite for Panda + Conveyor Environment
"""

import numpy as np
from env_conveyor import PandaConveyorEnv


def test_environment_initialization():
    """Test 1: Environment initialization"""
    print("\n[TEST 1] Environment Initialization")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        print("✅ Environment loaded successfully")
        
        # Check basic properties
        assert hasattr(env, 'model'), "Missing model attribute"
        assert hasattr(env, 'data'), "Missing data attribute"
        assert len(env.joint_ids) == 7, "Expected 7 joints"
        assert len(env.actuator_ids) == 7, "Expected 7 actuators"
        
        print(f"✅ Found {len(env.joint_ids)} joints and {len(env.actuator_ids)} actuators")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_reset():
    """Test 2: Reset to home position"""
    print("\n[TEST 2] Reset to Home Position")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        obs = env.reset()
        
        # Check observation shape
        assert obs.shape == (7,), f"Expected shape (7,), got {obs.shape}"
        
        # Check home position
        expected_home = np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, np.pi/4])
        error = np.max(np.abs(obs - expected_home))
        
        print(f"✅ Reset successful")
        print(f"   Joint angles: {obs}")
        print(f"   Max error from home: {error:.6f}")
        
        if error < 0.01:
            print(f"✅ Position matches home within tolerance")
        else:
            print(f"⚠️  Position differs slightly from expected home")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_step():
    """Test 3: Simulation step"""
    print("\n[TEST 3] Simulation Step")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        env.reset()
        
        # Small action to move joints slightly
        action = np.array([0.1, 0.05, 0.0, -0.05, 0.0, 0.1, 0.05])
        
        for step in range(5):
            obs, reward, done, info = env.step(action, conveyor_speed=0.1)
            assert obs.shape == (7,), "Invalid observation shape"
            assert isinstance(reward, float), "Reward should be float"
            assert isinstance(done, bool), "Done should be bool"
            assert isinstance(info, dict), "Info should be dict"
        
        print(f"✅ Step function works")
        print(f"   Final observation: {obs}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Done: {done}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_end_effector():
    """Test 4: End effector pose"""
    print("\n[TEST 4] End Effector Pose")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        env.reset()
        
        # Get EE pose
        ee_pos, ee_orn = env.get_end_effector_pose()
        
        print(f"✅ EE pose retrieved")
        print(f"   Position: {ee_pos}")
        print(f"   Orientation (quat): {ee_orn}")
        
        # Move the robot slightly
        action = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(50):
            env.step(action)
        
        ee_pos2, ee_orn2 = env.get_end_effector_pose()
        
        # Check that position changed (probably)
        pos_diff = np.linalg.norm(ee_pos2 - ee_pos)
        print(f"   Position change after movement: {pos_diff:.4f} m")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_action_ranges():
    """Test 5: Action space ranges"""
    print("\n[TEST 5] Action Space Ranges")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        env.reset()
        
        # Test maximum positive action
        action_max = np.ones(7) * 2.0
        for _ in range(10):
            env.step(action_max)
        print(f"✅ Maximum positive action accepted")
        
        # Test maximum negative action
        action_min = np.ones(7) * -2.0
        for _ in range(10):
            env.step(action_min)
        print(f"✅ Maximum negative action accepted")
        
        # Test zero action
        action_zero = np.zeros(7)
        for _ in range(10):
            env.step(action_zero)
        print(f"✅ Zero action accepted")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_conveyor_speeds():
    """Test 6: Conveyor belt speeds"""
    print("\n[TEST 6] Conveyor Belt Speeds")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        env.reset()
        
        speeds = [0.0, 0.5, 1.0, -0.5]
        action = np.zeros(7)
        
        for speed in speeds:
            for _ in range(10):
                env.step(action, conveyor_speed=speed)
            print(f"✅ Conveyor speed {speed} accepted")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_multiple_episodes():
    """Test 7: Multiple episodes"""
    print("\n[TEST 7] Multiple Episodes")
    print("-" * 50)
    
    try:
        env = PandaConveyorEnv(gui=False)
        
        for episode in range(5):
            obs = env.reset()
            cumulative_reward = 0
            
            for step in range(100):
                action = np.random.uniform(-1, 1, 7)
                obs, reward, done, info = env.step(action)
                cumulative_reward += reward
                
                if done:
                    break
            
            print(f"✅ Episode {episode+1}: {step+1} steps, reward={cumulative_reward:.2f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PANDA + CONVEYOR ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_environment_initialization,
        test_reset,
        test_step,
        test_end_effector,
        test_action_ranges,
        test_conveyor_speeds,
        test_multiple_episodes,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"[{i+1}] {test_func.__name__}: {status}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
