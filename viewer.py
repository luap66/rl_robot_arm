#!/usr/bin/env python3
"""
MuJoCo Interactive Viewer for Panda + Conveyor
Displays the scene in the native MuJoCo viewer
"""

import numpy as np
import mujoco as mj
from pathlib import Path
import time


def main():
    print("=" * 70)
    print("MuJoCo INTERACTIVE VIEWER - Franka Panda + Conveyor Belt")
    print("=" * 70)
    
    # Load scene
    scene_file = Path(__file__).parent / "conveyor_scene.xml"
    
    if not scene_file.exists():
        print(f"❌ Scene file not found: {scene_file}")
        return
    
    print(f"\n📂 Loading scene: {scene_file.name}")
    
    try:
        # Load model
        model = mj.MjModel.from_xml_path(str(scene_file))
        data = mj.MjData(model)
        
        print("✅ Model loaded successfully")
        
        # Create viewer
        print("\n🎥 Starting MuJoCo Viewer...")
        print("   Controls:")
        print("   - Left mouse: Rotate view")
        print("   - Right mouse: Pan")
        print("   - Scroll: Zoom")
        print("   - Space: Play/Pause")
        print("   - ESC: Exit")
        print()
        
        viewer = mj.viewer.launch_passive(model, data)
        
        print("✅ Viewer started!")
        print("\n🤖 Running interactive simulation...")
        print("   (Move your mouse in the viewer window, use Space to pause)")
        
        # Get joint/actuator IDs
        joint_ids = []
        actuator_ids = []
        
        for i in range(1, 8):
            try:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}")
                if jid >= 0:
                    joint_ids.append(jid)
                
                aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                if aid >= 0:
                    actuator_ids.append(aid)
            except:
                pass
        
        # Simulation loop
        step_count = 0
        last_print = 0
        
        while viewer.is_running():
            # Sinusoidal motion for visualization
            t = step_count * model.opt.timestep
            
            if len(actuator_ids) > 0:
                for i, aid in enumerate(actuator_ids):
                    freq = 0.5 + i * 0.2  # Different frequencies
                    amplitude = 0.3
                    data.ctrl[aid] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
            
            # Step simulation
            mj.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            step_count += 1
            
            # Print progress every 500 steps
            if step_count - last_print >= 500:
                last_print = step_count
                elapsed = t
                print(f"   ⏱️  {elapsed:.1f}s - Step {step_count}")
        
        print(f"\n✅ Viewer closed after {step_count} steps ({t:.1f}s)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Hint: If you see 'No display found', you need:")
        print("   - X11 forwarding (SSH with -X flag)")
        print("   - Or VNC viewer")
        print("   - Or run 'export DISPLAY=:0' if you have a local display")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
