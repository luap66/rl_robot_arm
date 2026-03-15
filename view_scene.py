#!/usr/bin/env python3
"""
MuJoCo Viewer - Zeigt Panda + Fließband in 3D
Einfache Visualisierung mit mujoco_viewer
"""

import numpy as np
import mujoco
import mujoco_viewer
from pathlib import Path
import sys


def main():
    print("="*70)
    print("MUJOCO 3D VIEWER - Franka Panda + Fließband")
    print("="*70)
    
    # Szene laden
    scene_file = Path(__file__).parent / "conveyor_scene.xml"
    
    if not scene_file.exists():
        print(f"❌ Szene nicht gefunden: {scene_file}")
        return
    
    print(f"\n📂 Lade Szene: {scene_file.name}")
    
    try:
        # Model und Data laden
        model = mujoco.MjModel.from_xml_path(str(scene_file))
        data = mujoco.MjData(model)
        
        print("✅ Szene geladen")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.njnt}")
        print(f"   Actuators: {model.nu}")
        print(f"   Geoms: {model.ngeom}")
        
        print("\n🎮 Starte 3D Viewer...")
        print("   Steuerung:")
        print("   - Linke Maustaste + Drag: Kamera drehen")
        print("   - Rechte Maustaste + Drag: Kamera verschieben")
        print("   - Mausrad: Zoomen")
        print("   - ESC: Beenden")
        print("   - [P]: Pause")
        print("   - [H]: Hilfe anzeigen")
        print("\n   Fenster wird geöffnet...")
        
        # Viewer erstellen
        viewer = mujoco_viewer.MujocoViewer(model, data)
        
        # Kamera Position
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = -0.5
        viewer.cam.lookat[2] = 0.5
        
        print("✅ Viewer gestartet!")
        print("\n💡 Roboterarm bewegt sich mit sinusförmigen Trajektorien")
        print("   Drücke ESC zum Beenden\n")
        
        # Conveyor belt animation settings
        conveyor_speed = 0.25

        # Simulationsloop
        step = 0
        while viewer.is_alive:
            # Controller: Sinusförmige Bewegung
            for i in range(model.nu):
                freq = 1.0 + i * 0.2
                amplitude = 0.5
                t = data.time
                data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)

            # Conveyor speed is handled in env_conveyor.py for physical transport
            
            # Simulation step
            mujoco.mj_step(model, data)
            
            # Viewer update
            viewer.render()
            
            step += 1
            
            # Status alle 500 Steps
            if step % 500 == 0:
                print(f"   Simulation läuft... Zeit: {data.time:.2f}s | Steps: {step}")
        
        # Cleanup
        viewer.close()
        print("\n✅ Viewer geschlossen")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Viewer beendet (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        sys.exit(1)
