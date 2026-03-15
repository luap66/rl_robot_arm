#!/usr/bin/env python3
"""
MuJoCo 3D Viewer - Zeigt Panda + Fließband in 3D
"""

import numpy as np
import mujoco
from pathlib import Path
import sys


def launch_viewer():
    """Startet den MuJoCo 3D Viewer"""
    
    print("="*70)
    print("MUJOCO 3D VIEWER - Franka Panda + Fließband")
    print("="*70)
    
    # Szene laden
    scene_file = Path(__file__).parent / "conveyor_scene.xml"
    
    if not scene_file.exists():
        print(f"❌ Szene nicht gefunden: {scene_file}")
        return False
    
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
        print("   - Space: Pause/Play")
        print("   - ESC/Q: Beenden")
        print("\n")
        
        # Versuche verschiedene Viewer APIs
        
        # Option 1: mujoco.viewer.launch (neueste API)
        if hasattr(mujoco, 'viewer') and hasattr(mujoco.viewer, 'launch'):
            print("✅ Verwende mujoco.viewer.launch")
            
            def controller(model, data):
                """Controller für sinusförmige Bewegung"""
                for i in range(model.nu):
                    freq = 1.0 + i * 0.2
                    amplitude = 0.5
                    t = data.time
                    data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
            
            # Starte Viewer
            mujoco.viewer.launch(model, data)
            return True
        
        # Option 2: mujoco.viewer.launch_passive
        elif hasattr(mujoco, 'viewer') and hasattr(mujoco.viewer, 'launch_passive'):
            print("✅ Verwende mujoco.viewer.launch_passive")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Kamera einstellen
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Simulationsloop
                while viewer.is_running():
                    # Controller
                    for i in range(model.nu):
                        freq = 1.0 + i * 0.2
                        amplitude = 0.5
                        t = data.time
                        data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
                    
                    # Simulation step
                    mujoco.mj_step(model, data)
                    viewer.sync()
                
                return True
        
        # Option 3: Verwende Python Viewer (falls verfügbar)
        else:
            print("⚠️  Standard Viewer APIs nicht verfügbar")
            print("   Versuche Python-basierte Lösung...")
            
            # Importiere Python Viewer falls verfügbar
            try:
                import glfw
                from mujoco import GLContext
                
                print("✅ Verwende GLFW + OpenGL Renderer")
                
                # GLFW initialisieren
                if not glfw.init():
                    print("❌ GLFW konnte nicht initialisiert werden")
                    return False
                
                # Fenster erstellen
                width, height = 1280, 720
                window = glfw.create_window(width, height, "MuJoCo - Panda + Conveyor", None, None)
                if not window:
                    glfw.terminate()
                    print("❌ Fenster konnte nicht erstellt werden")
                    return False
                
                glfw.make_context_current(window)
                glfw.swap_interval(1)
                
                # Renderer erstellen
                context = GLContext(width, height)
                context.make_current()
                
                # Kamera Setup
                cam = mujoco.MjvCamera()
                cam.distance = 3.0
                cam.azimuth = 45.0
                cam.elevation = -20.0
                cam.lookat[0] = 0.0
                cam.lookat[1] = -0.5
                cam.lookat[2] = 0.5
                
                # Visualisierungs-Optionen
                opt = mujoco.MjvOption()
                
                # Scene und Context
                scene = mujoco.MjvScene(model, maxgeom=10000)
                context_render = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
                
                # Viewport
                viewport = mujoco.MjrRect(0, 0, width, height)
                
                print("✅ Viewer gestartet - Fenster offen")
                
                # Main loop
                while not glfw.window_should_close(window):
                    # Controller
                    for i in range(model.nu):
                        freq = 1.0 + i * 0.2
                        amplitude = 0.5
                        t = data.time
                        data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
                    
                    # Simulation step
                    mujoco.mj_step(model, data)
                    
                    # Render
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(viewport, scene, context_render)
                    
                    # Swap buffers
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                
                # Cleanup
                glfw.terminate()
                print("\n✅ Viewer geschlossen")
                return True
                
            except ImportError as e:
                print(f"❌ GLFW nicht verfügbar: {e}")
                print("\n💡 Installiere GLFW mit:")
                print("   pip install glfw")
                return False
            
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    try:
        success = launch_viewer()
        if not success:
            print("\n⚠️  Viewer konnte nicht gestartet werden")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✅ Viewer beendet (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
