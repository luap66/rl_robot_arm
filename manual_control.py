#!/usr/bin/env python3
"""
Manuelle Steuerung des Franka Panda Robot Arms
- Tastaturgesteuerte Kontrolle aller 7 Gelenke
- MuJoCo Native Viewer mit Callbacks
"""

import numpy as np
import mujoco as mj
from pathlib import Path
import json


class ManualRobotController:
    """Keyboard-kontrollierter Panda Robot Arm"""
    
    def __init__(self, scene_file="conveyor_scene.xml"):
        """Initialisiere die Kontrolle"""
        scene_path = Path(__file__).parent / scene_file
        
        if not scene_path.exists():
            raise FileNotFoundError(f"Szene nicht gefunden: {scene_path}")
        
        # Model und Data laden
        self.model = mj.MjModel.from_xml_path(str(scene_path))
        self.data = mj.MjData(self.model)
        
        # Kontrolparameter
        self.joint_speeds = np.zeros(7)  # Geschwindigkeit für jeden Joint
        self.max_speed = 1.0  # Maximale Gelenkgeschwindigkeit
        self.speed_increment = 0.2  # Inkremente für Beschleunigung
        
        # Gebundene Tasten
        self.key_bindings = {
            # Gelenk 1 (Q/A)
            ord('q'): {'joint': 0, 'direction': 1},
            ord('a'): {'joint': 0, 'direction': -1},
            
            # Gelenk 2 (W/S)
            ord('w'): {'joint': 1, 'direction': 1},
            ord('s'): {'joint': 1, 'direction': -1},
            
            # Gelenk 3 (E/D)
            ord('e'): {'joint': 2, 'direction': 1},
            ord('d'): {'joint': 2, 'direction': -1},
            
            # Gelenk 4 (R/F)
            ord('r'): {'joint': 3, 'direction': 1},
            ord('f'): {'joint': 3, 'direction': -1},
            
            # Gelenk 5 (T/G)
            ord('t'): {'joint': 4, 'direction': 1},
            ord('g'): {'joint': 4, 'direction': -1},
            
            # Gelenk 6 (Y/H)
            ord('y'): {'joint': 5, 'direction': 1},
            ord('h'): {'joint': 5, 'direction': -1},
            
            # Gelenk 7 (U/J)
            ord('u'): {'joint': 6, 'direction': 1},
            ord('j'): {'joint': 6, 'direction': -1},
        }
        
        # Aktuelle Position speichern
        self.current_positions = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
        
        print("\n" + "="*70)
        print("MANUELLE ROBOT-STEUERUNG - Franka Panda")
        print("="*70)
        print("\n📋 TASTATURBELEGUNG:")
        print("-" * 70)
        print("Gelenk 1: Q = +  | A = -")
        print("Gelenk 2: W = +  | S = -")
        print("Gelenk 3: E = +  | D = -")
        print("Gelenk 4: R = +  | F = -")
        print("Gelenk 5: T = +  | G = -")
        print("Gelenk 6: Y = +  | H = -")
        print("Gelenk 7: U = +  | J = -")
        print("-" * 70)
        print("Weitere Tasten:")
        print("  SPACE: Alle Gelenke stoppen (Reset auf 0)")
        print("  R:     Zur Home-Position (0, 0, 0, -π/4, 0, π/2, π/4)")
        print("  P:     Aktuelle Position anzeigen")
        print("  ESC:   Schließen")
        print("="*70 + "\n")
        
        self.home_position = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
    
    def keyboard_callback(self, key, action, mods):
        """MuJoCo Keyboard Callback"""
        
        KEY_RELEASED = 0
        KEY_PRESSED = 1
        
        if action != KEY_PRESSED:
            return
        
        # Stopptaste
        if key == mj.mjtKey.mjKEY_SPACE:
            print("⏸️  Alle Gelenke gestoppt")
            self.joint_speeds = np.zeros(7)
            return
        
        # Home-Position
        if key == ord('r'):
            print("🏠 Fahre zur Home-Position...")
            self.current_positions = self.home_position.copy()
            self.joint_speeds = np.zeros(7)
            self.update_control()
            return
        
        # Position anzeigen
        if key == ord('p'):
            print("\n📍 Aktuelle Gelenkpositionen:")
            for i, pos in enumerate(self.current_positions):
                print(f"   Gelenk {i+1}: {pos:7.3f} rad ({np.degrees(pos):7.1f}°)")
            return
        
        # Joint-Steuerung
        if key in self.key_bindings:
            key_info = self.key_bindings[key]
            joint_idx = key_info['joint']
            direction = key_info['direction']
            
            # Änder Geschwindigkeit
            self.joint_speeds[joint_idx] = direction * self.max_speed
            
            sign = '+' if direction > 0 else '-'
            print(f"   Gelenk {joint_idx+1}: {sign} (Geschwindigkeit: {abs(self.joint_speeds[joint_idx]):.1f})")
    
    def controller(self):
        """MuJoCo Controller Callback - wird jeden Simulationsschritt aufgerufen"""
        # Aktualisiere Positionen basierend auf Geschwindigkeiten
        dt = self.model.opt.timestep
        
        for i in range(7):
            # Gelenkgrenzen beachten (vereinfacht)
            self.current_positions[i] += self.joint_speeds[i] * dt
            
            # Grenzen: ±π
            self.current_positions[i] = np.clip(self.current_positions[i], -np.pi, np.pi)
        
        self.update_control()
    
    def update_control(self):
        """Schreibe aktuelle Positionen in den Control Vector"""
        # Für Positions-Kontrolle: direktes Setzen der Gelenkpositionen
        for i in range(7):
            # Finde Actuator ID für Gelenk i
            actuator_id = self.find_actuator_id(i)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = self.current_positions[i]
    
    def find_actuator_id(self, joint_index):
        """Finde Actuator ID für einen Joint"""
        # Versuche verschiedene Namenskonventionen
        for name in [f"joint{joint_index+1}", f"actuator{joint_index}", f"panda_joint{joint_index+1}"]:
            try:
                aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    return aid
            except:
                pass
        return -1
    
    def run(self):
        """Starte den interaktiven Viewer mit Tastaturkontrolle"""
        print("🎮 Starte MuJoCo Viewer mit Tastaturkontrolle...\n")
        
        try:
            with mj.Viewer(self.model, self.data) as viewer:
                # Stelle Kamera ein
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Keyboard Callback registrieren
                viewer.user_scancodes = False
                
                # Simulation Loop
                while viewer.is_running():
                    # Keyboard input abfragen
                    if hasattr(viewer, 'keyboard'):
                        for key, action in viewer.keyboard:
                            self.keyboard_callback(key, action, 0)
                    
                    # Controller ausführen
                    self.controller()
                    
                    # Schritt simulieren
                    mj.mj_step(self.model, self.data)
                    
                    # Synchronisiere mit Viewer
                    viewer.sync()
                
                print("\n✅ Viewer geschlossen")
        
        except AttributeError as e:
            print(f"\n⚠️  MuJoCo Viewer API unterscheidet sich, verwende alternatives Modell...\n")
            self.run_alternative()
    
    def run_alternative(self):
        """Alternative ohne direkte Keyboard-Callbacks"""
        print("🎮 Starte Simulation mit Tastaturpuffer...\n")
        
        with mj.Viewer(self.model, self.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 45.0
            viewer.cam.elevation = -20.0
            
            last_key_check = 0
            
            while viewer.is_running():
                # Hier wäre Tastaturabfrage (MuJoCo Viewer bietet limitierte Keyboard-Unterstützung)
                
                # Controller ausführen
                self.controller()
                
                # Schritt simulieren
                mj.mj_step(self.model, self.data)
                
                # Synchronisiere mit Viewer
                viewer.sync()
            
            print("\n✅ Viewer geschlossen")


def main():
    """Hauptfunktion"""
    try:
        controller = ManualRobotController()
        controller.run()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation beendet durch Benutzer")
    
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
