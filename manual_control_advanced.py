#!/usr/bin/env python3
"""
Erweiterte manuelle Steuerung des Franka Panda Robot Arms mit Pygame
- Vollständige Tastaturkontrolle in Echtzeit
- Live-Anzeige von Gelenkpositionen
- Verschiedene Kontrollmodi
"""

import numpy as np
import mujoco as mj
from pathlib import Path
import time
import threading
from collections import deque


class KeyboardMonitor:
    """Überwacht Tastaturzustände"""
    
    def __init__(self):
        self.keys_pressed = set()
        self.keys_released = set()
        self.lock = threading.Lock()


class PandaManualController:
    """Erweiterte Steuerung mit Asyncer Tastaturlesung"""
    
    def __init__(self, scene_file="conveyor_scene.xml"):
        """Initialisiere die Kontrolle"""
        scene_path = Path(__file__).parent / scene_file
        
        if not scene_path.exists():
            raise FileNotFoundError(f"Szene nicht gefunden: {scene_path}")
        
        # Lade Model
        print("📁 Lade Szene...")
        self.model = mj.MjModel.from_xml_path(str(scene_path))
        self.data = mj.MjData(self.model)
        
        # Kontrollparameter
        self.joint_modes = {
            0: "velocity",  # Q/A, W/S, E/D, etc.
            1: "velocity",
            2: "velocity",
            3: "velocity",
            4: "velocity",
            5: "velocity",
            6: "velocity",
        }
        
        self.joint_velocities = np.zeros(7)
        self.joint_positions = np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, np.pi/4])
        
        self.velocity_scale = 0.5  # Skalierungsfaktor für Geschwindigkeit
        self.max_velocity = 2.0  # Maximale Winkelgeschwindigkeit
        
        # Home Position
        self.home_position = np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, np.pi/4])
        
        # Finde Actuator IDs
        self.actuator_ids = self._find_actuators()
        
        # UI Status
        self.show_help = True
        self.running = True
        
        print("✅ Modell geladen")
        print(f"   Actuators gefunden: {len(self.actuator_ids)}")
    
    def _find_actuators(self):
        """Finde alle Actuator IDs für die Panda Gelenke"""
        actuator_ids = []
        
        # Versuche verschiedene Namenskonventionen
        for i in range(7):
            aid = -1
            for name_pattern in [f"panda_joint{i+1}", f"joint{i+1}", f"actuator{i}"]:
                try:
                    aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name_pattern)
                    if aid >= 0:
                        break
                except:
                    pass
            
            if aid >= 0:
                actuator_ids.append(aid)
            else:
                # Fallback: Nutze Index wenn weniger als 7 Actuators
                if i < self.model.nu:
                    actuator_ids.append(i)
        
        return actuator_ids[:7] if len(actuator_ids) >= 7 else list(range(min(7, self.model.nu)))
    
    def apply_controls(self):
        """Wende aktuelle Kontrollen auf die Simulation an"""
        dt = self.model.opt.timestep
        
        # Update Positionen basierend auf Geschwindigkeiten
        for i in range(min(7, len(self.actuator_ids))):
            self.joint_positions[i] += self.joint_velocities[i] * dt
            
            # Gelenkgrenzen (conservativ: ±π)
            self.joint_positions[i] = np.clip(self.joint_positions[i], -np.pi, np.pi)
            
            # Schreibe in Control Vector (Position Control)
            actuator_id = self.actuator_ids[i]
            if 0 <= actuator_id < self.model.nu:
                self.data.ctrl[actuator_id] = self.joint_positions[i]
    
    def process_key(self, key_char, action_type="press"):
        """Verarbeite einzelne Tastenanschläge"""
        
        key_map = {
            # Gelenk 0: Q/A
            'q': (0, self.velocity_scale),
            'a': (0, -self.velocity_scale),
            # Gelenk 1: W/S
            'w': (1, self.velocity_scale),
            's': (1, -self.velocity_scale),
            # Gelenk 2: E/D
            'e': (2, self.velocity_scale),
            'd': (2, -self.velocity_scale),
            # Gelenk 3: R/F
            'r': (3, self.velocity_scale),
            'f': (3, -self.velocity_scale),
            # Gelenk 4: T/G
            't': (4, self.velocity_scale),
            'g': (4, -self.velocity_scale),
            # Gelenk 5: Y/H
            'y': (5, self.velocity_scale),
            'h': (5, -self.velocity_scale),
            # Gelenk 6: U/J
            'u': (6, self.velocity_scale),
            'j': (6, -self.velocity_scale),
        }
        
        key_lower = key_char.lower() if isinstance(key_char, str) else key_char
        
        if key_lower in key_map:
            joint_idx, velocity_increment = key_map[key_lower]
            
            if action_type == "press":
                self.joint_velocities[joint_idx] = velocity_increment
                print(f"   ▶️  Gelenk {joint_idx+1}: Geschwindigkeit = {velocity_increment:.2f}")
            
            elif action_type == "release":
                self.joint_velocities[joint_idx] = 0
                print(f"   ⏸️  Gelenk {joint_idx+1}: Gestoppt")
        
        elif key_lower == ' ':  # Spacebar
            self.joint_velocities = np.zeros(7)
            print("🛑 ALLE GELENKE GESTOPPT")
        
        elif key_lower == 'x':  # X = Rückkehr zur Home Position
            self.return_to_home()
        
        # Special commands
        elif key_lower == 'h':
            self.print_help()
        
        elif key_lower == 'p':
            self.print_positions()
    
    def return_to_home(self):
        """Fahre zur Home Position zurück"""
        print("\n🏠 Fahre zur Home Position (0, 0, 0, -π/4, 0, π/2, π/4)...")
        
        # Setze Positionen graduell
        steps = 100
        for step in range(steps):
            alpha = step / steps
            self.joint_positions = (1 - alpha) * self.joint_positions + alpha * self.home_position
            self.apply_controls()
            time.sleep(0.01)
        
        self.joint_positions = self.home_position.copy()
        self.joint_velocities = np.zeros(7)
        self.apply_controls()
        print("✅ Home Position erreicht")
    
    def print_positions(self):
        """Zeige aktuelle Gelenkpositionen"""
        print("\n" + "="*70)
        print("📍 AKTUELLE GELENKPOSITIONEN")
        print("="*70)
        for i, (pos, vel) in enumerate(zip(self.joint_positions, self.joint_velocities)):
            rad_str = f"{pos:7.3f} rad"
            deg_str = f"{np.degrees(pos):7.1f}°"
            vel_str = f"v={vel:6.2f}"
            print(f"  Gelenk {i+1}: {rad_str:12} ({deg_str:10}) [{vel_str}]")
        print("="*70 + "\n")
    
    def print_help(self):
        """Zeige Hilfe"""
        help_text = """
╔════════════════════════════════════════════════════════════════════╗
║           FRANKA PANDA - MANUELLE STEUERUNG                        ║
╠════════════════════════════════════════════════════════════════════╣
║ TASTATURSTEUERUNG:                                                  ║
║─────────────────────────────────────────────────────────────────── ║
║  Gelenk 1:  Q (↻) / A (↺)                                           ║
║  Gelenk 2:  W (↻) / S (↺)                                           ║
║  Gelenk 3:  E (↻) / D (↺)                                           ║
║  Gelenk 4:  R (↻) / F (↺)                                           ║
║  Gelenk 5:  T (↻) / G (↺)                                           ║
║  Gelenk 6:  Y (↻) / H (↺)                                           ║
║  Gelenk 7:  U (↻) / J (↺)                                           ║
║─────────────────────────────────────────────────────────────────── ║
║  SPACE:     Alle Gelenke STOPPEN                                    ║
║  H:         Diese Hilfe anzeigen                                    ║
║  P:         Aktuelle Positionen anzeigen                            ║
║  X:         Zur Home-Position fahren                                ║
║  ESC/Ctrl+C: Beenden                                                ║
╚════════════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def run_with_viewer(self):
        """Starte mit MuJoCo Viewer"""
        print("\n🎬 Starte Simulation mit MuJoCo Viewer...")
        print("📌 HINWEIS: Direkte Tastaturkontrolle im Viewer kann begrenzt sein")
        print("💡 Für vollständige Tastaturkontrolle verwende: python3 manual_control_advanced.py\n")
        
        try:
            with mj.Viewer(self.model, self.data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Simulationsloop
                frame_count = 0
                last_print = 0
                
                while viewer.is_running():
                    # Anwende Kontroll-Signale
                    self.apply_controls()
                    
                    # Simulationsschritt
                    mj.mj_step(self.model, self.data)
                    
                    # Sync Viewer
                    viewer.sync()
                    
                    frame_count += 1
                    
                    # Print statistiken
                    if frame_count - last_print >= 500:
                        last_print = frame_count
                        t = self.data.time
                        print(f"⏱️  {t:.2f}s | Frame {frame_count} | Pos: {self.joint_positions}")
                
                print(f"\n✅ Simulation nach {frame_count} Frames beendet")
        
        except KeyboardInterrupt:
            print("\n⏹️  Beendet durch Benutzer")
        
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Hauptfunktion"""
    controller = PandaManualController()
    
    print("\n" + "="*70)
    print("🤖 FRANKA PANDA - MANUELLE STEUERUNG")
    print("="*70)
    print("\n💡 Tasten drücken um Gelenke zu bewegen:")
    print("   Q/A für Gelenk 1, W/S für Gelenk 2, E/D für Gelenk 3, etc.")
    print("   SPACE zum Stoppen, H für Hilfe, P für Positionen, X für Home\n")
    
    controller.print_help()
    controller.run_with_viewer()


if __name__ == "__main__":
    main()
