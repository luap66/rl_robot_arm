# Franka Panda Robot Arm with Conveyor Belt (MuJoCo)

Eine vollständige Simulationsumgebung für den Franka Emika Panda 7-DOF Roboterarm mit einem realistischen Fließband-Modell basierend auf MuJoCo.

## Features

✅ **Franka Panda 7-DOF Arm**
- 7 Gelenke mit Aktoren
- Realistisches Inertia-Modell
- Gelenkpositions- und Geschwindigkeitskontrolle

✅ **Conveyor Belt (Fließband)**
- STEP CAD-Datei konvertiert zu STL-Mesh
- Vereinfachte Geometrie (986 Vertices, 48 KB)
- Physikalisch simuliert in MuJoCo

✅ **MuJoCo Physics Engine**
- Hochleistungs-Simulation
- Stabile numerische Integration
- Kollisionserkennung

## Architektur

```
rl_robot_arm/
├── env_conveyor.py          # PandaConveyorEnv Klasse
├── demo_conveyor.py         # Demonstrations-Skript
├── create_scene.py          # Scene XML Generator
├── conveyor_scene.xml       # Generated MuJoCo Scene
├── assets/
│   ├── Belt Conveyor.STEP   # Original CAD Datei
│   └── conveyor_belt.stl    # Vereinfachtes Mesh
├── requirements.txt         # Python Dependencies
└── README.md               # Dieses File
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone oder öffne das Projekt
cd rl_robot_arm

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `mujoco>=1.0.0` - Physics Engine
- `numpy` - Numerische Berechnungen
- `trimesh` - Mesh-Processing (für STEP→STL Konvertierung)

## Verwendung

### 1. Basis-Beispiel

```python
from env_conveyor import PandaConveyorEnv
import numpy as np

# Umgebung initialisieren
env = PandaConveyorEnv(gui=False)

# Reset zur Home-Position
obs = env.reset()
print(f"Joint angles: {obs}")

# Simulation step mit Gelenkbefehlen
action = np.random.uniform(-1, 1, 7)  # 7 DOF velocities
obs, reward, done, info = env.step(action, conveyor_speed=0.5)

# Get End-Effector Position
ee_pos, ee_orn = env.get_end_effector_pose()
print(f"EE Position: {ee_pos}")
```

### 2. Demo ausführen

```bash
python3 demo_conveyor.py
```

Output:
```
============================================================
DEMO: Franka Panda with Conveyor Belt
============================================================
✓ Loaded 7 arm joints and 7 actuators

1. Reset robot to home position...
   ✓ Home position: [ 0. 0. 0. -0.785 0. 1.571 0.785]
   ...

✓ Demo completed!
```

### 3. Test der Environment

```bash
python3 env_conveyor.py
```

## API Dokumentation

### `PandaConveyorEnv`

#### Constructor

```python
env = PandaConveyorEnv(gui: bool = False, dt: float = 0.001)
```

**Parameters:**
- `gui`: MuJoCo Viewer aktivieren (Standard: False)
- `dt`: Timestep für Simulation in Sekunden (Standard: 0.001)

#### Methods

##### `reset()` → `np.ndarray`
Setzt den Roboter in die Home-Position zurück.

```python
obs = env.reset()  # obs.shape = (7,)
```

Returns: Joint positions (7 DOF)

##### `step(action, conveyor_speed)` → `Tuple[np.ndarray, float, bool, Dict]`
Führt einen Simulationsschritt aus.

```python
action = np.array([v1, v2, v3, v4, v5, v6, v7])  # Joint velocities
obs, reward, done, info = env.step(action, conveyor_speed=0.5)
```

**Parameters:**
- `action`: Joint velocity commands (shape: 7)
- `conveyor_speed`: Fließband-Geschwindigkeit (0-1)

**Returns:**
- `obs`: Joint positions (shape: 7)
- `reward`: Reward signal (float)
- `done`: Episode finished (bool)
- `info`: Additional information (dict)

##### `get_end_effector_pose()` → `Tuple[np.ndarray, np.ndarray]`
Gibt Position und Orientierung des End-Effectors zurück.

```python
position, orientation = env.get_end_effector_pose()
# position.shape = (3,)  # x, y, z
# orientation.shape = (4,)  # x, y, z, w (quaternion)
```

##### `close()`
Schließt die Umgebung und gibt Ressourcen frei.

```python
env.close()
```

## Scene Konfiguration

Die MuJoCo Scene wird automatisch generiert aus [create_scene.py](create_scene.py):

### Komponenten

1. **Panda Roboter**
   - 7 Gelenke (joint1-joint7)
   - 7 Aktoren (actuator1-actuator7)
   - Realistische Massen und Inertialatrizen
   - Home Position: `[0, 0, 0, -π/4, 0, π/2, π/4]`

2. **Conveyor Belt**
   - STL Mesh: 986 Vertices, 48 KB
   - Materialien: Metall (grau)
   - Pos: `[0, -0.5, 0]` (relativ zur Basis)

3. **Physikalische Parameter**
   - Gravity: `[0, 0, -9.81]`
   - Timestep: 0.001s (1kHz)
   - Damping: 0.3 pro Gelenk

## Technische Details

### STEP → STL Konvertierung

Die STEP-Datei (Belt Conveyor 2000×400×700 mm) wurde konvertiert zu:
- **Original**: 1,601,944 Faces → **Vereinfacht**: 986 Faces
- **Methode**: Convex Hull Simplification (scipy.spatial)
- **Dateigröße**: 1.6 MB → 48.2 KB

### Coordinate System

```
     Z (up)
     |
     |--- Y
    /
   X
```

- **Panda Base**: [0, -0.5, 0]
- **Conveyor Belt**: Aligned with Panda

### Joint Ranges (in Radians)

| Joint | Min | Max | Max Velocity |
|-------|-----|-----|--------------|
| J1 | -2.897 | 2.897 | 2.175 |
| J2 | -1.763 | 1.763 | 2.175 |
| J3 | -2.897 | 2.897 | 2.175 |
| J4 | -3.072 | -0.070 | 2.175 |
| J5 | -2.897 | 2.897 | 2.610 |
| J6 | -0.018 | 3.753 | 2.610 |
| J7 | -2.897 | 2.897 | 2.610 |

## Beispiel RL-Training

```python
import numpy as np
from env_conveyor import PandaConveyorEnv

def train_reaching_policy():
    env = PandaConveyorEnv(gui=False)
    
    for episode in range(100):
        obs = env.reset()
        cumulative_reward = 0
        
        for step in range(500):
            # Simple policy: move towards target position
            target = np.array([0.3, 0, 0.5, -0.4, 0, 1.5, 0.8])
            action = 0.1 * (target - obs)
            
            obs, reward, done, _ = env.step(action, conveyor_speed=0.5)
            cumulative_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode+1}: Reward = {cumulative_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    train_reaching_policy()
```

## Debugging & Troubleshooting

### Problem: "XML Error: Schema violation"
**Lösung**: Stelle sicher, dass alle `<inertial>` Elemente die erforderlichen Attribute haben:
```xml
<inertial mass="1.0" pos="0 0 0" fullinertia="0.01 0.01 0.01 0 0 0"/>
```

### Problem: Segmentation Fault beim Beenden
**Lösung**: Immer `env.close()` aufrufen oder in try-finally verwenden:
```python
try:
    env = PandaConveyorEnv()
    # ... code ...
finally:
    env.close()
```

### Problem: Roboter bewegt sich nicht
**Überprüfe**:
1. Sind Aktoren richtig an Gelenke gebunden?
2. Ist die Action-Größe korrekt (7 DOF)?
3. Ist die action-Magnitude groß genug?

## Weitere Ressourcen

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Franka Emika Documentation](https://frankaemika.github.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## Changelog

### v1.0 (2024-01-XX)
- ✅ Initial implementation
- ✅ STEP to STL conversion
- ✅ Scene XML generation
- ✅ PandaConveyorEnv class
- ✅ Demo script

## License

MIT License - siehe LICENSE file für Details

## Support

Bei Fragen oder Problemen:
1. Überprüfe den Troubleshooting-Abschnitt
2. Führe `demo_conveyor.py` aus um zu überprüfen, dass die Installation korrekt ist
3. Überprüfe MuJoCo logs: `~/.mujoco/`

---

**Status**: ✅ Functional and Tested
