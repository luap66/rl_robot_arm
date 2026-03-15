# Franka Panda Robot Environment with Conveyor Belt (MuJoCo)

**Eine vollständige Simulationsumgebung für den Franka Emika Panda 7-DOF Roboterarm mit realistische Fließband-Integration in MuJoCo.**

## ✨ Features

✅ **Franka Panda 7-DOF Robot Arm**
- 7 Gelenke mit vollständiger Kinematik und Dynamik
- Realistische Inertialatrizen
- Gelenkpositionen- und Geschwindigkeitskontrolle

✅ **Conveyor Belt (Fließband)**
- STEP CAD → STL Mesh konvertiert (986 Vertices, 48 KB)
- Variable Geschwindigkeit (0-1 m/s)

✅ **MuJoCo Physics Engine**
- Hochleistungs-Simulation (1000 Hz)
- Stabile numerische Integration

✅ **Gym-kompatible API**
- `reset()` - Home-Position
- `step(action)` - Simulationsschritt
- `get_end_effector_pose()` - EE Pose

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python3 demo_conveyor.py
```

## 📁 Struktur

```
rl_robot_arm/
├── env_conveyor.py          # Main Environment
├── demo_conveyor.py         # Demo
├── test_suite.py            # Tests (7/7 ✅)
├── conveyor_scene.xml       # MuJoCo Scene
├── create_scene.py          # Scene Generator
└── assets/
    ├── Belt Conveyor.STEP   # Original CAD
    └── conveyor_belt.stl    # Vereinfachtes Mesh
```

## 📚 Dokumentation

Siehe [DOCUMENTATION.md](DOCUMENTATION.md) für:
- API Reference
- RL Training Beispiele
- Troubleshooting
- Technische Details

## 💡 Beispiel

```python
from env_conveyor import PandaConveyorEnv
import numpy as np

env = PandaConveyorEnv(gui=False)
obs = env.reset()

for step in range(500):
    action = np.random.uniform(-1, 1, 7)
    obs, reward, done, info = env.step(action, conveyor_speed=0.5)

env.close()
```

## 🧪 Test Results

```
Result: 7/7 tests passed
🎉 ALL TESTS PASSED!
```

## 📊 Spezifikationen

| Eigenschaft | Wert |
|-----------|------|
| Joints | 7 revolute |
| Timestep | 0.001s (1 kHz) |
| Gravity | -9.81 m/s² |
| Conveyor Mesh | 986 vertices |

---

**Status**: ✅ Production Ready | Fully Tested
