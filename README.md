# RL Robot Arm – Panda Conveyor Belt Task

Reinforcement Learning environment to train a **Franka Emika Panda** robotic arm to pick up a cube and place it onto a moving conveyor belt. Built with MuJoCo, Gymnasium, and Stable Baselines3.

## Task

The robot arm must:
1. Reach and grasp a cube
2. Lift it off the ground
3. Place it onto the moving conveyor belt
4. Release it at the correct position

A magnetic grasp system attaches the cube to the gripper on contact and releases it on command.

## Project Structure

```
rl_robot_arm/
├── env_conveyor.py          # Low-level MuJoCo simulation environment
├── gym_env.py               # Gymnasium wrapper with reward shaping & config
├── train_sb3.py             # PPO training script
├── eval_sb3.py              # Model evaluation script
├── demo_conveyor.py         # Interactive demo & manual control
├── plot_training.py         # Training curve plots
├── plot_hausarbeit.py       # Publication-quality plots
├── requirements.txt         # Python dependencies
└── assets/
    └── conveyor_belt/models/
        ├── conveyor_scene.xml   # Main MuJoCo scene
        ├── panda_local.xml      # Franka Panda robot definition
        └── conveyor.xml         # Conveyor belt model
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- `mujoco >= 2.3.0`
- `gymnasium >= 0.29.0`
- `stable-baselines3 >= 2.3.0`
- `numpy >= 1.21.0`

Optional: `tensorboard`, `matplotlib`

---

## Training (`train_sb3.py`)

Trainiert einen PPO-Agenten mit Stable Baselines3. Unterstützt parallele Umgebungen, Entropie-Scheduling und optionale GUI-Darstellung.

```bash
python train_sb3.py [OPTIONEN]
```

| Option | Standard | Beschreibung |
|---|---|---|
| `--timesteps N` | `200000` | Gesamtanzahl der Trainings-Schritte |
| `--num-envs N` | `1` | Anzahl paralleler Umgebungen (SubprocVecEnv) |
| `--gui` | aus | MuJoCo-Viewer während des Trainings öffnen |
| `--render-every N` | `0` | Bei `--num-envs > 1`: alle N Rollouts eine Episode im GUI rendern (0 = aus) |
| `--gui-sleep SEC` | `0.02` | Wartezeit pro Schritt beim Rendern (Sekunden) |
| `--control-conveyor` | aus | Agent steuert zusätzlich die Bandgeschwindigkeit |
| `--ent-coef F` | `0.01` | Anfangs-Entropiekoeffizient (PPO) |
| `--ent-coef-final F` | `0.003` | End-Entropiekoeffizient (linear abfallend) |
| `--log-dir DIR` | `logs` | Verzeichnis für TensorBoard-Logs |
| `--run-name NAME` | auto | Name des Runs (Standard: `ppo_panda_YYYYMMDD_HHMMSS`) |

**PPO-Hyperparameter (fest):**
- `n_steps=512`, `batch_size=256`, `n_epochs=5`
- `clip_range=0.15`, `vf_coef=0.5`, `max_grad_norm=0.5`
- Lernrate: linear von `3e-4` auf `3e-5`
- SDE aktiviert (`use_sde=True`, `sde_sample_freq=4`)

**Ausgabe:** Modell (`ppo_panda_conveyor.zip`) und Normalisierungsstatistiken (`vecnormalize.pkl`) werden im Run-Verzeichnis gespeichert.

**Beispiele:**

```bash
# Schnelles Training mit 4 parallelen Envs
python train_sb3.py --timesteps 500000 --num-envs 4

# Training mit GUI-Vorschau alle 5 Rollouts
python train_sb3.py --num-envs 4 --gui --render-every 5

# Langer Lauf mit Entropie-Scheduling und eigenem Namen
python train_sb3.py --timesteps 1000000 --num-envs 8 \
    --ent-coef 0.05 --ent-coef-final 0.001 \
    --run-name experiment_v1

# Agent steuert auch das Förderband
python train_sb3.py --control-conveyor --timesteps 300000 --num-envs 4
```

---

## Evaluation (`eval_sb3.py`)

Lädt ein trainiertes Modell und wertet es über mehrere Episoden aus. Gibt Erfolgsrate, Reward-Statistiken und Meilensteinraten aus.

```bash
python eval_sb3.py [OPTIONEN]
```

| Option | Standard | Beschreibung |
|---|---|---|
| `--model PATH` | `ppo_panda_conveyor` | Pfad zur gespeicherten `.zip`-Modelldatei |
| `--algo {ppo,a2c}` | `ppo` | Algorithmus des gespeicherten Modells |
| `--episodes N` | `30` | Anzahl der Evaluierungs-Episoden |
| `--gui` | aus | Simulation grafisch darstellen |
| `--gui-sleep SEC` | `0.03` | Wartezeit pro Schritt beim Rendern |
| `--vecnormalize PATH` | auto | Pfad zur `vecnormalize.pkl` (Standard: neben dem Modell) |
| `--deterministic` | an | Deterministische Aktionen (empfohlen für Evaluation) |
| `--stochastic` | aus | Stochastische Aktionen |
| `--control-conveyor` | aus | Agent steuert die Bandgeschwindigkeit |
| `--conveyor-speed F` | `5.0` | Bandgeschwindigkeit (m/s) wenn nicht vom Agenten gesteuert |
| `--max-steps N` | `700` | Maximale Schritte pro Episode |
| `--randomize-cube` | an | Würfel-Startposition zufällig |
| `--fixed-cube` | aus | Würfel immer an fester Position |
| `--seed N` | `42` | Zufalls-Seed |

**Ausgabe pro Episode:**
```
[eval] episode 001 reward=+12.345 success=True done_reason=success_on_belt_end release_source=agent
```

**Zusammenfassung am Ende:**
```
[summary] success_rate=0.800 (24/30)
[summary] milestone_rates= grasp:0.933 on_belt:0.867 end:0.800
```

**Beispiele:**

```bash
# Standard-Evaluation mit GUI
python eval_sb3.py --model logs/experiment_v1/ppo_panda_conveyor.zip --gui --episodes 20

# 100 Episoden headless, feste Würfelposition
python eval_sb3.py --model logs/experiment_v1/ppo_panda_conveyor.zip \
    --episodes 100 --fixed-cube

# Stochastische Aktionen testen
python eval_sb3.py --model logs/experiment_v1/ppo_panda_conveyor.zip \
    --stochastic --episodes 50
```

---

## Demo (`demo_conveyor.py`)

Zeigt die Simulation ohne trainierten Agenten. Vier Modi stehen zur Verfügung:

```bash
python demo_conveyor.py [--manual | --manual-train | --ik]
```

| Modus | Befehl | Beschreibung |
|---|---|---|
| Scripted Demo *(Standard)* | `python demo_conveyor.py` | Skriptgesteuerter Bewegungsablauf: Arm führt koordinierte Sinusbewegung aus, hebt den Würfel auf und legt ihn aufs Band. Nach Abschluss bleibt der Viewer mit sanfter Dauerbewegung offen. |
| Manuelle Steuerung (Basis) | `python demo_conveyor.py --manual` | Tkinter-GUI mit Schiebereglern für alle 7 Gelenke und die Bandgeschwindigkeit. Buttons: *Zero All*, *Release Cube*, *Quit*. Nutzt direkt `PandaConveyorEnv` (Position-Control). |
| Manuelle Steuerung (Training) | `python demo_conveyor.py --manual-train` | Wie `--manual`, aber mit dem Gymnasium-Interface (`PandaConveyorGym`). Regler steuern Gelenkgeschwindigkeiten (Velocity-Control) und den Release-Befehl. Zeigt Reward-Parts live in der Konsole. |
| IK Pick-and-Place | `python demo_conveyor.py --ik` | Vollautomatische Demo mit Jacobian-basierter inverser Kinematik. Der Arm nähert sich dem Würfel, greift ihn (Weld-Constraint), hebt ihn an und legt ihn auf dem Band ab. Kein trainiertes Modell nötig. |

---

## Monitoring

```bash
tensorboard --logdir logs/
```

Geloggte Metriken:
- `rollout/reached_grasped_rate` – Anteil Episoden mit erfolgreichem Greifvorgang
- `rollout/reached_on_belt_rate` – Anteil Episoden mit Platzierung auf dem Band
- `rollout/grasped_ratio_mean` – Ø Zeitanteil pro Episode, in der gegriffen wurde
- `train/ent_coef` – aktueller Entropiekoeffizient

---

## Environment Details

**Observation space** (28-dim):
- 7 Gelenkpositionen + 7 Gelenkgeschwindigkeiten
- Würfelposition & -orientierung
- Bandposition & -geschwindigkeit
- Greifer-Zustand & Greif-Flag

**Action space** (7 oder 8 dim):
- 7 Gelenkgeschwindigkeiten (Velocity-Control)
- Optional: + 1 Release-Befehl (Agent löst Griff)
- Optional: + 1 Bandgeschwindigkeit (bei `--control-conveyor`)

**Reward-Struktur** (mehrstufig):
1. Annähern an den Würfel
2. Greifen & Anheben
3. Transport zum Förderband
4. Erfolgreiche Ablage auf dem Band

**Algorithmus:** PPO mit VecNormalize, linearem Lernraten-Schedule (3e-4 → 3e-5) und Entropie-Scheduling.
