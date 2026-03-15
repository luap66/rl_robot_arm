#!/usr/bin/env python3
"""
Interaktiver MuJoCo Viewer - Panda + Fließband
Zeigt die Szene in einem MuJoCo Viewer mit interaktiven Kontrollen
"""

import numpy as np
import mujoco as mj
from pathlib import Path
import os
import sys


def launch_interactive_viewer():
    """Starte interaktiven MuJoCo Viewer mit der Panda+Fließband Szene"""
    
    print("="*70)
    print("INTERAKTIVER MUJOCO VIEWER - Franka Panda + Conveyor")
    print("="*70)
    
    # Szene laden
    scene_file = Path(__file__).parent / "conveyor_scene.xml"
    
    if not scene_file.exists():
        print(f"❌ Szene nicht gefunden: {scene_file}")
        return False
    
    print(f"\n📂 Lade Szene: {scene_file.name}")
    
    try:
        # Model und Data laden
        model = mj.MjModel.from_xml_path(str(scene_file))
        data = mj.MjData(model)
        
        print("✅ Szene geladen")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.njnt}")
        print(f"   Actuators: {model.nu}")
        print(f"   Geoms: {model.ngeom}")
        
        # Versuche Viewer zu starten
        print("\n🎮 Starte interaktiven Viewer...")
        print("   Steuerung:")
        print("   - Maus: Drehen/Zoomen")
        print("   - RECHTS-Klick + Drag: Verschieben")
        print("   - Scroll: Zoomen")
        print("   - 'ESC': Schließen")
        print("   - Space: Pause/Play")
        print("\n")
        
        # Launcher Funktion prüfen
        if hasattr(mj, 'Viewer'):
            # MuJoCo 2.x+ mit Viewer Klasse
            with mj.Viewer(model) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Simulationsloop
                while viewer.is_running():
                    # Kontrolliere Motoren mit sinusförmigen Bewegungen
                    for i in range(model.nu):
                        freq = 1.0 + i * 0.2
                        amplitude = 0.5
                        t = data.time
                        data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
                    
                    # Schritt simulieren
                    mj.mj_step(model, data)
                    viewer.sync()
                
                print("\n✅ Viewer geschlossen")
                return True
        
        elif hasattr(mj, 'viewer') and hasattr(mj.viewer, 'launch_passive'):
            # MuJoCo 2.x mit launch_passive
            with mj.viewer.launch_passive(model) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Simulationsloop
                while viewer.is_running():
                    # Kontrolliere Motoren
                    for i in range(model.nu):
                        freq = 1.0 + i * 0.2
                        amplitude = 0.5
                        t = data.time
                        data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
                    
                    # Schritt simulieren
                    mj.mj_step(model, data)
                    viewer.sync()
                
                print("\n✅ Viewer geschlossen")
                return True
        
        else:
            # Fallback: Renderer für Bilder + HTML Viewer
            print("\n⚠️  MuJoCo Viewer API nicht verfügbar")
            print("   Alternative: Erstelle HTML Viewer mit interaktiver Szene-Inspektion")
            
            create_interactive_html_scene_viewer(model, data)
            return True
            
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_interactive_html_scene_viewer(model, data):
    """Erstelle interaktiven HTML Viewer mit Szene-Informationen"""
    
    print("\n📄 Erstelle HTML Viewer...")
    
    # Sammle Szene-Informationen
    bodies = []
    for i in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        body_pos = data.xpos[i]
        body_quat = data.xquat[i]
        bodies.append({
            'id': i,
            'name': body_name,
            'pos': [float(x) for x in body_pos],
            'quat': [float(x) for x in body_quat],
        })
    
    joints = []
    for i in range(model.njnt):
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        joint_addr = model.jnt_qposadr[i]
        qpos = data.qpos[joint_addr] if joint_addr < len(data.qpos) else 0
        joints.append({
            'id': i,
            'name': joint_name,
            'qpos': float(qpos),
            'addr': int(joint_addr),
        })
    
    actuators = []
    for i in range(model.nu):
        actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl = data.ctrl[i] if i < len(data.ctrl) else 0
        actuators.append({
            'id': i,
            'name': actuator_name,
            'ctrl': float(ctrl),
        })
    
    geometries = []
    for i in range(model.ngeom):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        geom_type = model.geom_type[i]
        type_names = ['plane', 'sphere', 'capsule', 'box', 'mesh', 'hfield', 'cylinder', 'ellipsoid']
        type_name = type_names[geom_type] if geom_type < len(type_names) else 'unknown'
        
        geometries.append({
            'id': i,
            'name': geom_name,
            'type': type_name,
            'size': [float(x) for x in model.geom_size[i]],
        })
    
    # HTML generieren
    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MuJoCo Szene Viewer - Franka Panda + Conveyor</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        h1 {{
            color: #1e3c72;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .status {{
            background: #4caf50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .sections {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #1e3c72;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 3px solid #2a5298;
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            color: #2a5298;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        
        th {{
            background: #f0f0f0;
            color: #1e3c72;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #2a5298;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .code {{
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
            padding: 8px 12px;
            border-radius: 4px;
            color: #c7254e;
            word-break: break-all;
        }}
        
        .value {{
            color: #2a5298;
            font-weight: 500;
        }}
        
        .spec-item {{
            padding: 10px;
            background: #f8f9fa;
            border-left: 4px solid #2a5298;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
        
        .spec-label {{
            font-weight: 600;
            color: #1e3c72;
            margin-bottom: 3px;
        }}
        
        .spec-value {{
            color: #666;
        }}
        
        footer {{
            text-align: center;
            color: white;
            padding: 20px;
            margin-top: 40px;
        }}
        
        .info-banner {{
            background: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 MuJoCo Scene Viewer</h1>
            <p class="subtitle">Franka Emika Panda 7-DOF + Conveyor Belt</p>
        </header>
        
        <div class="status">
            ✅ Szene geladen | {model.nbody} Bodies | {model.njnt} Joints | {model.nu} Actuators | {model.ngeom} Geometries
        </div>
        
        <div class="info-banner">
            <strong>💡 Hinweis:</strong> Diese interaktive Seite zeigt die aktuelle Struktur und Zustand der Szene. 
            Für echte 3D-Visualisierung mit Mausbedienung siehe die Sektion "Start Viewer" unten.
        </div>
        
        <div class="sections">
            <!-- SPEZIFIKATIONEN -->
            <div class="section">
                <h2>📊 Szene-Spezifikationen</h2>
                <div class="spec-item">
                    <div class="spec-label">Robot Model</div>
                    <div class="spec-value">Franka Emika Panda (7 DOF)</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Physics Engine</div>
                    <div class="spec-value">MuJoCo {mj.__version__}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Timestep</div>
                    <div class="spec-value">{model.opt.timestep * 1000:.1f} ms</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Gravity</div>
                    <div class="spec-value">[{model.opt.gravity[0]:.2f}, {model.opt.gravity[1]:.2f}, {model.opt.gravity[2]:.2f}] m/s²</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Scene Time</div>
                    <div class="spec-value" id="scene-time">{data.time:.3f} s</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Conveyor Mesh</div>
                    <div class="spec-value">986 vertices, 48 KB STL</div>
                </div>
            </div>
            
            <!-- BODIES -->
            <div class="section">
                <h2>🏗️ Bodies ({len(bodies)})</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Position (X, Y, Z)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''
                        <tr>
                            <td>{body['id']}</td>
                            <td><span class="code">{body['name']}</span></td>
                            <td class="value">[{body['pos'][0]:.3f}, {body['pos'][1]:.3f}, {body['pos'][2]:.3f}]</td>
                        </tr>
                        ''' for body in bodies)}
                    </tbody>
                </table>
            </div>
            
            <!-- JOINTS -->
            <div class="section">
                <h2>🔗 Joints ({len(joints)})</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Position (rad)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''
                        <tr>
                            <td>{joint['id']}</td>
                            <td><span class="code">{joint['name']}</span></td>
                            <td class="value">{joint['qpos']:.4f}</td>
                        </tr>
                        ''' for joint in joints)}
                    </tbody>
                </table>
            </div>
            
            <!-- ACTUATORS -->
            <div class="section">
                <h2>⚡ Actuators ({len(actuators)})</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Control Signal</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''
                        <tr>
                            <td>{act['id']}</td>
                            <td><span class="code">{act['name']}</span></td>
                            <td class="value">{act['ctrl']:.4f} V</td>
                        </tr>
                        ''' for act in actuators)}
                    </tbody>
                </table>
            </div>
            
            <!-- GEOMETRIES -->
            <div class="section">
                <h2>🎯 Geometries ({len(geometries)})</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Size</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''
                        <tr>
                            <td>{geom['id']}</td>
                            <td><span class="code">{geom['name']}</span></td>
                            <td>{geom['type'].upper()}</td>
                            <td class="value">[{', '.join(f'{s:.3f}' for s in geom['size'])}]</td>
                        </tr>
                        ''' for geom in geometries)}
                    </tbody>
                </table>
            </div>
            
            <!-- VIEWER STARTEN -->
            <div class="section">
                <h2>🎮 3D Viewer Starten</h2>
                <p style="margin-bottom: 15px; color: #666;">
                    Um die Szene in 3D mit Mausbedienung zu sehen, führe diesen Befehl aus:
                </p>
                <div style="background: #f5f5f5; padding: 12px; border-radius: 5px; margin-bottom: 15px; font-family: monospace; word-break: break-all;">
                    <strong>python3 interactive_viewer.py</strong>
                </div>
                <div style="background: #e3f2fd; padding: 12px; border-radius: 5px; color: #1565c0;">
                    <strong>⌨️ Steuerung:</strong><br>
                    • Linker Drag: Drehen<br>
                    • Rechter Drag: Verschieben<br>
                    • Scroll: Zoomen<br>
                    • Space: Play/Pause<br>
                    • ESC: Schließen
                </div>
            </div>
        </div>
        
        <footer>
            <p>Generated with MuJoCo Physics Simulation Engine</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Szene: <strong>conveyor_scene.xml</strong> | 
                Modell: <strong>Franka Panda</strong> | 
                Fließband: <strong>STL Mesh</strong>
            </p>
        </footer>
    </div>
    
    <script>
        // Update scene time alle 2 Sekunden
        setInterval(function() {{
            // In echter Implementierung würde WebSocket/API verwendet
            // Für jetzt nur statisch anzeigen
        }}, 2000);
    </script>
</body>
</html>
"""
    
    # HTML speichern
    output_file = Path(__file__).parent / "viewer.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML Viewer erstellt: {output_file}")
    
    # Browser öffnen
    import webbrowser
    try:
        webbrowser.open(f'file://{output_file}')
        print(f"✅ Browser geöffnet: {output_file}")
    except Exception as e:
        print(f"⚠️  Browser konnte nicht geöffnet werden: {e}")
        print(f"   Öffne manuell: {output_file}")
    
    return output_file


def main():
    try:
        success = launch_interactive_viewer()
        if not success:
            print("\n⚠️  Viewer konnte nicht gestartet werden")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✅ Viewer beendet")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
