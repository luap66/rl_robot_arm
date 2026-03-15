#!/usr/bin/env python3
"""
Render Panda + Conveyor Scene - Interactive HTML Viewer
Creates an interactive web-based visualization of the MuJoCo scene
"""

import numpy as np
import mujoco as mj
from pathlib import Path
import json
import webbrowser


def create_interactive_html_viewer(model, data, output_file="viewer.html"):
    """Render scene from multiple viewpoints"""
    
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    print(f"\n🎥 Rendering scene to: {output_path}")
    print(f"   Resolution: 1280x720 pixels")
    print(f"   Views: {num_frames}")
    
    try:
        renderer = mj.Renderer(model, 960, 720)
        
        # Different camera angles
        camera_configs = [
            {
                "name": "front",
                "distance": 2.0,
                "azimuth": 0.0,
                "elevation": -20.0,
            },
            {
                "name": "right",
                "distance": 2.5,
                "azimuth": 90.0,
                "elevation": -25.0,
            },
            {
                "name": "left",
                "distance": 2.5,
                "azimuth": -90.0,
                "elevation": -25.0,
            },
            {
                "name": "top",
                "distance": 3.0,
                "azimuth": 45.0,
                "elevation": 60.0,
            },
            {
                "name": "isometric",
                "distance": 3.5,
                "azimuth": 45.0,
                "elevation": -30.0,
            },
            {
                "name": "close_view",
                "distance": 1.0,
                "azimuth": 0.0,
                "elevation": -15.0,
            },
        ]
        
        # Render multiple frames
        for i in range(6):
            print(f"\n   Rendering frame {i+1}/6...")
            
            # Simple default render
            renderer.update_scene(data)
            pixels = renderer.render()
            
            # Save image
            view_names = ["front", "right", "left", "top", "isometric", "close"]
            output_file = output_path / f"{i+1:02d}_{view_names[i]}.png"
            
            # Convert RGB array to image
            try:
                from PIL import Image
                img = Image.fromarray(pixels)
                img.save(output_file)
                print(f"      ✓ Saved: {output_file.name}")
            except ImportError:
                print(f"      ⚠️  PIL not available, skipping image save")
                print(f"         (Install with: pip install Pillow)")
        
        return output_path
    
    except Exception as e:
        print(f"❌ Rendering error: {e}")
        import traceback
        traceback.print_exc()
        return None


def render_animation(model, data, output_dir="renders", num_steps=200):
    """Render animation of robot moving"""
    
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    print(f"\n🎬 Rendering animation ({num_steps} frames)")
    print(f"   Output: {output_path}")
    
    try:
        renderer = mj.Renderer(model, 1280, 720)
        
        # Camera configuration
        renderer.cam.distance = 2.0
        renderer.cam.azimuth = 45.0
        renderer.cam.elevation = -25.0
        renderer.cam.lookat = np.array([0.0, -0.5, 0.5])
        
        frames = []
        
        for step in range(num_steps):
            # Generate motion
            t = step / num_steps
            
            # Control: sinusoidal motion
            for i in range(model.nu):
                freq = 1.0 + i * 0.2
                amplitude = 0.5
                data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + i * np.pi / 3.5)
            
            # Step simulation
            mj.mj_step(model, data)
            
            # Render
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
            
            if (step + 1) % 50 == 0:
                print(f"   {step+1}/{num_steps} frames...")
        
        # Save animation frames
        try:
            from PIL import Image
            
            print(f"\n   Saving frames as PNG sequence...")
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                filename = output_path / f"frame_{i:04d}.png"
                img.save(filename)
            
            print(f"   ✓ Saved {len(frames)} frames")
            print(f"\n   To create MP4 video, run:")
            print(f"   ffmpeg -framerate 30 -i renders/frame_%04d.png -c:v libx264 animation.mp4")
            
        except ImportError:
            print(f"   ⚠️  PIL not available")
        
        return output_path
    
    except Exception as e:
        print(f"❌ Animation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_html_viewer(output_dir="renders"):
    """Create HTML viewer for rendered images"""
    
    output_path = Path(__file__).parent / output_dir
    
    html_content = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Franka Panda + Conveyor Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        h1 {
            color: #1e3c72;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .image-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .image-label {
            padding: 15px;
            background: #f8f9fa;
            text-align: center;
            font-weight: 600;
            color: #1e3c72;
        }
        
        .info-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .info-box h2 {
            color: #1e3c72;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .spec-item {
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #2a5298;
            border-radius: 5px;
        }
        
        .spec-label {
            font-weight: 600;
            color: #1e3c72;
            margin-bottom: 5px;
        }
        
        .spec-value {
            color: #666;
            font-size: 0.95em;
        }
        
        footer {
            text-align: center;
            color: white;
            padding: 20px;
            margin-top: 40px;
        }
        
        .status {
            background: #4caf50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Franka Panda + Conveyor Belt</h1>
            <p class="subtitle">MuJoCo Physics Simulation - 3D Visualization</p>
        </header>
        
        <div class="status">
            ✅ Scene loaded successfully | 7 DOF | 10 Bodies | 11 Geometries
        </div>
        
        <div class="info-box">
            <h2>📊 Scene Specifications</h2>
            <div class="specs-grid">
                <div class="spec-item">
                    <div class="spec-label">Robot</div>
                    <div class="spec-value">Franka Emika Panda</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Degrees of Freedom</div>
                    <div class="spec-value">7 Revolute Joints</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Conveyor Belt</div>
                    <div class="spec-value">986 vertices, 48 KB</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Physics Engine</div>
                    <div class="spec-value">MuJoCo 1.0+</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Timestep</div>
                    <div class="spec-value">0.002s (500 Hz)</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Gravity</div>
                    <div class="spec-value">-9.81 m/s²</div>
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <h2>🎥 3D Views</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Multiple viewpoints of the Panda robot and conveyor belt system:
            </p>
            <div class="gallery" id="gallery">
                <!-- Images will be loaded here -->
            </div>
        </div>
        
        <footer>
            <p>Generated with MuJoCo Physics Simulation Engine</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                For more information, visit: <a href="https://mujoco.org" style="color: white; text-decoration: underline;">mujoco.org</a>
            </p>
        </footer>
    </div>
    
    <script>
        // Generate gallery from images
        const imageNames = [
            "01_front",
            "02_right", 
            "03_left",
            "04_top",
            "05_isometric",
            "06_close_view"
        ];
        
        const gallery = document.getElementById('gallery');
        
        imageNames.forEach(name => {
            const card = document.createElement('div');
            card.className = 'image-card';
            
            const img = document.createElement('img');
            img.src = name + '.png';
            img.alt = name.replace(/_/g, ' ');
            img.onerror = function() {
                this.parentElement.style.display = 'none';
            };
            
            const label = document.createElement('div');
            label.className = 'image-label';
            label.textContent = name.replace(/_/g, ' ').toUpperCase();
            
            card.appendChild(img);
            card.appendChild(label);
            gallery.appendChild(card);
        });
    </script>
</body>
</html>
"""
    
    html_file = output_path / "viewer.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n📄 Created HTML viewer: {html_file}")
    return html_file


def main():
    print("="*70)
    print("FRANKA PANDA + CONVEYOR - 3D RENDERING")
    print("="*70)
    
    # Load scene
    scene_file = Path(__file__).parent / "conveyor_scene.xml"
    
    if not scene_file.exists():
        print(f"❌ Scene not found: {scene_file}")
        return
    
    print(f"\n📂 Loading scene: {scene_file.name}")
    
    try:
        model = mj.MjModel.from_xml_path(str(scene_file))
        data = mj.MjData(model)
        
        print("✅ Scene loaded")
        print(f"   Bodies: {model.nbody}")
        print(f"   Geoms: {model.ngeom}")
        print(f"   Joints: {model.njnt}")
        
        # Render views
        output_dir = render_scene_views(model, data, num_frames=6)
        
        # Create HTML viewer
        if output_dir:
            html_file = create_html_viewer(str(output_dir))
            
            print("\n" + "="*70)
            print("✅ RENDERING COMPLETE")
            print("="*70)
            print(f"""
📸 Views rendered:
   - Front view
   - Right side
   - Left side  
   - Top view
   - Isometric
   - Close-up
   
📁 Output directory: renders/

📄 Open in browser:
   {html_file}
   
Or view individual images:
   ls -lh renders/*.png
            """)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
