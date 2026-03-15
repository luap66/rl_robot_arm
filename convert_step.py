"""
Convert STEP file to STL for MuJoCo compatibility
"""

import trimesh
from pathlib import Path

def convert_step_to_stl(step_file: str, output_file: str = None) -> str:
    """
    Convert a STEP file to STL format.
    
    Args:
        step_file: Path to the STEP file
        output_file: Path for the output STL file. If None, uses same name with .stl extension
        
    Returns:
        Path to the converted STL file
    """
    step_path = Path(step_file)
    
    if output_file is None:
        output_file = str(step_path.with_suffix('.stl'))
    
    print(f"Loading STEP file: {step_file}")
    
    try:
        # Load the STEP file (trimesh loads it as a Scene object for complex files)
        scene = trimesh.load(step_file)
        
        # If it's a scene with multiple meshes, merge them
        if isinstance(scene, trimesh.Scene):
            print(f"  Found Scene with {len(scene.geometry)} geometries")
            meshes = []
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            
            if meshes:
                # Concatenate all meshes
                mesh = trimesh.util.concatenate(meshes)
            else:
                print("No valid meshes found in scene")
                return None
        else:
            mesh = scene
        
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        
        # Export as STL (binary format for smaller file size)
        mesh.export(output_file, file_type='stl_ascii')
        print(f"✓ Converted to: {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"Error converting file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Convert the conveyor belt STEP file
    step_file = "/home/paulw/projects/rl_robot_arm/assets/Belt Conveyor 2000x400x700.STEP"
    output_file = "/home/paulw/projects/rl_robot_arm/assets/conveyor_belt.stl"
    
    result = convert_step_to_stl(step_file, output_file)
    if result:
        print("\n✓ Conversion successful!")

