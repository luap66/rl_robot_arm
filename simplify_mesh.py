"""
Simplify and optimize the conveyor belt mesh for MuJoCo
"""

import trimesh
from pathlib import Path

def simplify_mesh(stl_file: str, output_file: str = None, target_reduction: float = 0.95) -> str:
    """
    Simplify a mesh to reduce polygon count.
    
    Args:
        stl_file: Path to the STL file
        output_file: Path for the output simplified STL file
        target_reduction: Target polygon reduction ratio (0.95 = reduce to 5% of original)
        
    Returns:
        Path to the simplified STL file
    """
    if output_file is None:
        path = Path(stl_file)
        output_file = str(path.parent / f"{path.stem}_simplified.stl")
    
    print(f"Loading mesh: {stl_file}")
    mesh = trimesh.load(stl_file)
    
    original_faces = len(mesh.faces)
    print(f"  Original faces: {original_faces}")
    print(f"  Original vertices: {len(mesh.vertices)}")
    
    # Simplify the mesh using voxel reduction
    print(f"  Simplifying...")
    try:
        mesh_simplified = mesh.voxelized(pitch=0.05).as_mesh()
    except:
        # If voxelization fails, use convex hull instead
        print("  Voxelization failed, using convex hull...")
        mesh_simplified = mesh.convex_hull
    
    print(f"  Simplified faces: {len(mesh_simplified.faces)}")
    print(f"  Simplified vertices: {len(mesh_simplified.vertices)}")
    print(f"  Reduction: {(1 - len(mesh_simplified.faces)/original_faces)*100:.1f}%")
    
    # Export
    mesh_simplified.export(output_file, file_type='stl_ascii')
    print(f"✓ Saved to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
    
    return output_file


def create_simplified_conveyor_box(output_file: str):
    """
    Create a simplified conveyor belt as a box for collision physics.
    This is much simpler and more suitable for simulation.
    """
    # Create a simple box representing the conveyor belt platform
    # Dimensions from the filename: 2000x400x700mm = 2.0x0.4x0.7m
    # But these seem to be in mm, so convert to reasonable simulation scale
    
    # Create a long rectangular platform
    conveyor_width = 0.4  # 400mm = 0.4m
    conveyor_length = 2.0  # 2000mm = 2.0m
    conveyor_height = 0.05  # 50mm = 0.05m (belt thickness)
    
    mesh = trimesh.creation.box(
        extents=[conveyor_length, conveyor_width, conveyor_height]
    )
    
    mesh.export(output_file, file_type='stl_ascii')
    print(f"✓ Created simplified conveyor box: {output_file}")
    
    return output_file


if __name__ == "__main__":
    stl_file = "/home/paulw/projects/rl_robot_arm/assets/conveyor_belt.stl"
    
    # Create a very simplified version for actual use
    print("=== Creating simplified collision mesh ===\n")
    simplified_file = "/home/paulw/projects/rl_robot_arm/assets/conveyor_belt_simplified.stl"
    simplify_mesh(stl_file, simplified_file, target_reduction=0.98)
    
    # Also create a basic box version for faster simulation
    print("\n=== Creating basic conveyor box mesh ===\n")
    box_file = "/home/paulw/projects/rl_robot_arm/assets/conveyor_belt_box.stl"
    create_simplified_conveyor_box(box_file)
