#!/usr/bin/env python3
"""
Convert .obj files to .usd files with physics properties.
This creates production-ready USD assets that can be directly spawned in Isaac Sim.

Usage:
    # Single file
    ./isaaclab.sh -p classifier/isaac/obj_2_usd.py --input object.obj --output object.usd --headless

    # Batch conversion of the EGAD meshes for the data-collection env (default object dir data/egad_usd)
    ./isaaclab.sh -p classifier/isaac/obj_2_usd.py --input-folder data/egad_mesh \
        --output-folder data/egad_usd --center --headless

--center moves each mesh so its vertex centroid is at the prim origin. The USDs used to
collect the released classifier data were centred this way (scale 0.0006, convex hull).
"""

import argparse

from isaaclab.app import AppLauncher

# Create parser
parser = argparse.ArgumentParser(
    description="Convert OBJ to USD with physics properties",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Single file mode
parser.add_argument(
    "--input", "-i",
    type=str,
    default=None,
    help="Input .obj file path (for single file conversion)"
)

parser.add_argument(
    "--output", "-o",
    type=str,
    default=None,
    help="Output .usd file path (for single file conversion)"
)

# Batch mode
parser.add_argument(
    "--input-folder",
    type=str,
    default=None,
    help="Input folder containing .obj files (for batch conversion)"
)

parser.add_argument(
    "--output-folder",
    type=str,
    default=None,
    help="Output folder for .usd files (for batch conversion)"
)

parser.add_argument(
    "--scale", "-s",
    type=float,
    default=0.0006,  # 10x smaller again (was 0.01, now 0.001)
    help="Scale factor (e.g., 0.001 for objects, 0.0001 for very small objects)"
)

parser.add_argument(
    "--mass", "-m",
    type=float,
    default=0.5,
    help="Mass in kilograms"
)

parser.add_argument(
    "--static-friction",
    type=float,
    default=0.7,
    help="Static friction coefficient"
)

parser.add_argument(
    "--dynamic-friction",
    type=float,
    default=0.6,
    help="Dynamic friction coefficient"
)

parser.add_argument(
    "--restitution",
    type=float,
    default=0.1,
    help="Restitution (bounciness) coefficient"
)

parser.add_argument(
    "--collision",
    type=str,
    default="convexHull",
    choices=["convexHull", "convexDecomposition", "meshSimplification", "none"],
    help="Collision mesh approximation method"
)

parser.add_argument(
    "--center",
    action="store_true",
    help="Translate the mesh so its vertex centroid is at the origin (used for the EGAD objects "
         "of the data-collection env)"
)

parser.add_argument(
    "--color",
    type=float,
    nargs=3,
    default=None,  # Will be randomized if not specified
    metavar=("R", "G", "B"),
    help="RGB color values (0-1 range). If not specified, color will be randomized."
)

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim starts."""

import os
import random
from pathlib import Path
import glob

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
import trimesh


def convert_obj_to_usd_with_physics(
    obj_path: str,
    usd_path: str,
    scale: float = 1.0,
    mass: float = 0.1,
    static_friction: float = 0.7,
    dynamic_friction: float = 0.6,
    restitution: float = 0.1,
    collision_approximation: str = "convexHull",
    color: tuple = (0.8, 0.6, 0.4),
    center: bool = False,
):
    """
    Convert OBJ file to USD with full physics properties.
    
    Args:
        obj_path: Path to input .obj file
        usd_path: Path to output .usd file
        scale: Uniform scale factor (to convert units to meters)
        mass: Mass in kilograms
        static_friction: Static friction coefficient
        dynamic_friction: Dynamic friction coefficient
        restitution: Restitution (bounciness) coefficient
        collision_approximation: Collision mesh approximation ("convexHull", "convexDecomposition", "meshSimplification", "none")
        color: RGB color tuple (0-1 range)
    """
    
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    print(f"\n{'='*80}")
    print(f"Converting OBJ to USD with Physics")
    print(f"{'='*80}")
    print(f"Input:  {obj_path}")
    print(f"Output: {usd_path}")
    print(f"Scale:  {scale}")
    print(f"Mass:   {mass} kg")
    print(f"{'='*80}\n")
    
    # Load mesh using trimesh
    print("[1/6] Loading mesh...")
    mesh = trimesh.load(obj_path, force='mesh')
    print(f"      - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    if center:
        centroid = mesh.vertices.mean(axis=0)
        mesh.vertices = mesh.vertices - centroid
        print(f"      - Centred mesh (moved by {-centroid})")
    
    # Create USD stage
    print("[2/6] Creating USD stage...")
    stage = Usd.Stage.CreateNew(usd_path)
    
    # Set up/forward axis (standard in USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Create root Xform
    root_path = "/Object"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    print(f"      - Root prim: {root_path}")
    
    # Apply scale if needed
    if scale != 1.0:
        xformable = UsdGeom.Xformable(root)
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(scale, scale, scale))
        print(f"      - Applied scale: {scale}")
    
    # Create mesh geometry
    print("[3/6] Creating mesh geometry...")
    mesh_path = f"{root_path}/Mesh"
    mesh_geom = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set vertices
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
    mesh_geom.GetPointsAttr().Set(points)
    
    # Set faces
    face_indices = mesh.faces.flatten().tolist()
    mesh_geom.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh_geom.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    
    # Set color
    colors = [Gf.Vec3f(color[0], color[1], color[2]) for _ in mesh.vertices]
    mesh_geom.GetDisplayColorAttr().Set(colors)
    
    # Compute normals if available
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        normals = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh.vertex_normals]
        mesh_geom.GetNormalsAttr().Set(normals)
    
    print(f"      - Mesh geometry created")
    
    # Add rigid body physics
    print("[4/6] Adding rigid body physics...")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    print(f"      - Rigid body enabled")
    
    # Set mass
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(mass)
    print(f"      - Mass: {mass} kg")
    
    # Add collision
    print("[5/6] Adding collision...")
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_geom.GetPrim())
    
    # Set collision approximation
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_geom.GetPrim())
    mesh_collision_api.CreateApproximationAttr(collision_approximation)
    print(f"      - Collision approximation: {collision_approximation}")
    
    # Add physics material
    print("[6/6] Adding physics material...")
    material_path = f"{root_path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    
    physics_material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material_api.CreateStaticFrictionAttr(static_friction)
    physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material_api.CreateRestitutionAttr(restitution)
    
    # Bind material to mesh
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_geom.GetPrim())
    binding_api.Bind(material)
    
    print(f"      - Static friction: {static_friction}")
    print(f"      - Dynamic friction: {dynamic_friction}")
    print(f"      - Restitution: {restitution}")
    
    # Save the stage
    stage.Save()
    
    print(f"\n{'='*80}")
    print(f"SUCCESS: USD file created with physics properties")
    print(f"{'='*80}")
    print(f"File: {usd_path}")
    print(f"Size: {os.path.getsize(usd_path) / 1024:.2f} KB")
    print(f"\nYou can now spawn this USD file directly in Isaac Sim!")
    print(f"{'='*80}\n")


def main():
    # Determine mode: single file or batch
    single_mode = args_cli.input is not None and args_cli.output is not None
    batch_mode = args_cli.input_folder is not None and args_cli.output_folder is not None
    
    if not single_mode and not batch_mode:
        print("ERROR: Must specify either:")
        print("  1. --input and --output for single file conversion")
        print("  2. --input-folder and --output-folder for batch conversion")
        simulation_app.close()
        return
    
    if single_mode and batch_mode:
        print("ERROR: Cannot use both single and batch mode together")
        simulation_app.close()
        return
    
    # Single file mode
    if single_mode:
        if not os.path.exists(args_cli.input):
            print(f"ERROR: Input file not found: {args_cli.input}")
            simulation_app.close()
            return
        
        # Create output directory if needed
        output_dir = os.path.dirname(args_cli.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Randomize color if not specified
        if args_cli.color is None:
            color = (random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0))
            print(f"[INFO] Using random color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        else:
            color = tuple(args_cli.color)
        
        # Convert single file
        try:
            convert_obj_to_usd_with_physics(
                obj_path=args_cli.input,
                usd_path=args_cli.output,
                scale=args_cli.scale,
                mass=args_cli.mass,
                static_friction=args_cli.static_friction,
                dynamic_friction=args_cli.dynamic_friction,
                restitution=args_cli.restitution,
                collision_approximation=args_cli.collision,
                color=color,
                center=args_cli.center,
            )
        except Exception as e:
            print(f"\nERROR: Conversion failed")
            print(f"       {str(e)}\n")
            raise
    
    # Batch mode
    else:
        if not os.path.exists(args_cli.input_folder):
            print(f"ERROR: Input folder not found: {args_cli.input_folder}")
            simulation_app.close()
            return
        
        # Create output folder if needed
        if not os.path.exists(args_cli.output_folder):
            os.makedirs(args_cli.output_folder)
            print(f"Created output folder: {args_cli.output_folder}")
        
        # Find all .obj files
        obj_files = glob.glob(os.path.join(args_cli.input_folder, "*.obj"))
        
        if not obj_files:
            print(f"ERROR: No .obj files found in {args_cli.input_folder}")
            simulation_app.close()
            return
        
        print(f"\n{'='*80}")
        print(f"BATCH CONVERSION MODE")
        print(f"{'='*80}")
        print(f"Input folder:  {args_cli.input_folder}")
        print(f"Output folder: {args_cli.output_folder}")
        print(f"Found {len(obj_files)} .obj file(s)")
        print(f"{'='*80}\n")
        
        # Convert each file
        successful = 0
        failed = 0
        
        for idx, obj_path in enumerate(obj_files, 1):
            obj_filename = os.path.basename(obj_path)
            usd_filename = os.path.splitext(obj_filename)[0] + ".usd"
            usd_path = os.path.join(args_cli.output_folder, usd_filename)
            
            # Randomize color for each object if not specified
            if args_cli.color is None:
                color = (random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0))
            else:
                color = tuple(args_cli.color)
            
            print(f"\n[{idx}/{len(obj_files)}] Converting: {obj_filename}")
            print(f"           Color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
            
            try:
                convert_obj_to_usd_with_physics(
                    obj_path=obj_path,
                    usd_path=usd_path,
                    scale=args_cli.scale,
                    mass=args_cli.mass,
                    static_friction=args_cli.static_friction,
                    dynamic_friction=args_cli.dynamic_friction,
                    restitution=args_cli.restitution,
                    collision_approximation=args_cli.collision,
                    color=color,
                    center=args_cli.center,
                )
                successful += 1
            except Exception as e:
                print(f"           ERROR: {str(e)}")
                failed += 1
                continue
        
        # Summary
        print(f"\n{'='*80}")
        print(f"BATCH CONVERSION COMPLETE")
        print(f"{'='*80}")
        print(f"Total files:  {len(obj_files)}")
        print(f"Successful:   {successful}")
        print(f"Failed:       {failed}")
        print(f"{'='*80}\n")
    
    # Close the app
    simulation_app.close()


if __name__ == "__main__":
    main()