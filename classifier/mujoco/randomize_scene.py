import os
import csv
import random
import numpy as np
import itertools
import xml.etree.ElementTree as ET
import mujoco
import glfw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mujoco import mjv_defaultOption
import io
import pandas as pd
from PIL import Image


# Directory containing this script; default inputs are resolved relative to it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository root (classifier/mujoco/ -> repo) and git-ignored data directory.
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _absolutize_compiler_dirs(root, base_xml_path):
    """Make <compiler meshdir/texturedir> absolute so the generated XML can be
    written to any directory (e.g. data/) and still find the robot meshes and
    textures shipped next to the base XML."""
    compiler = root.find("compiler")
    if compiler is None:
        return
    base_dir = os.path.dirname(os.path.abspath(base_xml_path))
    for attr in ("meshdir", "texturedir"):
        val = compiler.get(attr)
        if val is not None and not os.path.isabs(val):
            compiler.set(attr, os.path.abspath(os.path.join(base_dir, val)))


class DensityPattern:
    """Define different spatial density patterns for scene generation"""
    
    @staticmethod
    def clustered_center(n_objects, workspace_limits):
        """
        Dense cluster in center, sparse elsewhere.
        Good for testing isolated vs cluttered targets.
        """
        positions = []
        center_x = (workspace_limits["x"][0] + workspace_limits["x"][1]) / 2
        center_y = (workspace_limits["y"][0] + workspace_limits["y"][1]) / 2
        
        # 60% of objects in dense center cluster
        n_center = int(n_objects * 0.6)
        for i in range(n_center):
            # Dense packing in center (0.03-0.08m spacing)
            angle = random.uniform(0, 2*np.pi)
            radius = random.uniform(0.03, 0.15)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            positions.append([x, y, z])
        
        # 40% scattered around edges (0.15-0.40m spacing)
        for i in range(n_objects - n_center):
            angle = random.uniform(0, 2*np.pi)
            radius = random.uniform(0.2, 0.4)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            
            # Clamp to workspace
            x = np.clip(x, workspace_limits["x"][0], workspace_limits["x"][1])
            y = np.clip(y, workspace_limits["y"][0], workspace_limits["y"][1])
            positions.append([x, y, z])
        
        return positions
    
    @staticmethod
    def two_clusters(n_objects, workspace_limits):
        """
        Two dense clusters on opposite sides.
        Target can be in cluster, between them, or isolated.
        """
        positions = []
        center_x = (workspace_limits["x"][0] + workspace_limits["x"][1]) / 2
        center_y = (workspace_limits["y"][0] + workspace_limits["y"][1]) / 2
        
        # Cluster 1: left side
        n_cluster1 = n_objects // 2
        cluster1_center = [center_x - 0.2, center_y]
        for i in range(n_cluster1):
            angle = random.uniform(0, 2*np.pi)
            radius = random.uniform(0.02, 0.12)
            x = cluster1_center[0] + radius * np.cos(angle)
            y = cluster1_center[1] + radius * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            positions.append([x, y, z])
        
        # Cluster 2: right side
        cluster2_center = [center_x + 0.2, center_y]
        for i in range(n_objects - n_cluster1):
            angle = random.uniform(0, 2*np.pi)
            radius = random.uniform(0.02, 0.12)
            x = cluster2_center[0] + radius * np.cos(angle)
            y = cluster2_center[1] + radius * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            positions.append([x, y, z])
        
        return positions
    
    @staticmethod
    def uniform_random(n_objects, workspace_limits):
        """
        Uniformly random distribution.
        Variable local density by chance.
        """
        positions = []
        for i in range(n_objects):
            x = random.uniform(*workspace_limits["x"])
            y = random.uniform(*workspace_limits["y"])
            z = random.uniform(*workspace_limits["z"])
            positions.append([x, y, z])
        return positions
    
    @staticmethod
    def ring_pattern(n_objects, workspace_limits):
        """
        Objects in a ring around center.
        Center is empty - target can be in center (isolated) or in ring (cluttered).
        """
        positions = []
        center_x = (workspace_limits["x"][0] + workspace_limits["x"][1]) / 2
        center_y = (workspace_limits["y"][0] + workspace_limits["y"][1]) / 2
        
        for i in range(n_objects):
            angle = random.uniform(0, 2*np.pi)
            # Ring at radius 0.15-0.25m
            radius = random.uniform(0.15, 0.25)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            
            # Add some noise to make it less perfect
            x += random.uniform(-0.05, 0.05)
            y += random.uniform(-0.05, 0.05)
            
            # Clamp to workspace
            x = np.clip(x, workspace_limits["x"][0], workspace_limits["x"][1])
            y = np.clip(y, workspace_limits["y"][0], workspace_limits["y"][1])
            positions.append([x, y, z])
        
        return positions
    
    @staticmethod
    def gradient_density(n_objects, workspace_limits):
        """
        Density increases from left to right.
        Good for testing edge effects.
        """
        positions = []
        x_min, x_max = workspace_limits["x"]
        
        for i in range(n_objects):
            # Sample x uniformly
            x = random.uniform(x_min, x_max)
            
            # Density increases with x
            # Left side: sparse (0.15m spacing)
            # Right side: dense (0.03m spacing)
            density_factor = (x - x_min) / (x_max - x_min)
            y_range = 0.4 - 0.35 * density_factor  # Smaller range = higher density
            
            center_y = (workspace_limits["y"][0] + workspace_limits["y"][1]) / 2
            y = random.uniform(center_y - y_range, center_y + y_range)
            z = random.uniform(*workspace_limits["z"])
            
            positions.append([x, y, z])
        
        return positions


def select_target_position_strategy():
    """
    Randomly select where to place target relative to density.
    This creates natural diversity in LOCAL complexity.
    """
    strategies = [
        "in_dense",      # Target in densest region (HIGH complexity)
        "near_dense",    # Target near dense region (MEDIUM-HIGH)
        "between",       # Target between clusters (MEDIUM)
        "isolated",      # Target far from others (LOW)
        "edge_sparse",   # Target at edge, away from center (LOW-MEDIUM)
    ]
    return random.choice(strategies)


def place_target_object(positions, target_strategy, workspace_limits):
    """
    Select or adjust target position based on strategy.
    
    Returns:
        target_index: Index of target in positions list
    """
    if len(positions) == 0:
        return 0
    
    center_x = (workspace_limits["x"][0] + workspace_limits["x"][1]) / 2
    center_y = (workspace_limits["y"][0] + workspace_limits["y"][1]) / 2
    
    if target_strategy == "in_dense":
        # Find densest region, place target there
        # Use first object (usually in dense area for most patterns)
        return 0
    
    elif target_strategy == "isolated":
        # Place target far from all others
        # Add new position far from existing ones
        max_attempts = 50
        best_pos = None
        best_min_dist = 0
        
        for attempt in range(max_attempts):
            x = random.uniform(*workspace_limits["x"])
            y = random.uniform(*workspace_limits["y"])
            z = random.uniform(*workspace_limits["z"])
            test_pos = np.array([x, y])
            
            # Calculate minimum distance to existing objects
            min_dist = min([np.linalg.norm(test_pos - np.array(pos[:2])) 
                           for pos in positions])
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = [x, y, z]
        
        if best_pos and best_min_dist > 0.15:
            positions.insert(0, best_pos)  # Add as first object (target)
            return 0
        else:
            # Fallback: use object furthest from center
            distances = [np.linalg.norm(np.array(pos[:2]) - np.array([center_x, center_y])) 
                        for pos in positions]
            return np.argmax(distances)
    
    elif target_strategy == "between":
        # Place target between clusters
        positions.insert(0, [center_x, center_y, random.uniform(*workspace_limits["z"])])
        return 0
    
    elif target_strategy == "near_dense":
        # Place near dense region but not in it
        if len(positions) > 3:
            # Near first few objects (usually dense)
            near_pos = positions[random.randint(0, min(3, len(positions)-1))]
            angle = random.uniform(0, 2*np.pi)
            offset_dist = random.uniform(0.10, 0.18)  # Nearby but not touching
            x = near_pos[0] + offset_dist * np.cos(angle)
            y = near_pos[1] + offset_dist * np.sin(angle)
            z = random.uniform(*workspace_limits["z"])
            
            x = np.clip(x, workspace_limits["x"][0], workspace_limits["x"][1])
            y = np.clip(y, workspace_limits["y"][0], workspace_limits["y"][1])
            
            positions.insert(0, [x, y, z])
            return 0
        return 0
    
    elif target_strategy == "edge_sparse":
        # Place at edge, away from center
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0.25, 0.35)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = random.uniform(*workspace_limits["z"])
        
        x = np.clip(x, workspace_limits["x"][0], workspace_limits["x"][1])
        y = np.clip(y, workspace_limits["y"][0], workspace_limits["y"][1])
        
        positions.insert(0, [x, y, z])
        return 0
    
    return 0


def generate_diverse_scene_xml(
    base_xml_path,
    output_xml_path,
    mesh_folder,
    workspace_limits,
    num_objects_range,
    camera_name,
    camera_pos_base,
    camera_pos_range,
    xyaxes_base,
    xyaxes_range,
    target_scale_factor
):
    """
    Generate scene with diverse density patterns and target placement.
    
    Returns:
        Tuple of (target_object_name, camera_position, xyaxes_str)
    """
    
    # Parse base XML
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    _absolutize_compiler_dirs(root, base_xml_path)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    
    # Random materials
    def random_rgba():
        return [random.random() for _ in range(3)] + [1.0]
    
    materials = []
    for i in range(20):
        mat_name = f"random_mat_{i}"
        rgba = " ".join(map(str, random_rgba()))
        material = ET.SubElement(asset, "material", {
            "name": mat_name,
            "rgba": rgba,
            "reflectance": str(random.uniform(0, 0.5)),
            "shininess": str(random.uniform(0.1, 1.0)),
            "specular": str(random.uniform(0.1, 1.0))
        })
        material.tail = "\n"
        materials.append(mat_name)
    
    # Random lighting
    light_pos = [
        round(random.uniform(-1.5, 1.5), 3),
        round(random.uniform(-1.5, 1.5), 3),
        round(random.uniform(1.0, 2.5), 3)
    ]
    diffuse = [round(random.uniform(0, 0.5), 2)] * 3
    
    light = ET.SubElement(worldbody, "light", {
        "name": "random_light",
        "pos": " ".join(map(str, light_pos)),
        "dir": "0 0 -1",
        "diffuse": " ".join(map(str, diffuse)),
        "castshadow": "true",
        "directional": "true"
    })
    light.tail = "\n"
    
    # Randomize ambient light
    visual = root.find("visual")
    if visual is not None:
        global_ = visual.find("global")
        if global_ is not None:
            ambient = [round(random.uniform(0.1, 0.9), 2)] * 3
            global_.set("ambient", " ".join(map(str, ambient)))
    
    # Select number of objects
    num_objects = random.randint(*num_objects_range)
    
    # Select density pattern
    density_patterns = [
        DensityPattern.clustered_center,
        DensityPattern.two_clusters,
        DensityPattern.uniform_random,
        DensityPattern.ring_pattern,
        DensityPattern.gradient_density
    ]
    selected_pattern = random.choice(density_patterns)
    pattern_name = selected_pattern.__name__
    
    # Generate positions with selected pattern
    positions = selected_pattern(num_objects, workspace_limits)
    
    # Select target placement strategy
    target_strategy = select_target_position_strategy()
    
    # Place target object
    target_index = place_target_object(positions, target_strategy, workspace_limits)
    
    print(f"Scene: {num_objects} objects, pattern={pattern_name}, target_strategy={target_strategy}")
    
    # Get mesh files
    mesh_files = [f for f in os.listdir(mesh_folder) if f.endswith(".obj")]
    selected_meshes = random.sample(mesh_files, min(len(positions), len(mesh_files)))
    
    target_object_name = None
    
    for i, (mesh_file, pos) in enumerate(zip(selected_meshes, positions)):
        obj_id = os.path.splitext(mesh_file)[0]
        mesh_name = f"random_obj_{i}_{obj_id}"
        mesh_path = os.path.abspath(os.path.join(mesh_folder, mesh_file))  # absolute: XML may live anywhere
        base_scale = 0.0015
        
        # Target object (index target_index)
        if i == target_index:
            target_object_name = mesh_name
            mesh_scale = " ".join([str(base_scale * target_scale_factor)] * 3)
        else:
            scale_factors = base_scale * (1 + 0.1 * np.random.uniform(-1, 1, size=3))
            mesh_scale = " ".join([f"{s:.6f}" for s in scale_factors])
        
        # Add mesh asset
        mesh = ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "file": mesh_path,
            "scale": mesh_scale
        })
        mesh.tail = "\n"
        
        # Random quaternion
        quat = np.random.randn(4)
        quat /= np.linalg.norm(quat)
        quat_str = " ".join(map(str, quat))
        
        # Add body
        body = ET.SubElement(worldbody, "body", {
            "name": mesh_name,
            "pos": " ".join(map(str, pos)),
            "quat": quat_str
        })
        body.tail = "\n"
        
        joint = ET.SubElement(body, "joint", {
            "name": f"joint_{i}",
            "type": "free",
            "damping": "0.8"
        })
        joint.tail = "\n"
        
        geom1 = ET.SubElement(body, "geom", {
            "mesh": mesh_name,
            "class": "obj_visual",
            "material": random.choice(materials)
        })
        geom1.tail = "\n"
        
        geom2 = ET.SubElement(body, "geom", {
            "mesh": mesh_name,
            "class": "obj_collision"
        })
        geom2.tail = "\n"
    
    # Add camera
    base_axes = list(map(float, xyaxes_base.split()))
    randomized_axes = base_axes + np.random.uniform(-xyaxes_range, xyaxes_range, 6)
    xyaxes_str = " ".join([f"{x:.3f}" for x in randomized_axes])
    
    random_pos = camera_pos_base + np.random.uniform(-1, 1, 3) * camera_pos_range
    pos_str = " ".join([f"{x:.3f}" for x in random_pos])
    
    camera = ET.SubElement(worldbody, "camera", {
        "name": camera_name,
        "mode": "fixed",
        "pos": pos_str,
        "xyaxes": xyaxes_str,
        "fovy": "45"
    })
    camera.tail = "\n"
    
    # Save XML
    tree.write(output_xml_path)
    print(f"Target: {target_object_name}")
    
    return target_object_name, random_pos, xyaxes_str

def generate_scene_xml(
    base_xml_path,
    output_xml_path,
    mesh_folder,
    workspace_limits,
    num_objects_range,
    camera_name,
    camera_pos_base,
    camera_pos_range,
    xyaxes_base,
    xyaxes_range,
    target_scale_factor,
    include_shelf_prob,
    include_obstacles,
    max_obstacles
):
    """
    Generate a randomized cluttered scene XML file for MuJoCo simulation.
    
    Args:
        base_xml_path: Path to the base XML template
        output_xml_path: Path to save the generated XML
        mesh_folder: Folder containing object mesh files (.obj)
        workspace_limits: Dictionary of workspace boundaries {'x': (min, max), ...}
        num_objects_range: Tuple of (min, max) number of objects
        camera_name: Name for the camera in the XML
        camera_pos_base: Base camera position [x, y, z]
        camera_pos_range: Maximum position randomization [dx, dy, dz]
        camera_angle_range: Maximum angle randomization in degrees
        target_scale_factor: Scale factor for the target object
    
    Returns:
        Tuple of (target_object_name, camera_position, camera_quaternion)
    """
    # Parse base XML
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    _absolutize_compiler_dirs(root, base_xml_path)

    # Ensure mesh assets section exists
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    worldbody = root.find("worldbody")

    # Generate random materials
    def random_rgba():
        return [random.random() for _ in range(3)] + [1.0]  # RGBA with full opacity

     # Random lighting setup
    light_pos = [
        round(random.uniform(-1.5, 1.5), 3),  # x
        round(random.uniform(-1.5, 1.5), 3),  # y
        round(random.uniform(1.0, 2.5), 3)    # z (keep light above scene)
    ]
    light_dir = [0, 0, -1]  # directional light pointing down

    # Light color and specular randomization
    diffuse = [round(random.uniform(0, 0.5), 2)] * 3

    light = ET.SubElement(worldbody, "light", {
        "name": "random_light",
        "pos": " ".join(map(str, light_pos)),
        "dir": " ".join(map(str, light_dir)),
        "diffuse": " ".join(map(str, diffuse)),
        "castshadow": "true",
        "directional": "true"
    })
    light.tail = "\n"
    
    # Randomize ambient light
    visual = root.find("visual")
    if visual is not None:
        global_ = visual.find("global")
        if global_ is not None:
            ambient = [round(random.uniform(0.1, 0.9), 2)] * 3
            global_.set("ambient", " ".join(map(str, ambient)))

    
    # Create several random materials
    materials = []
    for i in range(20):
        mat_name = f"random_mat_{i}"
        rgba = " ".join(map(str, random_rgba()))
        material = ET.SubElement(asset, "material", {
            "name": mat_name,
            "rgba": rgba,
            "reflectance": str(random.uniform(0, 0.5)),
            "shininess": str(random.uniform(0.1, 1.0)),
            "specular": str(random.uniform(0.1, 1.0))
        })
        material.text = "\n"
        material.tail = "\n"
        materials.append(mat_name)
    
    shelf_present = random.random() < include_shelf_prob
    shelf_pos = [0.1, 0.2, 1.24]
    table_height = 1.1
    og_shelf_size = [0.25, 0.25, 0.12]
    thickness = 0.02
    
    if shelf_present:
        shelf_offset_range = 0.02
        dx = random.uniform(-shelf_offset_range, shelf_offset_range)
        dy = random.uniform(-shelf_offset_range, shelf_offset_range)
        
        
        shelf_size_offset = 0.01
        dx_size = random.uniform(-shelf_offset_range, shelf_offset_range)
        dy_size = random.uniform(-shelf_offset_range, shelf_offset_range)
        dz_size = random.uniform(-shelf_offset_range, shelf_offset_range)
        shelf_size = [og_shelf_size[0] + dx_size, og_shelf_size[1] + dy_size, og_shelf_size[2] + dz_size ]
        
        shelf_center = [shelf_pos[0] + dx, shelf_pos[1] + dy, table_height + shelf_size[2]/2]

        shelf_body = ET.SubElement(worldbody, "body", {
            "name": "shelf",
            "pos": " ".join(map(str, shelf_pos)),
            "quat": "0 1 1 0"
        })
        shelf_body.tail = "\n"
        
        # Add free joint with high damping
        shelf_joint = ET.SubElement(shelf_body, "joint", {
            "name": "shelf_free_joint",
            "type": "free",
            "damping": "8.0"
        })
        shelf_joint.tail = "\n"
        

        def add_panel(name, pos, size):
            shelf_geom = ET.SubElement(shelf_body, "geom", {
                "name": name,
                "type": "box",
                "pos": " ".join(map(str, pos)),
                "size": " ".join(map(str, size)),
                "rgba": "0.6 0.6 0.6 1"
            })
            shelf_geom.tail = "\n"
            

        # Bottom
        add_panel("shelf_bottom", [0, 0, -shelf_size[2]], [shelf_size[0], shelf_size[1], thickness])
        # Back
        add_panel("shelf_back", [0, -shelf_size[1] + thickness, 0], [shelf_size[0], thickness, shelf_size[2]])
        # Left
        add_panel("shelf_left", [-shelf_size[0] + thickness, 0, 0], [thickness, shelf_size[1], shelf_size[2]])
        # Right
        add_panel("shelf_right", [shelf_size[0] - thickness, 0, 0], [thickness, shelf_size[1], shelf_size[2]])
        # Top
        add_panel("shelf_top", [0, 0, shelf_size[2]], [shelf_size[0], shelf_size[1], thickness])

     # Add optional random obstacles
    if include_obstacles:
        
        num_obs = random.randint(0, max_obstacles)
        for i in range(num_obs):
            
            while True:
                x = random.uniform(workspace_limits["x"][0], workspace_limits["x"][1])
                y = random.uniform(workspace_limits["y"][0], workspace_limits["y"][1])
                if not (shelf_present and abs(x - shelf_center[0]) < shelf_size[0]/2 and abs(y - shelf_center[1]) < shelf_size[1]/2):
                    break

            z = table_height + random.uniform(0.01, 0.05)
            shape = random.choice(["box", "cylinder"])
            min_size = 0.05
            max_size = 0.12
            size = []
            if shape == "box":
                for _ in range(3):
                    size.append(random.uniform(min_size, max_size))
            if shape == "cylinder":
                size = [random.uniform(min_size, max_size)] * (2)
                size.append(random.uniform(min_size, max_size))

            body = ET.SubElement(worldbody, "body", {
                "name": f"obstacle_{i}",
                "pos": f"{x:.3f} {y:.3f} {z:.3f}"
            })
            body.tail = "\n"
            geom_obs = ET.SubElement(body, "joint", {
                "name": f"obs_joint_{i}",
                "type": "free",
                "damping": "5.0"
            })
            geom_obs.tail = "\n"
            geom_obs1 = ET.SubElement(body, "geom", {
                "type": shape,
                "size": " ".join(map(str, size)),
                "rgba": "0.2 0.2 0.2 1"
            })
            geom_obs1.tail = "\n"
    
    # Get available mesh files
    mesh_files = [f for f in os.listdir(mesh_folder) if f.endswith(".obj")]
    num_objects = random.randint(*num_objects_range)
    selected_meshes = random.sample(mesh_files, num_objects)

    # Select a random target object
    target_object_index = random.randint(0, num_objects-1)
    target_object_name = None
    

    for i, mesh_file in enumerate(selected_meshes):
        obj_id = os.path.splitext(mesh_file)[0]
        mesh_name = f"random_obj_{i}" + "_" + obj_id
        mesh_path = os.path.abspath(os.path.join(mesh_folder, mesh_file))  # absolute: XML may live anywhere
        base_scale = 0.0015
        
        # Store target object name and apply different scaling
        if i == target_object_index:
            target_object_name = mesh_name
            mesh_scale = " ".join([str(base_scale * target_scale_factor)] * 3)
        else:
            base_scale = 0.0015
            scale_factors = base_scale * (1 + 0.1 * np.random.uniform(-1, 1, size=3))
            mesh_scale = " ".join([f"{s:.6f}" for s in scale_factors])

        # Add mesh asset
        mesh = ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "file": mesh_path,
            "scale": mesh_scale
        })
        mesh.tail = "\n    " if i < len(selected_meshes)-1 else "\n"
        
        # Random position and quaternion
        pos = [round(random.uniform(*workspace_limits[ax]), 3) for ax in ["x", "y", "z"]]
        quat = np.random.randn(4)
        quat /= np.linalg.norm(quat)
        quat_str = " ".join(map(str, quat))

        # Add body with visual and collision geoms
        body = ET.SubElement(worldbody, "body", {
            "name": mesh_name,
            "pos": " ".join(map(str, pos)),
            "quat": quat_str
        })
        body.text = "\n"
        body.tail = "\n"
        
        joint = ET.SubElement(body, "joint", {
            "name": f"joint_{i}",
            "type": "free",
            "damping": "0.8"
        })
        joint.tail = "\n"
        
        geom1 = ET.SubElement(body, "geom", {
            "mesh": mesh_name,
            "class": "obj_visual",
            "material": random.choice(materials)
        })
        geom1.tail = "\n"
        
        geom2 = ET.SubElement(body, "geom", {
            "mesh": mesh_name,
            "class": "obj_collision"
        })
        geom2.tail = "\n"

    # ===== Top-Down Camera Setup =====
    # Base orientation looking straight down (quaternion)
    # Add randomized xyaxes camera
    base_axes = list(map(float, xyaxes_base.split()))
    randomized_axes = base_axes + np.random.uniform(-xyaxes_range, xyaxes_range, 6)
    xyaxes_str = " ".join([f"{x:.3f}" for x in randomized_axes])
    
    # Randomize position slightly
    random_pos = camera_pos_base + np.random.uniform(-1, 1, 3) * camera_pos_range
    pos_str = " ".join([f"{x:.3f}" for x in random_pos])

    # Add top-down camera to XML
    camera = ET.SubElement(worldbody, "camera", {
        "name": camera_name,
        "mode": "fixed",
        "pos": pos_str,
        "xyaxes": xyaxes_str,
        "fovy": "45"
    })
    camera.tail = "\n"

    # Save modified XML
    tree.write(output_xml_path)
    print(f"Scene saved to {output_xml_path}")
    print(f"Target object: {target_object_name}")
 
    return target_object_name, random_pos, xyaxes_str


def get_body_corners_world(model, data, body_id):
    for i in range(model.ngeom):
        if model.geom_bodyid[i] != body_id:
            continue
        size = model.geom_size[i]
        xpos = data.geom_xpos[i]
        xmat = data.geom_xmat[i].reshape(3, 3)
  

    # Get the world coordinates of the box corners
    offsets = np.array([-1, 1]) * size[:, None]
    xyz_local = np.stack(list(itertools.product(*offsets))).T
    xyz_global = xpos[:, None] + xmat @ xyz_local

    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    return corners_homogeneous

def get_geom_corners_world(model, data, geom_id):
    size = model.geom_size[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)

    offsets = np.array([-1, 1]) * size[:, None]
    xyz_local = np.stack(list(itertools.product(*offsets))).T
    xyz_global = xpos[:, None] + xmat @ xyz_local

    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    return corners_homogeneous


def compute_camera_matrix(renderer, model, data, cam, cam_id):
    """Returns the 3x4 camera matrix for the specified named camera."""
    
    # Configure scene option and update scene
    for _ in range(500):
            mujoco.mj_step(model, data)
    option = mujoco.MjvOption()
    mjv_defaultOption(option)
    mujoco.mjv_updateScene(model, data, option, mujoco.MjvPerturb(), cam,
                           mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)

    # Access camera pose data
    cam_data = renderer.scene.camera[0]
    pos = data.cam(cam.fixedcamid).xpos
    z = -np.array(cam_data.forward)
    y = np.array(cam_data.up)
    x = np.cross(y, z)
    rot = np.vstack((x, y, z))
    # print(f"Pos: {pos}")
    # print(f"z: {z}")
    # print(f"y: {y}")
    # print(f"x: {x}")
    # print(f"rot: {rot}")
    # Translation matrix (4x4)
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4)
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Get FOV of the camera
    fov = model.cam_fovy[cam_id]
    # print(f"FOV: {fov}")

    # Focal transformation matrix (3x4)
    focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3)
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0

    # Final projection matrix
    return image @ focal @ rotation @ translation

def project_to_image(points, cam_mat):
    xs, ys, s = cam_mat @ points
    # x and y are in the pixel coordinate system.
    x = xs / s
    y = ys / s

    return x, y

def aabbs_intersect(min1, max1, min2, max2, margin=0.03):
    return np.all(max1 + margin >= min2) and np.all(max2 + margin >= min1)

def extract_difficulty_and_complexity(name):
    try:
        parts = name.split('_')
        letter = parts[3][0]
        grasp_difficulty = ord(letter.upper()) - ord('A') + 1
        shape_complexity = int(parts[3][1:])
        return grasp_difficulty, shape_complexity
    except:
        return None, None

def image_to_pixel_string(image):
    return ','.join(map(str, image.flatten()))

def get_body_name(model, body_id):
    start = model.name_bodyadr[body_id]
    if body_id + 1 < len(model.name_bodyadr):
        end = model.name_bodyadr[body_id + 1]
    else:
        end = len(model.names)
    return model.names[start:end].decode('utf-8')


def render_scene(model_path, camera_name, image_path, csv_path, neighbor_image_path, target_object_name=None):
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model)
        
        window = glfw.create_window(renderer.width, renderer.height, "MuJoCo Render", None, None)
        glfw.make_context_current(window)

        scene = mujoco.MjvScene(model, maxgeom=1000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        viewport = mujoco.MjrRect(0, 0, renderer.width, renderer.height)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        for i in range(model.ncam):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) == camera_name:
                cam.fixedcamid = i
                break

        for _ in range(5000):
            mujoco.mj_step(model, data)

        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        if target_object_name:
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_object_name)
            if target_body_id == -1:
                print("Target object not found.")
            else:
                # Getting the target information
                target_corners = get_body_corners_world(model, data, target_body_id)
                # print(f'corners: {corners}')
                cam_mat = compute_camera_matrix(renderer, model, data, cam, cam_id)
                projected = project_to_image(target_corners, cam_mat)
                # print(f'Projected: {projected}')
                xs, ys = projected

                # Compute the axis-aligned bounding box (AABB)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                bbox = (x_min, x_max, y_min, y_max)
                
                # Define the corners of the rectangle (clockwise)
                rect_x = [x_min, x_max, x_max, x_min, x_min]
                rect_y = [y_min, y_min, y_max, y_max, y_min]
                
                

                
                # Getting the furthest two points of the target object
                target_min = target_corners[:3, :].min(axis=1)
                target_max = target_corners[:3, :].max(axis=1)
                
                # Defining empty arrays
                neighbor_ids = []
                neighbor_dists = []
                neighbor_difficulties = []
                neighbor_complexities = []
                num_obstacles = 0
                
                # Looping through every objects to check for intersection
                # print(model.nbody)
      


                for i in range(model.nbody):
                    name = get_body_name(model, i)
                    if i == target_body_id or name is None:
                        continue
                    # Shelf handling: treat each geom separately
                    if "shelf" in name.lower():
                        for gid in range(model.ngeom):
                            if model.geom_bodyid[gid] != i:
                                continue  # Skip geoms not belonging to current body

                            corners = get_geom_corners_world(model, data, gid)
                            min_c = corners[:3, :].min(axis=1)
                            max_c = corners[:3, :].max(axis=1)

                            if aabbs_intersect(target_min, target_max, min_c, max_c, margin=0.02):
                                dists = np.linalg.norm(target_corners[:3, :, None] - corners[:3, None, :], axis=0)
                                min_dist = np.min(dists)
                                neighbor_ids.append((i, gid))  # Store body and geom id
                                neighbor_dists.append(min_dist)
                    else:
                        # Regular object logic
                        corners = get_body_corners_world(model, data, i)
                        if corners is None:
                            continue
                        min_c = corners[:3, :].min(axis=1)
                        max_c = corners[:3, :].max(axis=1)

                        if aabbs_intersect(target_min, target_max, min_c, max_c, margin=0.02):
                            dists = np.linalg.norm(target_corners[:3, :, None] - corners[:3, None, :], axis=0)
                            min_dist = np.min(dists)
                            neighbor_ids.append((i, None))  # None means whole body
                            neighbor_dists.append(min_dist)

                        if name.startswith("random_obj"):
                            gd, sc = extract_difficulty_and_complexity(name)
                            if gd is not None:
                                neighbor_difficulties.append(gd)
                                neighbor_complexities.append(sc)
                        else:
                            num_obstacles += 1
                
                target_gd, target_sc = extract_difficulty_and_complexity(target_object_name)
                # print(target_gd)
                # print(target_sc)
                # Expand the AABB of the target object by the margin in all directions
                radius = 0.03
                expanded_min = target_min - radius
                expanded_max = target_max + radius
                
                # Excluding target object itself as the freespace
                target_object_volume = np.prod(target_max - target_min)
                total_space_volume = np.prod(expanded_max - expanded_min)
                

                # Compute the free space as the volume of the expanded region
                total_free_space_vol = total_space_volume - target_object_volume
                free_space_vol = total_free_space_vol
                # print(f'free_space_volume: {free_space_vol}')
                print(neighbor_ids)
                for nid in neighbor_ids:
                    body_id, geom_id = nid

                    if geom_id is not None:
                        # Shelf part (specific geom)
                        corners = get_geom_corners_world(model, data, geom_id)
                    else:
                        # Whole object
                        corners = get_body_corners_world(model, data, body_id)

                    if corners is None:
                        continue  # Skip invalid

                    min_c = corners[:3, :].min(axis=1)
                    max_c = corners[:3, :].max(axis=1)
                    inter_min = np.maximum(target_min, min_c)
                    inter_max = np.minimum(target_max, max_c)
                    overlap_dims = np.maximum(0, inter_max - inter_min)
                    free_space_vol -= np.prod(overlap_dims)

                    if free_space_vol < 0:
                        free_space_vol = 0
                
                # Computing the ratio of free space
                free_space_vol_ratio = free_space_vol / total_free_space_vol
                
                target_z = data.xpos[target_body_id][2]
                table_height = 0.6
                # print(f'target z: {target_z}')

                if target_z < table_height:
                    print(f"Target object is below table at z = {target_z:.3f}, skipping image save.")
                else:
                    # Plot
                    pixels = renderer.render()

                    height, width = pixels.shape[:2]
                    # Check if bounding box is fully within image bounds
                    if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
                        print("Bounding box is outside the image. Skipping save.")
                    else:
                        # Create high-res figure
                        fig, ax = plt.subplots(figsize=(10, 7.5))  # High DPI
                        ax.imshow(pixels)
                        ax.plot(rect_x, rect_y, '-', c='lime', linewidth=2)
                        ax.set_axis_off()
                        # Save to memory buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                        # Read back the image with PIL, convert to numpy array
                        buf.seek(0)
                        green_box_image = np.array(Image.open(buf).convert('RGB'))

                        # This is what goes into the CSV
                        image_pixel_string = image_to_pixel_string(green_box_image)

                        # Save as PNG too, if you want
                        Image.fromarray(green_box_image).save(image_path)

                        # Save image
                        fig.savefig(image_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                        print(f"Image saved to {image_path}")
                        
                        # Render neighbor image
                        fig2, ax2 = plt.subplots(figsize=(10, 7.5))
                        ax2.imshow(pixels)
                        ax2.plot(rect_x, rect_y, '-', c='lime', linewidth=2)
                        # print(f'neighbor ID: {neighbor_ids}')

                        for nid, dist in zip(neighbor_ids, neighbor_dists):
                            body_id, geom_id = nid
                            name = get_body_name(model, body_id)

                            if geom_id is not None:
                                # Shelf part
                                corners = get_geom_corners_world(model, data, geom_id)
                            else:
                                corners = get_body_corners_world(model, data, body_id)

                            xs, ys = project_to_image(corners, cam_mat)
                            x_min, x_max = xs.min(), xs.max()
                            y_min, y_max = ys.min(), ys.max()
                            rect_x = [x_min, x_max, x_max, x_min, x_min]
                            rect_y = [y_min, y_min, y_max, y_max, y_min]
                            ax2.plot(rect_x, rect_y, '-', c='red', linewidth=1.5)
                            ax2.text(xs.mean(), ys.mean(), f"{dist:.2f}", color='white', fontsize=8)

                        ax2.text(10, 20, f"Free space: {free_space_vol_ratio:.4f}", color='white', fontsize=10)
                        ax2.set_axis_off()
                        fig2.savefig(neighbor_image_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig2)
                        print(f"Image saved to {neighbor_image_path}")

                        # Creating database
                        csv_row = {
                        "image_pixels": image_path,
                        "target_grasp_difficulty": target_gd,
                        "target_shape_complexity": target_sc,
                        "num_obstacles": num_obstacles,
                        "num_neighbors": len(neighbor_ids),
                        "mean_neighbor_distance": np.mean(neighbor_dists) if neighbor_dists else 0,
                        "min_neighbor_distance": min(neighbor_dists) if neighbor_dists else 0,
                        "mean_neighbor_grasp_difficulty": np.mean(neighbor_difficulties) if neighbor_difficulties else 0,
                        "mean_neighbor_shape_complexity": np.mean(neighbor_complexities) if neighbor_complexities else 0,
                        "free_space_volume": free_space_vol_ratio,
                        }
                        
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
                        else:
                            df = pd.DataFrame([csv_row])
                        df.to_csv(csv_path, index=False)
                        print(f"CSV updated at {csv_path}")
                        

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        glfw.terminate()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random cluttered MuJoCo scenes, render them and log the local clutter "
                    "around a target object (step 1 of the classifier pipeline).")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of scenes to generate and render")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index of the first scene (use to append to an existing CSV without "
                             "overwriting images)")
    parser.add_argument("--mesh_folder", type=str, default=os.path.join(DATA_DIR, "egad_mesh"),
                        help="Folder with EGAD .obj meshes (download from https://dougsm.github.io/egad/)")
    parser.add_argument("--base_xml", type=str,
                        default=os.path.join(SCRIPT_DIR, "obj_xml", "cluttered_scene.xml"),
                        help="Base MuJoCo scene (floor + table) the objects, lights and camera are added to")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(DATA_DIR, "classifier", "mujoco"),
                        help="Where the generated scene XML, images and CSV are written")
    parser.add_argument("--csv_name", type=str, default="diverse_local_complexity_data.csv",
                        help="Name of the output CSV inside --output_dir (rows are appended)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    image_dir = os.path.join(output_dir, "images")
    neighbor_dir = os.path.join(output_dir, "neighbor_check")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(neighbor_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, args.csv_name)

    if not os.path.isdir(args.mesh_folder):
        raise SystemExit(
            f"EGAD mesh folder not found: {args.mesh_folder}\n"
            "Download the EGAD .obj meshes from https://dougsm.github.io/egad/ and pass --mesh_folder.")

    config = {
        "base_xml_path": args.base_xml,
        "output_xml_path": os.path.join(output_dir, "random_cluttered_scene.xml"),
        "mesh_folder": args.mesh_folder,
        "workspace_limits": {
            "x": (0.2, 0.8),
            "y": (-0.45, 0.45),
            "z": (1.25, 1.25)
        },
        "num_objects_range": (3, 30),  # Wider range for diversity
        "camera_name": "top_front_cam",
        "camera_pos_base": np.array([2.117, 0.013, 2.060]),
        "camera_pos_range": np.array([0.0001, 0.0001, 0.0001]),
        "xyaxes_base": "-0.019 1.000 0.000 -0.607 -0.011 0.795",
        "xyaxes_range": 0.01,
        "target_scale_factor": 1.33,
    }
    
    num_images = args.num_images
    
    print("="*70)
    print("DIVERSE LOCAL COMPLEXITY SCENE GENERATION")
    print("="*70)
    print("Scene patterns: clustered, two_clusters, uniform, ring, gradient")
    print("Target strategies: in_dense, near_dense, between, isolated, edge_sparse")
    print(f"Total scenes: {num_images}")
    print(f"Output dir:   {output_dir}")
    print("="*70)
    
    for i in range(args.start_index, args.start_index + num_images):
        # Generate scene with random density pattern and target placement
        target_obj_name, cam_pos, cam_quat = generate_diverse_scene_xml(**config)
        
        image_name = f"scene_{i:05d}.png"
        neighbor_image = f"neighbor_{i:05d}.png"
        
        render_config = {
            "model_path": config["output_xml_path"],
            "camera_name": config["camera_name"],
            # Absolute paths are stored in the CSV so later steps work from any directory.
            "image_path": os.path.join(image_dir, image_name),
            "csv_path": csv_path,
            "neighbor_image_path": os.path.join(neighbor_dir, neighbor_image),
            "target_object_name": target_obj_name
        }
        
        render_scene(**render_config)
        
        done = i - args.start_index + 1
        if done % 100 == 0:
            print(f"Progress: {done}/{num_images} ({done/num_images*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Dataset generation complete!")
    print(f"CSV: {csv_path}")
    print("Next: python spatial_features_extraction_raw.py --input_csv <CSV>")
