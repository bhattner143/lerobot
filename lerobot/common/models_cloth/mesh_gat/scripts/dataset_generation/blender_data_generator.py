"""
This script performs a cloth simulation on a T-shirt model in Blender.
THis script is based on script_cloth_tshirtv2_coarse_random_motion.py and OptimalProjection.py
"""

import bpy
import pickle
import os
import time
import numpy as np
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import bmesh
except ImportError:
    print("Cannot import bmesh")
    import torch
    import argparse
    from utils.pytorch3d_pcd_utils import PyTorchRenderer, read_mesh_vertices, normalize_depth
    # import pyvista as pv
except ImportError:
    print("Cannot import cv2, torch, argparse, PyTorch3d")

try:
    # import matplotlib
    # matplotlib.use('Agg')  # Use a non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Cannot import matplotlib")

# --------------------------------------------------------- #
# ------------------------- utils ------------------------- #
# --------------------------------------------------------- #

def styled_message(message, fg_color="37", bg_color="44", bold=True):
    """
    Prints a styled message with customizable colors and bold effects.

    Args:
    - message (str): The message to display.
    - fg_color (str): ANSI code for foreground color (default: white = "37").
    - bg_color (str): ANSI code for background color (default: blue = "44").
    - bold (bool): Whether to make the text bold (default: True).
    """
    style = f"\033[{'1;' if bold else ''}{fg_color};{bg_color}m"
    reset = "\033[0m"
    print(f"\n{style}{message}{reset}\n")

reset = "\033[0m"
bold = "\033[1m"
green = "\033[92m"
yellow = "\033[93m"
red = "\033[91m"

def first_time_initialization(obj_file_path, base_dir, cloth_style):
    model = ClothBlenderModel(obj_file_path, base_dir, cloth_style = cloth_style)
    # Initialize scene
    model.ensure_obj_mode_and_del_existing_obj_datablock()
    # Generate the ground plane, camera and sunlight
    model.generate_plane_camera_sun()
    if cloth_style == 'T_Shirtv2':
        model.generate_t_shirtv2_model()
    elif cloth_style == 'T_Shirt_l1':
        model.generate_t_shirt_l1_model()
    elif cloth_style == 'T_Shirt_l2':
        model.generate_t_shirt_l2_model()
    elif cloth_style == 'T_Shirt_l3':
        model.generate_t_shirt_l3_model()

    # Save the current cloth as initial state, saved as template_{cloth_name}.pickle file
    model.save_current_cloth_as_init(save_file=True, overwrite = True)

# --------------------------------------------------------- #
# ----------- Global Cloth Simulation Parameters ---------- #
# --------------------------------------------------------- #

CLOTH_SETTINGS = {
    
    "mass": 1/3,  # Mass of cloth vertices
    "air_damping": 1,  # Air resistance (viscosity)

    # Stiffness parameters
    "tension_stiffness": 20/3,  # Resistance to stretching
    "compression_stiffness": 20/3,  # Resistance to compression
    "shear_stiffness": 12/3,  # Resistance to shear forces (set shear manually)
    "bending_stiffness": 20/3,  # Resistance to bending (set bend manually)

    # Damping parameters
    "tension_damping": 20/3,  # Damping for tension
    "compression_damping": 20/3,  # Damping for compression
    "shear_damping": 12/3,  # Damping for shear
    "bending_damping": 20/3,  # Damping for bending

    # Self-collision settings
    "use_self_collision": True,  # Enable self-collision
    "self_friction": 10/3,  # Friction for self-collision
    "self_distance_min": 0.01/3,  # Minimum allowed distance

    # backup value for tshirt l2
    # "mass": 0.5,  # Mass of cloth vertices
    # "air_damping": 1,  # Air resistance (viscosity)

    # # Stiffness parameters
    # "tension_stiffness": 10 ,  # Resistance to stretching
    # "compression_stiffness": 5,  # Resistance to compression
    # "shear_stiffness": 4,  # Resistance to shear forces (set shear manually)
    # "bending_stiffness": 4,  # Resistance to bending (set bend manually)

    # # Damping parameters
    # "tension_damping": 10,  # Damping for tension
    # "compression_damping": 5,  # Damping for compression
    # "shear_damping": 4,  # Damping for shear
    # "bending_damping": 4,  # Damping for bending

    # # Self-collision settings
    # "use_self_collision": True,  # Enable self-collision
    # "self_friction": 10,  # Friction for self-collision
    # "self_distance_min": 0.01/2,  # Minimum allowed distance

    # Default values for the cloth simulation
    # _bending_stiffness = 0.5       # Default value for cotton t-shirt bending stiffness
    # _self_friction = 5             # Default value for cotton t-shirt self friction
    # _self_collision = True         # Enable self-collision by default
    # _self_distance_min = 0.001     # Default self-collision minimum distance
    # _mass = 0.3                    # Default mass for a cotton t-shirt (~300g)
    # _impulse_clamp = 0.5           # Default impulse clamp for collision response
    # _structural_stiffness = 15     # Default structural stiffness to prevent overstretching
    # _air_damping = 0.2             # Default air damping (simulates air resistance)
    # _quality = 5                   # Default quality steps for the cloth simulation
    # _self_collision_distance = 0.005 # Default self-collision distance (5mm)
}  

CLOTH_SETTINGS_l3 = {
    
    "mass": 0.25,  # Mass of cloth vertices
    "air_damping": 1,  # Air resistance (viscosity)

    # Stiffness parameters
    "tension_stiffness": 30,  # Resistance to stretching
    "compression_stiffness": 10,  # Resistance to compression
    "shear_stiffness": 5,  # Resistance to shear forces (set shear manually)
    "bending_stiffness": 5,  # Resistance to bending (set bend manually)

    # Damping parameters
    "tension_damping": 5,  # Damping for tension
    "compression_damping": 5,  # Damping for compression
    "shear_damping": 5,  # Damping for shear
    "bending_damping": 5,  # Damping for bending

    # Self-collision settings
    "use_self_collision": True,  # Enable self-collision
    "self_friction": 5,  # Friction for self-collision
    "self_distance_min": 0.0005,  # Minimum allowed distance
}  

CLOTH_SETTINGS_l1 = {
    
    "mass": 0.2,  # Mass of cloth vertices
    "air_damping": 1,  # Air resistance (viscosity)

    # Stiffness parameters
    "tension_stiffness": 60,  # Resistance to stretching
    "compression_stiffness": 40,  # Resistance to compression
    "shear_stiffness": 20,  # Resistance to shear forces (set shear manually)
    "bending_stiffness": 10,  # Resistance to bending (set bend manually)

    # Damping parameters
    "tension_damping": 50,  # Damping for tension
    "compression_damping": 50,  # Damping for compression
    "shear_damping": 20,  # Damping for shear
    "bending_damping": 10,  # Damping for bending

    # Self-collision settings
    "use_self_collision": True,  # Enable self-collision
    "self_friction": 0.1,  # Friction for self-collision
    "self_distance_min": 0.00001,  # Minimum allowed distance
}  


class ClothBlenderModel:

    def __init__(self, obj_file_path, database_dir, cloth_style = None, multi = False):
        """
        Initializes "Blender Scene" and "Cloth object" with the given OBJ file path and base directory.

        Parameters:
        obj_file_path (str): Path to the cloth OBJ file.
        database_dir (str): Base directory for saving state files.
        """
        self.obj_file_path = obj_file_path
        self.database_dir = database_dir
        self.cloth_style = cloth_style
        self.multi = multi

        # scene settings
        self.scene = bpy.context.scene
        self.tree = None
        self.links = None
        self.rl = None
        self.composite = None
        self.map = None
        self.fileOutput = None

        # render settings
        self.scene.render.engine = 'CYCLES'

        # camera settings
        self.cam_loc = (0, 0, 3)
        self.cam_orientation = (0, 0, 0)
        self.fov = 45                                   # camera filed of view

        # light settings
        self.light_loc = (0.65, -0.4, 2.05)
        self.light_orientation = (0, 0, 0)
        self.light_intensity = 8.32                     # light intensity

        # cloth settings
        self.mesh_vertices = {}                         # dictionary to store all stages of mesh vertices
        self.cloth = None                               # cloth object
        self.cloth_color = (0, 0, 0, 1)      # cloth color
        self.baseZ = 0.01                               # base Z position of the cloth
    
        # state settings
        self.ms = 0                                     # manipulation step
        self.meshes = None                              # mesh data
        self.edges = None                               # edges data
        self.faces = None                               # faces data
        self.vtx_picks = []                             # vertex picks
        self.co_picks = None                            # coordinate picks
        self.moves = None                               # move vectors
        self.hand_loc = None                            # hand location

        self.selected_folders = []                      # selected folders for normal maps

    def assign(self, np_mesh):
        """
        Assigns NumPy mesh data to the cloth object's mesh.

        Parameters:
        np_mesh (np.ndarray): NumPy array of vertex positions.
        """
        print(
            f'Assigning np_mesh with shape {np_mesh.shape} to cloth data vertices.')
        for i in range(len(self.cloth.data.vertices)):
            self.cloth.data.vertices[i].co = np_mesh[i]

    def get_existing_object(self, type = None):
        # Find an existing camera
        obj = next(obj for obj in bpy.context.scene.objects if obj.type == type)
        
        return obj

    def ensure_obj_mode_and_del_existing_obj_datablock(self):
        """
        Delete all existing objects in the scene. (from Dips)
        Additionally, delete unneeded data blocks. (from Tokuda)
        """

        # Ensure Blender is in Object Mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Delete all existing objects in the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.outliner.orphans_purge(do_recursive=True)

        # Delete unneeded data blocks
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

        styled_message("Deleted all existing object and initialized sence.", fg_color="33", bg_color="44")

    def initialize_scene_depth(self):
        '''
        Delete all existing objects and initialize the scene.
        '''

        #Delete existing objects
        self.ensure_obj_mode_and_del_existing_obj_datablock()
        
        self.scene = bpy.context.scene
        self.scene.render.engine = 'CYCLES'  # Using render Cycles
        self.scene.cycle.device = 'GPU'  # Using GPU for rendering
        
        # Enable node-based compositing for the scene
        self.scene.use_nodes = True
        
        # Disable multiview rendering (stereo or multi-camera setups)
        self.scene.render.use_multiview = False
        
        # Set the rendering view format to Stereo 3D
        self.scene.render.views_format = 'STEREO_3D'
        
        # Enable the depth pass for the render layers, allowing depth information to be captured
        self.scene.view_layers["ViewLayer"].use_pass_z = True
        
        # Access the node tree of the scene (used for compositing)
        self.tree = self.scene.node_tree
        self.links = self.tree.links  # Access the links between nodes
        
        # Clear all existing nodes from the node tree, ensuring a clean slate
        for n in self.tree.nodes:
            self.tree.nodes.remove(n)
        
        # Clear all existing links between nodes
        for n in self.tree.links:
            self.tree.links.remove(n)
            
        # Add a new Render Layers node, which provides the rendered image and passes (e.g., depth)
        self.rl = self.tree.nodes.new(type="CompositorNodeRLayers")
        
        # Add a Composite node, which is used to output the final result of the node tree
        self.composite = self.tree.nodes.new(type="CompositorNodeComposite")
        self.composite.location = 300, 0  # Position the Composite node in the node editor
        
        # Add a Map Value node, used to remap the depth values from the Depth pass into a 0-255 range
        self.map = self.tree.nodes.new(type="CompositorNodeMapValue")
        self.map.location = 300, 300  # Position the Map Value node
        
        # Set the scaling factor for the depth values; this affects how the depth is visualized
        self.map.size = [0.2]
        
        # Enable clamping of the minimum value in the depth map (anything below this will be clamped to 0)
        self.map.use_min = True
        self.map.min = [0]  # Set the minimum value to 0
        
        # Enable clamping of the maximum value in the depth map (anything above this will be clamped to 255)
        self.map.use_max = True
        self.map.max = [255]  # Set the maximum value to 255
        
        # Connect the Depth output of the Render Layers node to the input of the Map Value node
        self.links.new(self.rl.outputs['Depth'], self.map.inputs[0])
        
        # Add a File Output node, used to save the rendered images to disk
        self.fileOutput = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.fileOutput.location = 500, 300  # Position the File Output node
        
        # Set the base path where the files will be saved
        self.fileOutput.base_path = self.database_dir
        
        # Connect the output of the Map Value node (processed depth values) to the File Output node
        self.links.new(self.map.outputs[0], self.fileOutput.inputs[0])

        styled_message("Deleted all existing object and initialized sence.", fg_color="33", bg_color="44")

    def generate_plane_camera_sun(self):
        """
        Generates the ground plane camera and sunlight, sets up the Blender scene
        """

        # --------------------------
        # Create the Ground Plane
        # --------------------------
        # Add Plane with Collision Modifier
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False,align='WORLD',location=(0, 0, -0.01),scale=(1, 1, 1))
        # Scale the plane to make it larger
        bpy.context.object.scale = [4, 4, 1]
        plane_obj = bpy.context.object
        
        # Collect vertices of the plane (not used further in the code)
        vertices_plane = np.array([v.co[:] for v in plane_obj.data.vertices])
        
        # Add Collision Modifier to the plane
        bpy.ops.object.modifier_add(type='COLLISION')
        bpy.context.object.collision.cloth_friction = 20
        # bpy.context.object.hide_render = True  # Hide ground object in the render

        # change table color to white
        mat = bpy.data.materials.new(name="TableMaterial")
        mat.diffuse_color = (0.2, 0.2, 0.2, 1)
        plane_obj.data.materials.append(mat)

        # --------------------------
        # Add Camera
        # -------------------------- 
        # Ensure the camera is set up
        if not any(obj.type == 'CAMERA' for obj in bpy.context.scene.objects):
            # Add a new camera if none exists
            bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 4.5), rotation=(0, 0, 0), scale=(1, 1, 1))
            
            # Get the newly created camera
            camera = bpy.context.object
            
            # Set this camera as the active camera for the scene
            bpy.context.scene.camera = camera
        else:
            # Find an existing camera
            camera = self.get_existing_object(type = 'CAMERA')
               
        # Position the camera for a top-down view
        camera.location       = self.cam_loc # Position the camera above the cloth (assuming self.baseZ = 0)
        camera.rotation_euler = self.cam_orientation  # Rotate around axes
        
        # --------------------------
        # Set Camera Field of View (FOV)
        # --------------------------
        # Define the FOV in degrees
        # fov_degrees = self.fov  # Adjust this value as needed
        # fov_radians = np.radians(fov_degrees)  # Convert to radians (Blender uses radians)
        # camera.data.angle = fov_radians  # Set the horizontal FOV
        
        # You can scale the camera object, but this does not affect the final render
        # camera.scale.x = 1.0  # Scale along the X-axis
        # camera.scale.y = 1.0  # Scale along the Y-axis
        # camera.scale.z = 1.0  # Scale along the Z-axis

        # Set render resolution (this is a scene property, not a camera property)
        self.scene.render.engine = 'CYCLES'
        self.scene.render.resolution_x = 1920  # Width in pixels
        self.scene.render.resolution_y = 1080 # Height in pixels
        self.scene.render.image_settings.file_format = 'PNG'
        
        bpy.context.scene.render.resolution_percentage = 100  # Render at 100% of the resolution
        # Output file path (change to your desired path)
        output_filepath = os.path.join(self.database_dir, "blender_render.png")
        self.scene.render.filepath = output_filepath
        
        # --------------------------
        # Add Sun light
        # --------------------------
        # Add a sun light if none exists
        if not any(obj.type == 'LIGHT' for obj in bpy.context.scene.objects):
            bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))  # Add a sun light
            light = bpy.context.object
            light.data.energy = self.light_intensity  # Adjust the brightness as needed                
        # Position the light at an angle to the cloth to cast proper shadows
        light.location       = self.light_loc  # Adjust this to suit your scene
        light.rotation_euler = self.light_orientation  # Angle the light to hit the cloth surface
        
        styled_message("Ground plane, camera and sun added.", fg_color="33", bg_color="44")

    def generate_t_shirtv2_model(self, texture_dir = None, shear=40, bend=40, scale=(1, 1, 1), centralize=True):
        """
        Generates the cloth model and prepares the cloth object with necessary modifiers for simulation.
        This method imports a T-shirt model, applies transformations, sets up materials, 
        and configures cloth simulation physics.
        
        Parameters:
        - shear (int): Shear stiffness for the cloth simulation.
        - bend (int): Bending stiffness for the cloth simulation.
        - scale (tuple): Scaling factors for the cloth object.
        - centralize (bool): Whether to centralize the object in the scene.
        """

        # --------------------------
        # Import and Prepare the Cloth Object
        # --------------------------
        # Import the cloth OBJ file (ensure the file path is provided in `obj_file_path`)
        bpy.ops.wm.obj_import(filepath=self.obj_file_path)
    
        # # Get the imported cloth object and rename it to "T_Shirtv2"
        self.cloth = bpy.context.selected_objects[0]
        self.cloth.name = 'T_Shirtv2'
    
        # Convert the mesh to NumPy arrays (initial state on the XZ plane)
        mesh_vertices_initial, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # Store the initial vertices in the dictionary
        self.mesh_vertices = {'mesh_vertices_initial': mesh_vertices_initial}
    
        # Ensure the imported cloth object is active and in Object Mode
        bpy.context.view_layer.objects.active = self.cloth
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.cloth.show_axis = True   

        # --------------------------
        # Step 1: Apply rotation and scale to ensure consistent axes
        # --------------------------
        # Apply transformations to align the mesh with the XY plane
        # This rotates the mesh to align with the XY plane and applies scaling
        # mesh_vertices*Rotation_matrix
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after setting the origin
        # mesh_vertices_1, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        # self.mesh_vertices['mesh_vertices_1_rotated_scaled'] = mesh_vertices_1
        
        # # --------------------------
        # # Step 2: Set the object's origin to the geometry center
        # # --------------------------
        # # Move the object's origin to the center of its mesh geometry
        # # This step aligns the pivot point with the mesh center for easier transformations
        # # object's origin to align with the mesh's geometry (e.g., center of the mesh), 
        # # you can move the origin using: 
        # # shifts the object origin to the center of the mesh and update self.cloth.location
        # # Performs mesh_vertices_initial-np.array(self.cloth.location) and
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
        # # Convert the mesh to NumPy arrays after applying rotation and scale
        # mesh_vertices_2, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        # self.mesh_vertices['mesh_vertices_2_origin'] = mesh_vertices_2
        
        # # --------------------------
        # # Step 3: Update the object's origin and scale 
        # # --------------------------
        self.cloth.location = (0, 0, self.baseZ)  # Move the object along the Z-axis
        self.cloth.scale = scale  # Apply scaling to the object
        
        # # --------------------------
        # # Step 4: Translates the mesh according to updated self.cloth.location  
        # # --------------------------
        # # This translates the mesh according to updated self.cloth.location 
        # # and since self.cloth.location is center of mesh so the entire mesh shifts
        # # so that the center of the mesh is at the origin
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
        
        # # Convert the mesh to NumPy arrays after applying location changes
        # mesh_vertices_3, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        # self.mesh_vertices['mesh_vertices_3'] = mesh_vertices_3
    
        # # Ensure the object is active after transformation
        # bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Step 5: Assign Material and Enable Smooth Shading
        # # --------------------------
        # # Set the material of the cloth object to the newly created cloth material
        
        # # Enable smooth shading for better appearance
        bpy.ops.object.shade_smooth()
    
        # # Assign the cloth material to the object
        # if self.cloth.data.materials:
        #     self.cloth.data.materials[0] = cloth_material  # Replace the existing material
        # else:
        #     self.cloth.data.materials.append(cloth_material)  # Add the material if none exist
        if texture_dir is not None: 
            bsdf_directory = os.path.join(texture_dir, "bsdf_info")
            normal_filename = self.select_random_folder_and_find_normal_file(bsdf_directory)
            self.pattern(self.cloth, normal_filename)
    
        # # --------------------------
        # # Step 6: Set Up Modifiers for Simulation
        # # --------------------------
        # # Add Vertex Weight Mix Modifier for weight manipulation
        vw_mix_mod = self.cloth.modifiers.new(name='VertexWeightMix', type='VERTEX_WEIGHT_MIX')
    
        # # Add Hook Modifiers for manipulating specific vertices
        hook_mod1 = self.cloth.modifiers.new(name='Hook1', type='HOOK')
        if self.multi:  # If multiple hooks are enabled, create another Hook modifier
            hook_mod2 = self.cloth.modifiers.new(name='Hook2', type='HOOK')
    
        # Add the Cloth physics modifier to enable cloth simulation
        cloth_mod = self.cloth.modifiers.new(name='Cloth', type='CLOTH')
        
    
        # # Convert the mesh to NumPy arrays after adding modifiers
        mesh_vertices_4, self.edges, faces, self.faces = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the final vertices in the dictionary
        self.mesh_vertices['mesh_vertices_4'] = mesh_vertices_4
        self.meshes = mesh_vertices_4

        # # --------------------------
        # # Using for debugging
        # # --------------------------
        # # Extract x, y, z coordinates from the vertices
        # x = [v[0] for v in mesh_vertices_4]
        # y = [v[1] for v in mesh_vertices_4]
        # z = [v[2] for v in mesh_vertices_4]
    
        # # Create a 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
    
        # # Scatter plot of the vertices
        # ax.scatter(x, y, z, c='blue', marker='o', s=10, label='Mesh Vertices')

        # # Limit the Z-axis from 0 to 0.2
        # ax.set_zlim(-0.1, 0.2)

        # # Set labels and title
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D Plot of Mesh Vertices')
    
        # # Add legend
        # ax.legend()
    
        # # Show grid and plot
        # ax.grid(True)
        # plt.savefig('3D_Plot_Mesh_Vertices_tshirt_berkly.png')

        # exit()
        
        # --------------------------
        # Configure Cloth Modifier Settings
        # --------------------------
        # Set physical properties for the cloth simulation
        # Usage in the Cloth Modifier
        cloth_mod.settings.mass = CLOTH_SETTINGS["mass"]
        cloth_mod.settings.air_damping = CLOTH_SETTINGS["air_damping"]

        cloth_mod.settings.tension_stiffness = CLOTH_SETTINGS["tension_stiffness"]
        cloth_mod.settings.compression_stiffness = CLOTH_SETTINGS["compression_stiffness"]
        cloth_mod.settings.shear_stiffness = CLOTH_SETTINGS["shear_stiffness"]
        cloth_mod.settings.bending_stiffness = CLOTH_SETTINGS["bending_stiffness"]

        cloth_mod.settings.tension_damping = CLOTH_SETTINGS["tension_damping"]
        cloth_mod.settings.compression_damping = CLOTH_SETTINGS["compression_damping"]
        cloth_mod.settings.shear_damping = CLOTH_SETTINGS["shear_damping"]
        cloth_mod.settings.bending_damping = CLOTH_SETTINGS["bending_damping"]

        cloth_mod.collision_settings.use_self_collision = CLOTH_SETTINGS["use_self_collision"]
        cloth_mod.collision_settings.self_friction = CLOTH_SETTINGS["self_friction"]
        cloth_mod.collision_settings.self_distance_min = CLOTH_SETTINGS["self_distance_min"]
    
        # --------------------------
        # Final Message
        # --------------------------
        # Log or print a styled message indicating the cloth model generation is complete
        styled_message("Cloth t_shirtv2 (Berkley model) model generated.", fg_color="33", bg_color="44")

    def generate_t_shirt_l1_model(self, texture_dir = None, shear=40, bend=40, scale=(1, 1, 1), centralize=True):
        """
        Generates the cloth model and prepares the cloth object with necessary modifiers for simulation.
        This method imports a T-shirt model, applies transformations, sets up materials, 
        and configures cloth simulation physics.
        
        Parameters:
        - shear (int): Shear stiffness for the cloth simulation.
        - bend (int): Bending stiffness for the cloth simulation.
        - scale (tuple): Scaling factors for the cloth object.
        - centralize (bool): Whether to centralize the object in the scene.
        """

        # --------------------------
        # Import and Prepare the Cloth Object
        # --------------------------
        # Import the cloth OBJ file (ensure the file path is provided in `obj_file_path`)
        bpy.ops.wm.obj_import(filepath=self.obj_file_path)
    
        # # Get the imported cloth object and rename it to "T_Shirt_l1"
        self.cloth = bpy.context.selected_objects[0]
        self.cloth.name = 'T_Shirt_l1'
    
        # Convert the mesh to NumPy arrays (initial state on the XZ plane)
        mesh_vertices_initial, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # Store the initial vertices in the dictionary
        self.mesh_vertices = {'mesh_vertices_initial': mesh_vertices_initial}
    
        # Ensure the imported cloth object is active and in Object Mode
        bpy.context.view_layer.objects.active = self.cloth
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.cloth.show_axis = True   

        # --------------------------
        # Step 1: Apply rotation and scale to ensure consistent axes
        # --------------------------
        # Apply transformations to align the mesh with the XY plane
        # This rotates the mesh to align with the XY plane and applies scaling
        # mesh_vertices*Rotation_matrix
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after setting the origin
        mesh_vertices_1, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_1_rotated_scaled'] = mesh_vertices_1
        
        # # --------------------------
        # # Step 2: Set the object's origin to the geometry center
        # # --------------------------
        # # Move the object's origin to the center of its mesh geometry
        # # This step aligns the pivot point with the mesh center for easier transformations
        # # object's origin to align with the mesh's geometry (e.g., center of the mesh), 
        # # you can move the origin using: 
        # # shifts the object origin to the center of the mesh and update self.cloth.location
        # # Performs mesh_vertices_initial-np.array(self.cloth.location) and
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
        # # Convert the mesh to NumPy arrays after applying rotation and scale
        mesh_vertices_2, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_2_origin'] = mesh_vertices_2
        
        # # --------------------------
        # # Step 3: Update the object's origin and scale 
        # # --------------------------
        self.cloth.location = (0, 0, self.baseZ)  # Move the object along the Z-axis
        self.cloth.rotation_euler = (0, 0, -np.pi/2) # rotate the object around z axis by 90 degree
        self.cloth.scale = scale  # Apply scaling to the object
        
        # # --------------------------
        # # Step 4: Translates the mesh according to updated self.cloth.location  
        # # --------------------------
        # # This translates the mesh according to updated self.cloth.location 
        # # and since self.cloth.location is center of mesh so the entire mesh shifts
        # # so that the center of the mesh is at the origin
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after applying location changes
        mesh_vertices_3, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_3'] = mesh_vertices_3
    
        # # Ensure the object is active after transformation
        bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Step 5: Assign Material and Enable Smooth Shading
        # # --------------------------
        # # Enable smooth shading for better appearance
        bpy.ops.object.shade_smooth()
  
        if texture_dir is not None:
            bsdf_directory = os.path.join(texture_dir, "bsdf_info")
            normal_filename = self.select_random_folder_and_find_normal_file(bsdf_directory)
            self.pattern(self.cloth, normal_filename)
    
        # # --------------------------
        # # Step 6: Set Up Modifiers for Simulation
        # # --------------------------
        # # Add Vertex Weight Mix Modifier for weight manipulation
        vw_mix_mod = self.cloth.modifiers.new(name='VertexWeightMix', type='VERTEX_WEIGHT_MIX')
    
        # # Add Hook Modifiers for manipulating specific vertices
        hook_mod1 = self.cloth.modifiers.new(name='Hook1', type='HOOK')
        if self.multi:  # If multiple hooks are enabled, create another Hook modifier
            hook_mod2 = self.cloth.modifiers.new(name='Hook2', type='HOOK')
    
        # Add the Cloth physics modifier to enable cloth simulation
        cloth_mod = self.cloth.modifiers.new(name='Cloth', type='CLOTH')
        
    
        # # Convert the mesh to NumPy arrays after adding modifiers
        mesh_vertices_4, self.edges, faces, self.faces = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the final vertices in the dictionary
        self.mesh_vertices['mesh_vertices_4'] = mesh_vertices_4
        self.meshes = mesh_vertices_4

        # --------------------------
        # Configure Cloth Modifier Settings
        # --------------------------
        # Set physical properties for the cloth simulation
        # Usage in the Cloth Modifier
        cloth_mod.settings.mass = CLOTH_SETTINGS_l1["mass"]
        cloth_mod.settings.air_damping = CLOTH_SETTINGS_l1["air_damping"]

        cloth_mod.settings.tension_stiffness = CLOTH_SETTINGS_l1["tension_stiffness"]
        cloth_mod.settings.compression_stiffness = CLOTH_SETTINGS_l1["compression_stiffness"]
        cloth_mod.settings.shear_stiffness = CLOTH_SETTINGS_l1["shear_stiffness"]
        cloth_mod.settings.bending_stiffness = CLOTH_SETTINGS_l1["bending_stiffness"]

        cloth_mod.settings.tension_damping = CLOTH_SETTINGS_l1["tension_damping"]
        cloth_mod.settings.compression_damping = CLOTH_SETTINGS_l1["compression_damping"]
        cloth_mod.settings.shear_damping = CLOTH_SETTINGS_l1["shear_damping"]
        cloth_mod.settings.bending_damping = CLOTH_SETTINGS_l1["bending_damping"]

        cloth_mod.collision_settings.use_self_collision = CLOTH_SETTINGS_l1["use_self_collision"]
        cloth_mod.collision_settings.self_friction = CLOTH_SETTINGS_l1["self_friction"]
        cloth_mod.collision_settings.self_distance_min = CLOTH_SETTINGS_l1["self_distance_min"]
            
        # Clear any existing animation data for the cloth
        self.cloth.animation_data_clear()
        # Clear any existing vertex groups
        self.cloth.vertex_groups.clear()  
        # ensure active
        bpy.context.view_layer.objects.active = self.cloth

        # # --------------------------
        # # Using for debugging
        # # --------------------------
        # # Extract x, y, z coordinates from the vertices
        # x = [v[0] for v in mesh_vertices_4]
        # y = [v[1] for v in mesh_vertices_4]
        # z = [v[2] for v in mesh_vertices_4]
    
        # # Create a 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
    
        # # Scatter plot of the vertices
        # ax.scatter(x, y, z, c='blue', marker='o', s=10, label='Mesh Vertices')

        # # Limit the Z-axis from 0 to 0.2
        # ax.set_zlim(-0.1, 0.2)

        # # Set labels and title
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D Plot of Mesh Vertices')
    
        # # Add legend
        # ax.legend()
    
        # # Show grid and plot
        # ax.grid(True)
        # plt.savefig('3D_Plot_Mesh_Vertices_4.png')

        # exit()

        # --------------------------
        # Final Message
        # --------------------------
        # Log or print a styled message indicating the cloth model generation is complete
        styled_message("Cloth t_shirt l 0.5 model generated.", fg_color="33", bg_color="44")

    def generate_t_shirt_l2_model(self, texture_dir = None, shear=40, bend=40, scale=(1, 1, 1), centralize=True):
        """
        Generates the cloth model and prepares the cloth object with necessary modifiers for simulation.
        This method imports a T-shirt model, applies transformations, sets up materials, 
        and configures cloth simulation physics.
        
        Parameters:
        - shear (int): Shear stiffness for the cloth simulation.
        - bend (int): Bending stiffness for the cloth simulation.
        - scale (tuple): Scaling factors for the cloth object.
        - centralize (bool): Whether to centralize the object in the scene.
        """

        # --------------------------
        # Import and Prepare the Cloth Object
        # --------------------------
        # Import the cloth OBJ file (ensure the file path is provided in `obj_file_path`)
        bpy.ops.wm.obj_import(filepath=self.obj_file_path)
    
        # # Get the imported cloth object and rename it to "T_Shirt_l2"
        self.cloth = bpy.context.selected_objects[0]
        self.cloth.name = 'T_Shirt_l2'
    
        # Convert the mesh to NumPy arrays (initial state on the XZ plane)
        mesh_vertices_initial, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # Store the initial vertices in the dictionary
        self.mesh_vertices = {'mesh_vertices_initial': mesh_vertices_initial}
    
        # Ensure the imported cloth object is active and in Object Mode
        bpy.context.view_layer.objects.active = self.cloth
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.cloth.show_axis = True   

        # --------------------------
        # Step 1: Apply rotation and scale to ensure consistent axes
        # --------------------------
        # Apply transformations to align the mesh with the XY plane
        # This rotates the mesh to align with the XY plane and applies scaling
        # mesh_vertices*Rotation_matrix
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after setting the origin
        mesh_vertices_1, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_1_rotated_scaled'] = mesh_vertices_1
        
        # # --------------------------
        # # Step 2: Set the object's origin to the geometry center
        # # --------------------------
        # # Move the object's origin to the center of its mesh geometry
        # # This step aligns the pivot point with the mesh center for easier transformations
        # # object's origin to align with the mesh's geometry (e.g., center of the mesh), 
        # # you can move the origin using: 
        # # shifts the object origin to the center of the mesh and update self.cloth.location
        # # Performs mesh_vertices_initial-np.array(self.cloth.location) and
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
        # # Convert the mesh to NumPy arrays after applying rotation and scale
        mesh_vertices_2, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_2_origin'] = mesh_vertices_2
        
        # # --------------------------
        # # Step 3: Update the object's origin and scale 
        # # --------------------------
        self.cloth.location = (0, 0, self.baseZ)  # Move the object along the Z-axis
        self.cloth.rotation_euler = (0, 0, -np.pi/2) # rotate the object around z axis by 90 degree
        self.cloth.scale = scale  # Apply scaling to the object
        
        # # --------------------------
        # # Step 4: Translates the mesh according to updated self.cloth.location  
        # # --------------------------
        # # This translates the mesh according to updated self.cloth.location 
        # # and since self.cloth.location is center of mesh so the entire mesh shifts
        # # so that the center of the mesh is at the origin
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after applying location changes
        mesh_vertices_3, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_3'] = mesh_vertices_3
    
        # # Ensure the object is active after transformation
        bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Step 5: Assign Material and Enable Smooth Shading
        # # --------------------------
        # # Enable smooth shading for better appearance
        bpy.ops.object.shade_smooth()
    
        if texture_dir is not None:
            bsdf_directory = os.path.join(texture_dir, "bsdf_info")
            normal_filename = self.select_random_folder_and_find_normal_file(bsdf_directory)
            self.pattern(self.cloth, normal_filename)
    
        # # --------------------------
        # # Step 6: Set Up Modifiers for Simulation
        # # --------------------------
        # # Add Vertex Weight Mix Modifier for weight manipulation
        vw_mix_mod = self.cloth.modifiers.new(name='VertexWeightMix', type='VERTEX_WEIGHT_MIX')
    
        # # Add Hook Modifiers for manipulating specific vertices
        hook_mod1 = self.cloth.modifiers.new(name='Hook1', type='HOOK')
        if self.multi:  # If multiple hooks are enabled, create another Hook modifier
            hook_mod2 = self.cloth.modifiers.new(name='Hook2', type='HOOK')
    
        # Add the Cloth physics modifier to enable cloth simulation
        cloth_mod = self.cloth.modifiers.new(name='Cloth', type='CLOTH')
        
    
        # # Convert the mesh to NumPy arrays after adding modifiers
        mesh_vertices_4, self.edges, faces, self.faces = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the final vertices in the dictionary
        self.mesh_vertices['mesh_vertices_4'] = mesh_vertices_4
        self.meshes = mesh_vertices_4
        
        # --------------------------
        # Configure Cloth Modifier Settings
        # --------------------------
        # Set physical properties for the cloth simulation
        # Usage in the Cloth Modifier
        cloth_mod.settings.mass = CLOTH_SETTINGS["mass"]
        cloth_mod.settings.air_damping = CLOTH_SETTINGS["air_damping"]

        cloth_mod.settings.tension_stiffness = CLOTH_SETTINGS["tension_stiffness"]
        cloth_mod.settings.compression_stiffness = CLOTH_SETTINGS["compression_stiffness"]
        cloth_mod.settings.shear_stiffness = CLOTH_SETTINGS["shear_stiffness"]
        cloth_mod.settings.bending_stiffness = CLOTH_SETTINGS["bending_stiffness"]

        cloth_mod.settings.tension_damping = CLOTH_SETTINGS["tension_damping"]
        cloth_mod.settings.compression_damping = CLOTH_SETTINGS["compression_damping"]
        cloth_mod.settings.shear_damping = CLOTH_SETTINGS["shear_damping"]
        cloth_mod.settings.bending_damping = CLOTH_SETTINGS["bending_damping"]

        cloth_mod.collision_settings.use_self_collision = CLOTH_SETTINGS["use_self_collision"]
        cloth_mod.collision_settings.self_friction = CLOTH_SETTINGS["self_friction"]
        cloth_mod.collision_settings.self_distance_min = CLOTH_SETTINGS["self_distance_min"]

        # Clear any existing animation data for the cloth
        self.cloth.animation_data_clear()
        # Clear any existing vertex groups
        self.cloth.vertex_groups.clear()  
        # ensure active
        bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Using for debugging
        # # --------------------------
        # # Extract x, y, z coordinates from the vertices
        # x = [v[0] for v in mesh_vertices_4]
        # y = [v[1] for v in mesh_vertices_4]
        # z = [v[2] for v in mesh_vertices_4]
    
        # # Create a 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
    
        # # Scatter plot of the vertices
        # ax.scatter(x, y, z, c='blue', marker='o', s=10, label='Mesh Vertices')

        # # Limit the Z-axis from 0 to 0.2
        # ax.set_zlim(-0.1, 0.2)

        # # Set labels and title
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D Plot of Mesh Vertices')
    
        # # Add legend
        # ax.legend()
    
        # # Show grid and plot
        # ax.grid(True)
        # plt.savefig('3D_Plot_Mesh_Vertices_4.png')

        # exit()

        # --------------------------
        # Final Message
        # --------------------------
        # Log or print a styled message indicating the cloth model generation is complete
        styled_message("Cloth t_shirt l2 model generated.", fg_color="33", bg_color="44")

    def generate_t_shirt_l3_model(self, texture_dir = None, shear=40, bend=40, scale=(1, 1, 1), centralize=True):
        """
        Generates the cloth model and prepares the cloth object with necessary modifiers for simulation.
        This method imports a T-shirt model, applies transformations, sets up materials, 
        and configures cloth simulation physics.
        
        Parameters:
        - shear (int): Shear stiffness for the cloth simulation.
        - bend (int): Bending stiffness for the cloth simulation.
        - scale (tuple): Scaling factors for the cloth object.
        - centralize (bool): Whether to centralize the object in the scene.
        """
    
        # --------------------------
        # Create the Cloth Material
        # --------------------------
        # Create a new material for the cloth object with color
        # cloth_material = bpy.data.materials.new(name='Cloth_Material')
        # cloth_material.diffuse_color = self.cloth_color
    
        # --------------------------
        # Import and Prepare the Cloth Object
        # --------------------------
        # Import the cloth OBJ file 
        bpy.ops.wm.obj_import(filepath=self.obj_file_path)
    
        # # Get the imported cloth object and rename it to "T_Shirt_l3"
        self.cloth = bpy.context.selected_objects[0]
        self.cloth.name = 'T_Shirt_l3'
    
        # Convert the mesh to NumPy arrays (initial state on the XZ plane)
        mesh_vertices_initial, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # Store the initial vertices in the dictionary
        self.mesh_vertices = {'mesh_vertices_initial': mesh_vertices_initial}
    
        # Ensure the imported cloth object is active and in Object Mode
        bpy.context.view_layer.objects.active = self.cloth
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.cloth.show_axis = True   

        # --------------------------
        # Step 1: Apply rotation and scale to ensure consistent axes
        # --------------------------
        # Apply transformations to align the mesh with the XY plane
        # This rotates the mesh to align with the XY plane and applies scaling
        # mesh_vertices*Rotation_matrix
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after setting the origin
        mesh_vertices_1, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_1_rotated_scaled'] = mesh_vertices_1
        
        # # --------------------------
        # # Step 2: Set the object's origin to the geometry center
        # # --------------------------
        # # Move the object's origin to the center of its mesh geometry
        # # This step aligns the pivot point with the mesh center for easier transformations
        # # object's origin to align with the mesh's geometry (e.g., center of the mesh), 
        # # you can move the origin using: 
        # # shifts the object origin to the center of the mesh and update self.cloth.location
        # # Performs mesh_vertices_initial-np.array(self.cloth.location) and
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
        # # Convert the mesh to NumPy arrays after applying rotation and scale
        mesh_vertices_2, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_2_origin'] = mesh_vertices_2
        
        # # --------------------------
        # # Step 3: Update the object's origin and scale 
        # # --------------------------
        self.cloth.location = (0, 0, self.baseZ)  # Move the object along the Z-axis
        self.cloth.rotation_euler = (0, 0, -np.pi/2) # rotate the object around z axis by 90 degree
        self.cloth.scale = scale  # Apply scaling to the object
        
        # # --------------------------
        # # Step 4: Translates the mesh according to updated self.cloth.location  
        # # --------------------------
        # # This translates the mesh according to updated self.cloth.location 
        # # and since self.cloth.location is center of mesh so the entire mesh shifts
        # # so that the center of the mesh is at the origin
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after applying location changes
        mesh_vertices_3, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_3'] = mesh_vertices_3
    
        # # Ensure the object is active after transformation
        bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Step 5: Assign Material and Enable Smooth Shading
        # # --------------------------
        # # Enable smooth shading for better appearance
        bpy.ops.object.shade_smooth()
    
        if texture_dir is not None:
            bsdf_directory = os.path.join(texture_dir, "bsdf_info")
            normal_filename = self.select_random_folder_and_find_normal_file(bsdf_directory)
            self.pattern(self.cloth, normal_filename)

        # # --------------------------
        # # Step 6: Set Up Modifiers for Simulation
        # # --------------------------
        # # Add Vertex Weight Mix Modifier for weight manipulation
        vw_mix_mod = self.cloth.modifiers.new(name='VertexWeightMix', type='VERTEX_WEIGHT_MIX')
    
        # # Add Hook Modifiers for manipulating specific vertices
        hook_mod1 = self.cloth.modifiers.new(name='Hook1', type='HOOK')
        if self.multi:  # If multiple hooks are enabled, create another Hook modifier
            hook_mod2 = self.cloth.modifiers.new(name='Hook2', type='HOOK')
    
        # Add the Cloth physics modifier to enable cloth simulation
        cloth_mod = self.cloth.modifiers.new(name='Cloth', type='CLOTH')
        
    
        # # Convert the mesh to NumPy arrays after adding modifiers
        mesh_vertices_4, self.edges, faces, self.faces = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the final vertices in the dictionary
        self.mesh_vertices['mesh_vertices_4'] = mesh_vertices_4

        # --------------------------
        # Configure Cloth Modifier Settings
        # --------------------------
        # Set physical properties for the cloth simulation
        # Usage in the Cloth Modifier
        cloth_mod.settings.mass = CLOTH_SETTINGS_l3["mass"]
        cloth_mod.settings.air_damping = CLOTH_SETTINGS_l3["air_damping"]

        cloth_mod.settings.tension_stiffness = CLOTH_SETTINGS_l3["tension_stiffness"]
        cloth_mod.settings.compression_stiffness = CLOTH_SETTINGS_l3["compression_stiffness"]
        cloth_mod.settings.shear_stiffness = CLOTH_SETTINGS_l3["shear_stiffness"]
        cloth_mod.settings.bending_stiffness = CLOTH_SETTINGS_l3["bending_stiffness"]

        cloth_mod.settings.tension_damping = CLOTH_SETTINGS_l3["tension_damping"]
        cloth_mod.settings.compression_damping = CLOTH_SETTINGS_l3["compression_damping"]
        cloth_mod.settings.shear_damping = CLOTH_SETTINGS_l3["shear_damping"]
        cloth_mod.settings.bending_damping = CLOTH_SETTINGS_l3["bending_damping"]

        cloth_mod.collision_settings.use_self_collision = CLOTH_SETTINGS_l3["use_self_collision"]
        cloth_mod.collision_settings.self_friction = CLOTH_SETTINGS_l3["self_friction"]
        cloth_mod.collision_settings.self_distance_min = CLOTH_SETTINGS_l3["self_distance_min"]

        # Clear any existing animation data for the cloth
        self.cloth.animation_data_clear()
        # Clear any existing vertex groups
        self.cloth.vertex_groups.clear()  
        # ensure active
        bpy.context.view_layer.objects.active = self.cloth

        # # --------------------------
        # # Using for debugging
        # # --------------------------
        # # Extract x, y, z coordinates from the vertices
        # x = [v[0] for v in mesh_vertices_4]
        # y = [v[1] for v in mesh_vertices_4]
        # z = [v[2] for v in mesh_vertices_4]
    
        # # Create a 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
    
        # # Scatter plot of the vertices
        # ax.scatter(x, y, z, c='blue', marker='o', s=10, label='Mesh Vertices')

        # # Limit the Z-axis from 0 to 0.2
        # ax.set_zlim(-0.1, 0.2)

        # # Set labels and title
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D Plot of Mesh Vertices')
    
        # # Add legend
        # ax.legend()
    
        # # Show grid and plot
        # ax.grid(True)
        # plt.savefig('Mesh_Vertices_init.png')

        # exit()
    
        # --------------------------
        # Final Message
        # --------------------------
        # Log or print a styled message indicating the cloth model generation is complete
        styled_message("Cloth t_shirt l2 model generated.", fg_color="33", bg_color="44")

    def generate_t_shirt_l4_model(self, texture_dir = None, shear=40, bend=40, scale=(1, 1, 1), centralize=True):
        """
        Generates the cloth model and prepares the cloth object with necessary modifiers for simulation.
        This method imports a T-shirt model, applies transformations, sets up materials, 
        and configures cloth simulation physics.
        
        Parameters:
        - shear (int): Shear stiffness for the cloth simulation.
        - bend (int): Bending stiffness for the cloth simulation.
        - scale (tuple): Scaling factors for the cloth object.
        - centralize (bool): Whether to centralize the object in the scene.
        """
    
        # --------------------------
        # Import and Prepare the Cloth Object
        # --------------------------
        # Import the cloth OBJ file
        bpy.ops.wm.obj_import(filepath=self.obj_file_path)
    
        # Get the imported cloth object and rename it to "T_Shirt_l4"
        self.cloth = bpy.context.selected_objects[0]
        self.cloth.name = 'T_Shirt_l4'
    
        # Convert the mesh to NumPy arrays (initial state on the XZ plane)
        mesh_vertices_initial, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # Store the initial vertices in the dictionary
        self.mesh_vertices = {'mesh_vertices_initial': mesh_vertices_initial}
    
        # Ensure the imported cloth object is active and in Object Mode
        bpy.context.view_layer.objects.active = self.cloth
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.cloth.show_axis = True   

        # --------------------------
        # Step 1: Apply rotation and scale to ensure consistent axes
        # --------------------------
        # Apply transformations to align the mesh with the XY plane
        # This rotates the mesh to align with the XY plane and applies scaling
        # mesh_vertices*Rotation_matrix
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after setting the origin
        mesh_vertices_1, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_1_rotated_scaled'] = mesh_vertices_1
        
        # # --------------------------
        # # Step 2: Set the object's origin to the geometry center
        # # --------------------------
        # # Move the object's origin to the center of its mesh geometry
        # # This step aligns the pivot point with the mesh center for easier transformations
        # # object's origin to align with the mesh's geometry (e.g., center of the mesh), 
        # # you can move the origin using: 
        # # shifts the object origin to the center of the mesh and update self.cloth.location
        # # Performs mesh_vertices_initial-np.array(self.cloth.location) and
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
        # # Convert the mesh to NumPy arrays after applying rotation and scale
        mesh_vertices_2, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_2_origin'] = mesh_vertices_2
        
        # # --------------------------
        # # Step 3: Update the object's origin and scale 
        # # --------------------------
        self.cloth.location = (0, 0, self.baseZ)  # Move the object along the Z-axis
        self.cloth.rotation_euler = (0, 0, -np.pi/2) # rotate the object around z axis by 90 degree
        self.cloth.scale = scale  # Apply scaling to the object
        
        # # --------------------------
        # # Step 4: Translates the mesh according to updated self.cloth.location  
        # # --------------------------
        # # This translates the mesh according to updated self.cloth.location 
        # # and since self.cloth.location is center of mesh so the entire mesh shifts
        # # so that the center of the mesh is at the origin
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # # Convert the mesh to NumPy arrays after applying location changes
        mesh_vertices_3, edges, faces, faces_tri = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the updated vertices in the dictionary
        self.mesh_vertices['mesh_vertices_3'] = mesh_vertices_3
    
        # # Ensure the object is active after transformation
        # bpy.context.view_layer.objects.active = self.cloth
    
        # # --------------------------
        # # Step 5: Assign Material and Enable Smooth Shading
        # # --------------------------
        # # Set the material of the cloth object to the newly created cloth material
        
        # # Enable smooth shading for better appearance
        bpy.ops.object.shade_smooth()
    
        # # Assign the cloth material to the object
        # if self.cloth.data.materials:
        #     self.cloth.data.materials[0] = cloth_material  # Replace the existing material
        # else:
        #     self.cloth.data.materials.append(cloth_material)  # Add the material if none exist

        if texture_dir is not None:
            bsdf_directory = os.path.join(texture_dir, "bsdf_info")
            normal_filename = self.select_random_folder_and_find_normal_file(bsdf_directory)
            self.pattern(self.cloth, normal_filename)

        # # --------------------------
        # # Step 6: Set Up Modifiers for Simulation
        # # --------------------------
        # # Add Vertex Weight Mix Modifier for weight manipulation
        vw_mix_mod = self.cloth.modifiers.new(name='VertexWeightMix', type='VERTEX_WEIGHT_MIX')
    
        # # Add Hook Modifiers for manipulating specific vertices
        hook_mod1 = self.cloth.modifiers.new(name='Hook1', type='HOOK')
        if self.multi:  # If multiple hooks are enabled, create another Hook modifier
            hook_mod2 = self.cloth.modifiers.new(name='Hook2', type='HOOK')
    
        # Add the Cloth physics modifier to enable cloth simulation
        cloth_mod = self.cloth.modifiers.new(name='Cloth', type='CLOTH')
        
        # # Convert the mesh to NumPy arrays after adding modifiers
        mesh_vertices_4, self.edges, faces, self.faces = self.convert_mesh_to_np(self.cloth.data)
    
        # # Store the final vertices in the dictionary
        self.mesh_vertices['mesh_vertices_4'] = mesh_vertices_4
        self.meshes = mesh_vertices_4

        # --------------------------
        # Configure Cloth Modifier Settings
        # --------------------------
        # Set physical properties for the cloth simulation
        # Usage in the Cloth Modifier
        cloth_mod.settings.mass = CLOTH_SETTINGS["mass"]
        cloth_mod.settings.air_damping = CLOTH_SETTINGS["air_damping"]

        cloth_mod.settings.tension_stiffness = CLOTH_SETTINGS["tension_stiffness"]
        cloth_mod.settings.compression_stiffness = CLOTH_SETTINGS["compression_stiffness"]
        cloth_mod.settings.shear_stiffness = CLOTH_SETTINGS["shear_stiffness"]
        cloth_mod.settings.bending_stiffness = CLOTH_SETTINGS["bending_stiffness"]

        cloth_mod.settings.tension_damping = CLOTH_SETTINGS["tension_damping"]
        cloth_mod.settings.compression_damping = CLOTH_SETTINGS["compression_damping"]
        cloth_mod.settings.shear_damping = CLOTH_SETTINGS["shear_damping"]
        cloth_mod.settings.bending_damping = CLOTH_SETTINGS["bending_damping"]

        cloth_mod.collision_settings.use_self_collision = CLOTH_SETTINGS["use_self_collision"]
        cloth_mod.collision_settings.self_friction = CLOTH_SETTINGS["self_friction"]
        cloth_mod.collision_settings.self_distance_min = CLOTH_SETTINGS["self_distance_min"]

        # Clear any existing animation data for the cloth
        self.cloth.animation_data_clear()
        # Clear any existing vertex groups
        self.cloth.vertex_groups.clear()  
        # ensure active
        bpy.context.view_layer.objects.active = self.cloth
        
        # # --------------------------
        # # Using for debugging
        # # --------------------------
        # # Extract x, y, z coordinates from the vertices
        # x = [v[0] for v in mesh_vertices_4]
        # y = [v[1] for v in mesh_vertices_4]
        # z = [v[2] for v in mesh_vertices_4]
    
        # # Create a 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
    
        # # Scatter plot of the vertices
        # ax.scatter(x, y, z, c='blue', marker='o', s=10, label='Mesh Vertices')

        # # Limit the Z-axis from 0 to 0.2
        # ax.set_zlim(-0.1, 0.2)

        # # Set labels and title
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D Plot of Mesh Vertices')
    
        # # Add legend
        # ax.legend()
    
        # # Show grid and plot
        # ax.grid(True)
        # plt.savefig('3D_Plot_Mesh_Vertices_4.png')

        # exit()
    
        # --------------------------
        # Final Message
        # --------------------------
        # Log or print a styled message indicating the cloth model generation is complete
        styled_message("Cloth t_shirt l2 model generated.", fg_color="33", bg_color="44")

        return 

    def select_random_folder_and_find_normal_file(self, directory):
        """
        Selects a random folder within the specified directory and returns the path
        of a file whose name ends with '_NormalGL.png' or '_NormalGL.jpg'.
        If no such file is found, returns None.
        """
        for i in range(10):
            all_folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            available_folders = [d for d in all_folders if d not in self.selected_folders]
            if not available_folders:
                print("All folders have been selected. Resetting selection list.")
                available_folders = all_folders
                self.selected_folders = []

            selected_folder = random.choice(available_folders)
            self.selected_folders.append(selected_folder)
            selected_folder_path = os.path.join(directory, selected_folder)
            
            for file in os.listdir(selected_folder_path):
                if file.endswith("_NormalGL.png") or file.endswith("_NormalGL.jpg"):
                    normal_file_path = os.path.join(selected_folder_path, file)
                    return normal_file_path
        
        return None
    
    def pattern(self, obj, normal_filename=None, normal_mapping_scale=(1.0, 1.0)):
        '''Add texture to the cloth'''

        # 1. Create a new Material with nodes enabled
        mat = bpy.data.materials.new(name="MaterialTexture")
        mat.use_nodes = True
        
        # 2. Grab the existing Principled BSDF node (the main shader in the material)
        bsdf = mat.node_tree.nodes["Principled BSDF"]

        # # 3. Add a Texture Coordinate node (gives UV coordinates among others)
        # tex_coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')

        # # 4. Create another Mapping node specifically for the normal map
        # normal_mapping = mat.node_tree.nodes.new('ShaderNodeMapping')
        # normal_mapping.inputs['Scale'].default_value = (normal_mapping_scale[0], normal_mapping_scale[1], 0)

        # # 5. Add and configure a Normal Map texture node
        # normal = mat.node_tree.nodes.new('ShaderNodeTexImage')
        # print(f"normal_filename name: {normal_filename}")
        # normal.image = bpy.data.images.load(normal_filename)
        
        # #   - Tell Blender that this is a “Non-Color” data texture (important for normal maps)
        # #   - try to comment this line and see the results
        # normal.image.colorspace_settings.name = 'Non-Color'

        # # 6. Add a ShaderNodeNormalMap node and set its Strength (input[0]). This controls how strong the normal effect is.
        # normal_node = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        # normal_node.inputs[0].default_value = 2
        # #    Use a random normal strength between 0 and 10 (for variation)
        # # normal_node.inputs[0].default_value = random.uniform(0, 10)

        # # 7. Link the nodes together to form a texture pipeline
        # #  - Link the UV output into that Mapping node’s Vector input
        # mat.node_tree.links.new(normal_mapping.inputs['Vector'], tex_coord.outputs['UV'])
        # #  - Link the normal texture’s Vector input to the normal map’s Mapping output
        # mat.node_tree.links.new(normal.inputs['Vector'], normal_mapping.outputs['Vector'])
        # #   - Link the normal texture’s Color output to the NormalMap node’s Color input
        # mat.node_tree.links.new(normal_node.inputs['Color'], normal.outputs['Color'])
        # #   - Link NormalMap node’s Normal output to the Principled BSDF’s Normal input
        # mat.node_tree.links.new(bsdf.inputs['Normal'], normal_node.outputs['Normal'])

        # Add based color to the cloth
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = self.cloth_color

        # change the cloth materials
        # Metallic used to control the amount of light that is reflected by the surface
        mat.node_tree.nodes["Principled BSDF"].inputs[1].default_value = 0.795 
        # Roughness is used to control the amount of light that is scattered by the surface  
        mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.785
        # IOR is used to control the amount of light that is refracted by the surface 
        mat.node_tree.nodes["Principled BSDF"].inputs[3].default_value = 1.5
        # Alpha is used to control the transparency of the surface
        mat.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 0.96

        # 9. Finally, assign the created material to the object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def convert_mesh_to_np(self, mesh):
        """
        Converts a Blender mesh to a NumPy array.

        Parameters:
        mesh (bpy.types.Mesh): The mesh to convert.

        Returns:
        np.ndarray: A NumPy array of vertex coordinates.
        """
        n_vtx = len(mesh.vertices)
        np_mesh = np.empty((n_vtx, 3))
        for i in range(n_vtx):
            np_mesh[i] = np.array(mesh.vertices[i].co)

        # Extract edges (each edge is a tuple of two vertex indices)
        edges = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]
        np_edge = np.array(edges)        # Convert edges to a NumPy array

        # Extract faces (polygons)
        # Shape: (F, 3) or (F, 4) for triangles/quads
        np_face = np.array([list(p.vertices) for p in mesh.polygons])

        np_face_tri = self.triangulate_faces(np_face)

        print(f"{n_vtx} vertices in mesh, converted to array of shape {np_mesh.shape}")
        return np_mesh, np_edge, np_face, np_face_tri
    
    def triangulate_faces(self, faces):
        """
        Triangulates polygonal faces into triangular faces.

        Parameters:
        faces (np.ndarray): Array of face indices, shape (F, P), where P is variable.

        Returns:
        np.ndarray: Array of triangular face indices, shape (T, 3).
        """
        import numpy as np

        triangulated_faces = []
        for face in faces:
            if len(face) == 3:
                # Already a triangle
                triangulated_faces.append(face)
            elif len(face) > 3:
                # Split the polygon into triangles
                for i in range(1, len(face) - 1):
                    triangulated_faces.append([face[0], face[i], face[i + 1]])

        return np.array(triangulated_faces)
    
    def save_as_obj(self, vertices, faces, output_path):
        """
        Write a minimal OBJ file using only vertices (v) and faces (f).
        
        Parameters:
            vertices: Nx3 float array of vertex positions
            faces: MxK int array of vertex indices (0-based)
            output_path: path to the destination .obj file
        """
        with open(output_path, 'w') as f:
            f.write("# Minimal OBJ file\n")
            
            # Write vertices
            for vx, vy, vz in vertices:
                f.write(f"v {vx} {vy} {vz}\n")
            
            # Write faces (convert from 0-based to 1-based indexing)
            for face in faces:
                face_indices = [str(idx + 1) for idx in face]  # +1 for OBJ
                f.write("f " + " ".join(face_indices) + "\n")

        print(f"OBJ saved to: {output_path}")
    
    def find_keypoints_lists_t_shirt(self, np_mesh, plot=True):
        """
        Find the keypoints for a T-Shirt mesh.
        """
        # Extract x, y, z coordinates from the vertices
        x = np_mesh[:, 0]
        y = np_mesh[:, 1]
        z = np_mesh[:, 2]

        # Filter vertices within the x limits
        x_min = np.min(x)
        x_max = np.max(x)

        arm_pit_left = np_mesh[(np_mesh[:, 0] <= x_min)][0]
        arm_pit_right = np_mesh[(np_mesh[:, 0] >= x_max)][0]

        # Generate the indices for the vertices
        arm_pit_left_indices = np.where(np_mesh[:, 0] <= x_min)[0]
        arm_pit_right_indices = np.where(np_mesh[:, 0] >= x_max)[0]

        # Filter vertices within the y limits
        y_min = np.min(y)
        y_max = np.max(y)

        neckline_left  = np_mesh[(np_mesh[:, 1] >= y_max) & (np_mesh[:, 0] <= (x_min + x_max) / 2)][0]
        neckline_right = np_mesh[(np_mesh[:, 1] >= y_max) & (np_mesh[:, 0] >= (x_min + x_max) / 2)][0]

        neckline_left_indices = np.where((np_mesh[:, 1] >= y_max) & (np_mesh[:, 0] <= (x_min + x_max)/2))[0]
        neckline_right_indices = np.where((np_mesh[:, 1] >= y_max) & (np_mesh[:, 0] >= (x_min + x_max)/2))[0]

        # Find mesh vertices with y coordinates greater than or equal to center_line_y
        center_line_y = arm_pit_left[1]+neckline_left[1]/2 
        shoulder_left_region  = np_mesh[(np_mesh[:, 1] >= center_line_y) & (np_mesh[:, 1] < y_max)& (np_mesh[:, 0] <= (x_min + x_max)/2)]
        shoulder_left = shoulder_left_region[(shoulder_left_region[:, 0] <= np.min(shoulder_left_region[:,0]))][0]

        shoulder_right_region = np_mesh[(np_mesh[:, 1] >= center_line_y) & (np_mesh[:, 1] < y_max)& (np_mesh[:, 0] >= (x_min + x_max)/2)]
        shoulder_right = shoulder_right_region[(shoulder_right_region[:, 0] >= np.max(shoulder_right_region[:,0]))][0]

        shoulder_left_indices  = np.where((np_mesh[:, 0] == shoulder_left[0]) & (np_mesh[:, 1] == shoulder_left[1]) & (np_mesh[:, 2] == shoulder_left[2]))[0]
        shoulder_right_indices = np.where((np_mesh[:, 0] == shoulder_right[0]) & (np_mesh[:, 1] == shoulder_right[1]) & (np_mesh[:, 2] == shoulder_right[2]))[0]
        
        # Bottom hem points
        # Determine vertices lying at y_min
        y_min_vertices = np_mesh[np_mesh[:, 1] == y_min]

        # Add the bottom hem vertices to the keypoints
        # Determine vertices with y < 0
        vertices_below_y_threshold = np_mesh[np_mesh[:, 1] < -0.2]
        # Find vertices with minimum and maximum x values within those vertices
        bottom_hem_left  = vertices_below_y_threshold[np.argmin(vertices_below_y_threshold[:, 0])]
        bottom_hem_right = vertices_below_y_threshold[np.argmax(vertices_below_y_threshold[:, 0])]

        bottom_hem_left_indices = np.where((np_mesh[:, 0] == bottom_hem_left[0]) & (np_mesh[:, 1] == bottom_hem_left[1]) & (np_mesh[:, 2] == bottom_hem_left[2]))[0]
        bottom_hem_right_indices = np.where((np_mesh[:, 0] == bottom_hem_right[0]) & (np_mesh[:, 1] == bottom_hem_right[1]) & (np_mesh[:, 2] == bottom_hem_right[2]))[0]


        # Add the template mesh center keypoint
        # Calculate the bounding box
        bounding_box_min = np.min(np_mesh, axis=0)
        bounding_box_max = np.max(np_mesh, axis=0)

        # Calculate the center of the bounding box
        bounding_box_center = (bounding_box_min + bounding_box_max) / 2
        # Find the closest vertex to the bounding box center
        mesh_vertices_distances_from_bb_center = np.linalg.norm(np_mesh - bounding_box_center, axis=1)
        # Find the index of the vertex closest to the bounding box center      
        # Add the bounding box center to the keypoints
        center_indices = [np.argmin(np.linalg.norm(np_mesh - bounding_box_center, axis=1))]
        center         = np_mesh[center_indices[0]]


        # Step 1: Identify key points
        neck_line_depth   = 0.1932

        # Store key points in a list
        self.keypoints_details = [
            # {"neckline_left": neckline_left},
            {"arm_pit_left":arm_pit_left},
            {"arm_pit_right":arm_pit_right},

            {"neckline_left":neckline_left},
            {"neckline_right":neckline_right},

            {"shoulder_left":shoulder_left},
            {"shoulder_right":shoulder_right},

            {"bottom_hem_left": bottom_hem_left},
            {"bottom_hem_right": bottom_hem_right},

            {"center": center},
            
            {"arm_pit_left_indices": arm_pit_left_indices},
            {"arm_pit_right_indices": arm_pit_right_indices},

            {"neckline_left_indices": neckline_left_indices},
            {"neckline_right_indices": neckline_right_indices},

            {"shoulder_left_indices": shoulder_left_indices},
            {"shoulder_right_indices": shoulder_right_indices},

            {"bottom_hem_left_indices": bottom_hem_left_indices},
            {"bottom_hem_right_indices": bottom_hem_right_indices},
            
            {"center_indices": center_indices}
        ]

        styled_message(f"T-Shirt Key Points Identified: {self.keypoints_details}", fg_color="33", bg_color="44")

        self.keypoints = np.array([arm_pit_left_indices[0], arm_pit_right_indices[0], 
                            neckline_left_indices[0], neckline_right_indices[0], 
                            shoulder_left_indices[0], shoulder_right_indices[0],
                            bottom_hem_left_indices[0], bottom_hem_right_indices[0], 
                            center_indices[0]])

        # if plot:
        #     # Plot the mesh and highlight key points
        #     fig = plt.figure(figsize=(10, 8))
        #     ax = fig.add_subplot(111, projection='3d')

        #     # Plot the mesh vertices
        #     ax.scatter(np_mesh[:, 0], np_mesh[:, 1], np_mesh[:, 2],
        #                color='gray', s=10, label="Mesh Vertices")

        #     # # Plot the mesh faces if available
        #     # if faces is not None:
        #     #     mesh_faces = [[np_mesh[vertex] for vertex in face] for face in faces]
        #     #     ax.add_collection3d(Poly3DCollection(mesh_faces, facecolors='cyan', linewidths=0.5, edgecolors='gray', alpha=0.5))

        #     ax.scatter(*neckline_left, color='red', s=100,
        #                label=" neckline_left", marker="o")
        #     ax.scatter(*neckline_right, color='blue', s=100,
        #                   label="neckline_right", marker="o")
        #     ax.scatter(*shoulder_right, color='blue', s=100,
        #                label="shoulder_right", marker="o")
        #     ax.scatter(*shoulder_left, color='cyan', s=100,
        #                label="shoulder_left", marker="o")
        #     ax.scatter(*arm_pit_left, color='green', s=100,
        #                label="arm_pit_left", marker="o")
        #     ax.scatter(*arm_pit_right, color='orange', s=100,
        #                label="arm_pit_right", marker="o")
        #     ax.scatter(*bottom_hem_left, color='purple', s=100,
        #                label="bottom_hem_left", marker="o")
        #     ax.scatter(*bottom_hem_right, color='black', s=100,
        #                label="bottom_hem_right", marker="o")
        #     ax.scatter(*center, color='yellow', s=100,
        #                label="center", marker="o")

        #     # Set labels and title
        #     ax.set_title(
        #         "T-Shirt Mesh with Highlighted Key Points", fontsize=16)
        #     ax.set_xlabel("X", fontsize=12)
        #     ax.set_ylabel("Y", fontsize=12)
        #     ax.set_zlabel("Z", fontsize=12)

        #     # set z range
        #     ax.set_zlim(-0.05, 0.1)

        #     # Add a legend
        #     ax.legend()

        #     # Adjust the view angle for better visualization
        #     ax.view_init(elev=20, azim=30)

        #     # Save the plot
        #     plt.savefig("t_shirt_keypoints.png")

        return self.keypoints
    
    def save_image(self, output_path, resolution_x=1080, resolution_y=1080, file_format='PNG', frame=0):
        """
        Render an image at a specific frame and save it to the specified path.
        """

        # scene = bpy.context.scene
        
        # scene.frame_set(frame) 
        # scene.render.resolution_x = resolution_x
        # scene.render.resolution_y = resolution_y
        # scene.render.image_settings.file_format = file_format

        # scene.render.filepath = output_path + ".png"
        bpy.ops.render.render(write_still=True)

    def save_state(self, ms=None, meshes=None, edges=None, faces=None, faces_tri=None,
                   vtx_picks=None, co_picks=None, moves=None, hand_loc=None, convert=False, as_init=False, save_file=True,
                   keypoints = None, overwrite = False):
        """
        Saves the simulation state to a pickle file.

        Parameters:
        ms (int): Manipulation step index.
        meshes (np.ndarray): Mesh data.
        vtx_picks (list): Vertex picks.
        co_picks (np.ndarray): Coordinate picks.
        moves (np.ndarray): Move vectors.
        convert (bool): Whether to convert meshes from Blender meshes to NumPy arrays.
        as_init (bool): Whether to save as the initial state.
        """
        # Use current state if parameters are not provided
        ms = ms if ms is not None else self.ms
        meshes = meshes if meshes is not None else self.meshes
        edges = edges if edges is not None else self.edges
        facess = faces if faces is not None else self.faces
        vtx_picks = vtx_picks if vtx_picks is not None else self.vtx_picks
        co_picks = co_picks if co_picks is not None else self.co_picks
        moves = moves if moves is not None else self.moves
        hand_loc = hand_loc if hand_loc is not None else self.hand_loc
        keypoints = keypoints if keypoints is not None else self.keypoints

        # Convert Blender meshes to numpy arrays if needed
        if convert and isinstance(meshes[0], bpy.types.Mesh):
            np_meshes, np_edges, np_faces, np_faces_tri = np.array(
                [self.convert_mesh_to_np(mesh) for mesh in meshes])
        else:
            np_meshes = meshes
            np_edges = edges
            np_faces = faces
            np_faces_tri = faces_tri
            np_keypoints = keypoints

        # Construct state dictionary
        state = {
            'ms': ms,
            'meshes': np_meshes,
            'edges': np_edges,
            'faces': np_faces,
            'faces_tri': np_faces_tri,
            'vtx_picks': vtx_picks,
            'co_picks': co_picks,
            'moves': moves,
            'hand_loc': hand_loc,
            'keypoints': np_keypoints
        }
        try:
            state_for_TRTM = {'ms': ms,
                              'mesh_pos': np_meshes[0],
                              'edge_idx': np_edges[0],
                              'faces_poly': np_faces[0],
                              'face_idx': np_faces_tri[0],
                              'image': 'To be added',
                              'vtx_picks': vtx_picks,
                              'co_picks': co_picks,
                              'moves': moves,
                              'hand_loc': hand_loc,
                              'image': 'Need to be added',
                              'grasp_vtx_idx': 'Need to be added',
                              'group_edge_idx' : 'Need to be added',
                              'keypoint_idx'  : np_keypoints,
                              }
        except:
            state_for_TRTM = {'ms': ms,
                              'mesh_pos': np_meshes[0],
                              'edge_idx': np_edges[0],
                              'faces_poly': None,
                              'face_idx': None,
                              'image': 'To be added',
                              'vtx_picks': vtx_picks,
                              'co_picks': co_picks,
                              'moves': moves,
                              'hand_loc': None,
                              'image': 'Need to be added',
                              'grasp_vtx_idx': 'Need to be added',
                              'group_edge_idx' : 'Need to be added',
                              'keypoint_idx'  :  np_keypoints,
                              }

        if save_file:
            # Save state as init.pickle or state.pickle
            if as_init:
                file_name = f"template_{self.cloth.name.lower()}.pickle"
                file_path = os.path.join(self.database_dir, file_name)
                
                # Check if the init file exists
                if not os.path.exists(file_path) or overwrite:
                    with open(file_path, mode='wb') as f:
                        pickle.dump(state_for_TRTM, f)
                    styled_message(f"Init state saved to {file_path}")
                else:
                    styled_message(f"{file_name} exists")
                    
                # Ensure the folder exists (create it if it doesn't)
                start_states_folder = os.path.join(self.database_dir, "start_states")
                os.makedirs(start_states_folder, exist_ok=True)
                
                # Define the file path
                file_path = os.path.join(start_states_folder, "start_state.txt")
                
                # Write the array to the file
                mesh_pos_array = state_for_TRTM['mesh_pos']  # Assuming the array is 2601x3
                np.savetxt(file_path, mesh_pos_array, fmt="%.6f", comments="")
            else:
                file_name = f'state[pid{os.getpid()}].pickle'
                file_path = os.path.join(self.database_dir, file_name)
                with open(file_path, mode='wb') as f:
                    pickle.dump(state_for_TRTM, f)
                styled_message(f"State saved to {file_name}")

        return state
    
    def save_current_cloth_as_init(self, save_file=False, overwrite = False):
        """
        Saves the current cloth mesh as the initial state.
        """

        print('Saving current state as initial state...')
        np_mesh, np_edge, np_face, np_face_tri = self.convert_mesh_to_np(self.cloth.data)
        
        if self.cloth.name == 'T_Shirtv2':
            
            np_keypoint = self.find_keypoints_lists_t_shirt(np_mesh, plot = False)
            
            self.save_state(
            ms=0,
            meshes=np_mesh[None, ...],
            edges=np_edge[None, ...],
            keypoints = np_keypoint,
            faces=np_face[None, ...],
            faces_tri=np_face_tri[None, ...],
            vtx_picks=[],
            co_picks=[],
            moves=[],
            as_init=True,
            save_file=save_file,
            overwrite = overwrite
        )
        elif self.cloth.name == 'T_Shirt_l1':

            np_keypoint = self.find_keypoints_lists_t_shirt(np_mesh, plot = False)
            
            self.save_state(
            ms=0,
            meshes=np_mesh[None, ...],
            edges=np_edge[None, ...],
            keypoints = np_keypoint,
            faces=np_face[None, ...],
            faces_tri=np_face_tri[None, ...],
            vtx_picks=[],
            co_picks=[],
            moves=[],
            as_init=True,
            save_file=save_file,
            overwrite = overwrite
        )
        elif self.cloth.name == 'T_Shirt_l2':

            np_keypoint = self.find_keypoints_lists_t_shirt(np_mesh, plot = False)
            
            self.save_state(
            ms=0,
            meshes=np_mesh[None, ...],
            edges=np_edge[None, ...],
            keypoints = np_keypoint,
            faces=np_face[None, ...],
            faces_tri=np_face_tri[None, ...],
            vtx_picks=[],
            co_picks=[],
            moves=[],
            as_init=True,
            save_file=save_file,
            overwrite = overwrite
        )
        elif self.cloth.name == 'T_Shirt_l3':

            np_keypoint = self.find_keypoints_lists_t_shirt(np_mesh, plot = False)
            
            self.save_state(
            ms=0,
            meshes=np_mesh[None, ...],
            edges=np_edge[None, ...],
            keypoints = np_keypoint,
            faces=np_face[None, ...],
            faces_tri=np_face_tri[None, ...],
            vtx_picks=[],
            co_picks=[],
            moves=[],
            as_init=True,
            save_file=save_file,
            overwrite = overwrite
        )
        
        print('Saved initial state!')

class ClothSimulation(ClothBlenderModel):
    """
    Inherits from ClothBlenderModel which can handle the environment setup and cloth generation.
    This class is used to simulate the cloth manipulation and generate the dataset.
    """

    def __init__(self, obj_file_path, database_dir, cloth_style = None, multi = False, cam_move = False, unfold = False, cloth_model=None):
        """
        Initialize the ClothSimClass by calling the parent constructor.
        """
        super().__init__(obj_file_path, database_dir, cloth_style = cloth_style, multi = multi)
        self.unfold = unfold
        self.cam_move = cam_move
        self.cloth_model = cloth_model

        self.cam_final_loc = None 
        self.result_np_mesh = None

        self.lift_height_max = 0.2
        self.move_range = 1.5

    def select_topmost_and_neighbors(self, pick_list, grid_dimension):
        
        if len(pick_list) == 0:  # If the pick list is empty
            print("Pick list is empty!")
            return [], None
    
        # Get the z-coordinates of the vertices in the pick list
        pick_z_coords = self.meshes[pick_list, 2]
    
        # Find the index of the topmost vertex (highest z-value)
        topmost_index = np.argmax(pick_z_coords)
        topmost_vertex = pick_list[topmost_index]
        
        # Get the neighbors of the topmost vertex
        neighbors = [
            topmost_vertex,                        # The topmost vertex itself
            topmost_vertex + 1,                    # Right neighbor
            topmost_vertex - 1,                    # Left neighbor
            topmost_vertex + grid_dimension,       # Top neighbor
            topmost_vertex - grid_dimension        # Bottom neighbor
        ]
    
        # Filter neighbors: Keep only those in the pick_list
        neighbors = [n for n in neighbors if n in pick_list]
    
        return neighbors, topmost_vertex

    def find_pick_lists(self, co_picks, grasp_range=0.05, use_multi=False):

        # if self.cloth.name == "T_Shirt_l2" or "T_Shirt_l3" or "T_Shirt_l4":
        #     grasp_range = 0.15
        
        # Initialize pick lists
        self.pick_list1 = []
        
        # If multi-pick is enabled
        self.pick_list2 = []
        
        self.pick_list = []

        d1_smallest = 10
        d2_smallest = 10
        idx_1 = 10
        idx_2 = 10
        # Loop through all the cloth nodes near the co_picks to identify the nodes near the co-picks wthin the grasp range
        # Iterate over all vertices in the mesh
        for i in range(self.meshes.shape[0]):
            # Get distance between vertex and pick point
            d1 = ((self.meshes[i, 0] - co_picks[0][0]) ** 2 +
                  (self.meshes[i, 1] - co_picks[0][1]) ** 2) ** 0.5
            
            if d1 < d1_smallest:
                d1_smallest = d1
                idx_1 = i

            # if d1 < grasp_range:  # If vertex is within grasp range of the first pick point
            #     # append the vertex index to the pick_list with the first one has the smallest d1
            #     self.pick_list1.append(i)
            
            if use_multi:  # If multi-pick is enabled
                d2 = ((self.meshes[i, 0] - co_picks[1][0]) ** 2 +
                      (self.meshes[i, 1] - co_picks[1][1]) ** 2) ** 0.5
                
                if d2 < d2_smallest:
                    d2_smallest = d2
                    idx_2 = i

                # if d2 < grasp_range and d1 >= grasp_range:  # If vertex is within grasp range of the second pick point
                #     self.pick_list2.append(i)  # Add to pick_list2
        
        self.pick_list1.append(idx_1)
        self.pick_list2.append(idx_2)

        # Only pick the top nodes
        # Calculate the dimension of the mesh (assuming square grid)
        dimension = int(np.sqrt(self.meshes.shape[0]))
        
        # Pick first node from pick_list1
        if self.pick_list1 == []:  # If pick_list1 is empty
            print('pick_list1 is empty!')
        
        else:
            # Process pick_list1
            self.pick_list1, self.pick_point1 = self.select_topmost_and_neighbors(self.pick_list1, dimension)
            
            # Add neighbors to the combined pick list
            self.pick_list.extend(self.pick_list1)
        
        # Process pick_list2 (if multi-pick is enabled)
        if use_multi:
            if self.pick_list2 == []:  # If pick_list2 is empty
                print('pick_list2 is empty!')
            else:
                self.pick_list2, self.pick_point2 = self.select_topmost_and_neighbors(self.pick_list2, dimension)
                self.pick_list.extend(self.pick_list2)       

        # Print the selected pick vertices
        styled_message(f'Pick_list vertices: {self.pick_list}')
        
        if use_multi:  # If multi-pick is enabled
            return self.pick_list1, self.pick_list2, self.pick_list  # Return both pick lists
        
        return self.pick_list1, self.pick_list  # Return single pick list

    def sim_uniform_linear_movement(self, hand1, hand2, camera, hand1_positions, hand2_positions, frame_num, use_multi, move_camera):
        '''
        Simulate the uniform linear motion of the hands.
        ''' 

        # calculate camera motion
        if move_camera:
            delta_x = hand1_positions[1][0] - hand1_positions[0][0]
            delta_y = hand1_positions[1][1] - hand1_positions[0][1]
        camera_location = camera.location
        
        # update hand1 and hand2 positions
        # print(f"hand1_positions: {hand1_positions}") 
        for idx, hand1_position in enumerate(hand1_positions):
            # print(f"hand1_position: {hand1_position}")
            # Update Hand1's position and insert a keyframe
            hand1.location = (hand1_position[0], hand1_position[1], hand1_position[2])
            hand1.keyframe_insert(data_path="location", frame=frame_num)
        
            # Update Hand2's position (if multi-hand operation) and insert a keyframe
            if use_multi:
                hand2_position = hand2_positions[idx]
                # print(f"hand2_position: {hand2_position}")
                hand2.location = (hand2_position[0], hand2_position[1], hand2_position[2])
                hand2.keyframe_insert(data_path="location", frame=frame_num)
            
            if move_camera:
                # Update the camera's position and insert a keyframe
                camera.location.x += delta_x
                camera.location.y += delta_y
                camera.keyframe_insert(data_path="location", frame=frame_num)
                camera_location = camera.location

            # Increment frame number
            frame_num += 1

        return (hand1_positions[-1]), (hand2_positions[-1]), frame_num, camera_location

    def random_grasp_lift_move(self, use_multi = False, use_lift = False):
        frame_num = 0
        cloth = self.cloth
        camera = self.get_existing_object(type = 'CAMERA')

        # get current mesh vertices
        self.meshes = self.convert_mesh_to_np(cloth.data)[0]
        if hasattr(self, 'keypoints') is False:
            self.keypoints = self.find_keypoints_lists_t_shirt(self.meshes, plot = True)  

        # ------------------------------------------------------------------------------
        # generate the grasping point inside the cloth given the keypoints
        # ------------------------------------------------------------------------------

        # random select a pick point from the keypoints
        if cloth.name == "T_Shirt_l1" or "T_Shirt_l2" or "T_Shirt_l3":
            np_pick_1 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
            
            np_pick_2 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
        
        np_picks_both = np.column_stack((np_pick_1, np_pick_2)).T

        # get the closest pick point index of the mesh
        if use_multi:
            pick_list_1, pick_list_2, pick_list = self.find_pick_lists(np_picks_both, use_multi=use_multi)

        else:
            np_picks_both[1] = np_picks_both[0]     # Set both pick points to the same (for single-hand operation)
            np.around(np_picks_both, 3)             # Round the pick coordinates to 3 decimal places
            pick_list_1, pick_list = self.find_pick_lists(np_picks_both, use_multi=use_multi)
            pick_list_2 = pick_list_1
        
        if not pick_list or (use_multi and pick_list_1 == pick_list_2):
            pick_list_1 = [3]
            pick_list_2 = [208]
            pick_list = pick_list_1 + pick_list_2

        # ------------------------------------------------------------------------------
        # generate the motion of the hands including lifting, moving and dropping
        # ------------------------------------------------------------------------------
        # calculate the initial position of the hands using the pick points
        hand1_loc_init = np.array([self.meshes[pick_list_1[0]][0], self.meshes[pick_list_1[0]][1], self.meshes[pick_list_1[0]][2]])
        hand2_loc_init = np.array([self.meshes[pick_list_2[0]][0], self.meshes[pick_list_2[0]][1], self.meshes[pick_list_2[0]][2]])
        print(f"hand1_loc: {hand1_loc_init}")
        print(f"hand2_loc: {hand2_loc_init}")

        # Generate random movement vectors for the pick points
        np_moves = np.random.uniform(-self.move_range, self.move_range, (2, 2))
        np_moves[1]  = np_moves[0] # Set both move vectors to the same
        np.around(np_moves, 3)  # Round the move vectors to 3 decimal places
        move_length1 = np.linalg.norm(np_moves[0])
        move_length2 = np.linalg.norm(np_moves[1])
        move_norm1 = np_moves[0] / move_length1 if move_length1 > 0 else np.array([1, 0])
        move_norm2 = np_moves[1] / move_length2 if move_length2 > 0 else np.array([1, 0])

        # Generate random lift heights, ensuring a safe lifting height
        np_heights = np.random.uniform(0, self.lift_height_max, (2)) + max(hand1_loc_init[2], hand2_loc_init[2])
        np_heights[0] = max(np_heights[0], np.amax(self.meshes, axis=0)[2] + 0.1)
        if not use_lift:
            np_heights[0] = np.amax(self.meshes, axis=0)[2] + 0.01  # lifting the cloth a little bit
        np_heights[1] = np_heights[0]  # Set both lift heights to the same
        np.around(np_heights, 3)  # Round the heights to 3 decimal places
        print('finish random movement vertor generation.')
        lift_height1 = np_heights[0]
        lift_height2 = np_heights[1]

        # ------------------------------------------------------------------------------
        # Initial setting of the fabric vertices
        # ------------------------------------------------------------------------------
        # Initialize frame number for simulation keyframes
        frame_num = 0

        # Initialize cloth vertex groups for the pick points
        # Create a vertex group for non-picked vertices
        empty = cloth.vertex_groups.new(name='empty')
        # Create a vertex group for vertices picked by Hand1
        pick1 = cloth.vertex_groups.new(name='pick1')
        # If multi-hand interaction is enabled, create a group for Hand2
        pick2 = cloth.vertex_groups.new(name='pick2')
        # Create a general vertex group for all picked vertices
        pick = cloth.vertex_groups.new(name='pick')

        # Add the pick nodes to the corresponding vertex groups
        # Assign full weight to pick1 group
        pick1.add(pick_list_1, 1.0, 'REPLACE') 
        # Assign full weight to pick2 group 
        pick2.add(pick_list_2, 1.0, 'REPLACE')  
        # Assign full weight to the general pick group
        pick.add(pick_list, 1.0, 'REPLACE')  

        # Assign vertex groups to Hook modifiers
        cloth.modifiers['Hook1'].vertex_group = 'pick1'
        cloth.modifiers['Hook2'].vertex_group = 'pick2'

        # Set the Cloth modifier parameters
        cloth.modifiers["Cloth"].settings.vertex_group_mass = 'empty'  # Non-picked vertices are affected by cloth mass
        cloth.modifiers["VertexWeightMix"].vertex_group_a = 'empty'
        cloth.modifiers["VertexWeightMix"].vertex_group_b = 'pick'
        cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD'  # Add weights of the vertex groups
        cloth.modifiers["VertexWeightMix"].mix_set = 'OR'  # The OR mix set mode

        # ------------------------------------------------------------------------------
        # Initialize the hands position
        # ------------------------------------------------------------------------------
        # Add Hand1 and Hand2 as Empty objects in the scene
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand1'  # Name the first hand
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand1

        # If multi-hand interaction is enabled, add Hand2
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand2'
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand2

        # Assign Hand1 and Hand2 to the Hook modifiers
        cloth.modifiers["Hook1"].object = bpy.data.objects["Hand1"]
        cloth.modifiers["Hook2"].object = bpy.data.objects["Hand2"]

        # Set the stiffness for the pinning of cloth vertices
        cloth.modifiers["Cloth"].settings.pin_stiffness = 20

        # Clear any existing animation data for the hands
        hand1 = bpy.data.objects['Hand1']
        hand1.animation_data_clear()
        hand2 = bpy.data.objects['Hand2']
        hand2.animation_data_clear()

        # Initialize hand and cloth simulation (set initial positions)
        hand1.location = (hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2])
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        hand2.location = (hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2])
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 5

        # ------------------------------------------------------------------------------
        # Start generate the trajectories of the hands
        # ------------------------------------------------------------------------------
        # define the step size for the movement
        step = 0.01
        
        # generate the lift up trajectory from initial z to lift_height
        z1_positions = np.linspace(hand1_loc_init[2], lift_height1, int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        x1_positions = np.linspace(hand1_loc_init[0], hand1_loc_init[0], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        y1_positions = np.linspace(hand1_loc_init[1], hand1_loc_init[1], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        z2_positions = np.linspace(hand2_loc_init[2], lift_height2, int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        x2_positions = np.linspace(hand2_loc_init[0], hand2_loc_init[0], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        y2_positions = np.linspace(hand2_loc_init[1], hand2_loc_init[1], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the lifting action
        hand1_loc_lift, hand2_loc_lift, frame_num, camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi, move_camera = False)
        frame_num += 10

        # generation the horizontal movement trajectory for current x1, y1, x2, y2
        larger_move_length = max(move_length1, move_length2)
        x1_positions = np.linspace(hand1_loc_lift[0], hand1_loc_lift[0] + np_moves[0][0], int(np.abs(larger_move_length) / step) + 2)
        y1_positions = np.linspace(hand1_loc_lift[1], hand1_loc_lift[1] + np_moves[0][1], int(np.abs(larger_move_length) / step) + 2)
        z1_positions = np.linspace(hand1_loc_lift[2], hand1_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        x2_positions = np.linspace(hand2_loc_lift[0], hand2_loc_lift[0] + np_moves[1][0], int(np.abs(larger_move_length) / step) + 2)
        y2_positions = np.linspace(hand2_loc_lift[1], hand2_loc_lift[1] + np_moves[1][1], int(np.abs(larger_move_length) / step) + 2)
        z2_positions = np.linspace(hand2_loc_lift[2], hand2_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the horizontal movement
        hand1_loc_move, hand2_loc_move, frame_num , camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi, move_camera = True)
        
        # camera location fixed and wait for stabilization
        for i in range(10):
            camera.location = camera_location
            camera.keyframe_insert(data_path="location", frame=frame_num)
            frame_num += 1

        # generation of the drop trajectory 
        cloth.modifiers["VertexWeightMix"].mask_constant = 1
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # realease the fabric
        frame_num += 1
        release_at = frame_num 
        cloth.modifiers["VertexWeightMix"].mask_constant = 0
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # Wait for cloth to stabilize
        frame_num += 60

        styled_message(f"Finish trajectory generation. Total frames: {frame_num}")
        return frame_num

    def random_shear_fabric_using_dualarm(self):
        frame_num = 0
        cloth = self.cloth
        camera = self.get_existing_object(type = 'CAMERA')

        # get current mesh vertices
        self.meshes = self.convert_mesh_to_np(cloth.data)[0]
        if hasattr(self, 'keypoints') is False:
            self.keypoints = self.find_keypoints_lists_t_shirt(self.meshes, plot = True)  

        # ------------------------------------------------------------------------------
        # generate the grasping point inside the cloth given the keypoints
        # ------------------------------------------------------------------------------

        # random select a pick point from the keypoints
        if cloth.name == "T_Shirt_l1" or "T_Shirt_l2" or "T_Shirt_l3":
            np_pick_1 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
            
            np_pick_2 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
        
        np_picks_both = np.column_stack((np_pick_1, np_pick_2)).T

        pick_list_1, pick_list_2, pick_list = self.find_pick_lists(np_picks_both, use_multi=True)
        
        if pick_list == [] or pick_list_1 == pick_list_2:
            pick_list_1 = [3]
            pick_list_2 = [208]
            pick_list = pick_list_1 + pick_list_2

        # ------------------------------------------------------------------------------
        # generate the motion of the hands including lifting, moving and dropping
        # ------------------------------------------------------------------------------
        # calculate the initial position of the hands using the pick points
        hand1_loc_init = np.array([self.meshes[pick_list_1[0]][0], self.meshes[pick_list_1[0]][1], self.meshes[pick_list_1[0]][2]])
        hand2_loc_init = np.array([self.meshes[pick_list_2[0]][0], self.meshes[pick_list_2[0]][1], self.meshes[pick_list_2[0]][2]])
        print(f"hand1_loc: {hand1_loc_init}")
        print(f"hand2_loc: {hand2_loc_init}")

        # # Generate random movement vectors for the pick points
        # np_moves = np.random.uniform(-self.move_range, self.move_range, (2, 2))
        # np_moves[1]  = np_moves[0] # Set both move vectors to the same
        # np.around(np_moves, 3)  # Round the move vectors to 3 decimal places
        # move_length1 = np.linalg.norm(np_moves[0])
        # move_length2 = np.linalg.norm(np_moves[1])
        # move_norm1 = np_moves[0] / move_length1 if move_length1 > 0 else np.array([1, 0])
        # move_norm2 = np_moves[1] / move_length2 if move_length2 > 0 else np.array([1, 0])

        norm_dual = hand1_loc_init - hand2_loc_init
        length_dual = np.linalg.norm(norm_dual)
        rand_r = np.random.uniform(0, length_dual)
        rand_theta = np.random.uniform(0, 2 * np.pi)
        rand_x = rand_r * np.cos(rand_theta)
        rand_y = rand_r * np.sin(rand_theta)
        np_moves = np.zeros((2, 2))
        np_moves[0] = [rand_x - norm_dual[0], rand_y - norm_dual[1]]
        np.around(np_moves, 3)  # Round the move vectors to 3 decimal places
        move_length1 = np.linalg.norm(np_moves[0])
        move_length2 = np.linalg.norm(np_moves[1])

        # Generate random lift heights, ensuring a safe lifting height
        np_heights = np.zeros(2)
        np_heights[0] = np.amax(self.meshes, axis=0)[2] + 0.05  # lifting the cloth a little bit
        np_heights[1] = np_heights[0]  # Set both lift heights to the same
        np.around(np_heights, 3)  # Round the heights to 3 decimal places
        print('finish random movement vertor generation.')
        lift_height1 = np_heights[0]
        lift_height2 = np_heights[1]

        # ------------------------------------------------------------------------------
        # Initial setting of the fabric vertices
        # ------------------------------------------------------------------------------
        # Initialize frame number for simulation keyframes
        frame_num = 0

        # Initialize cloth vertex groups for the pick points
        # Create a vertex group for non-picked vertices
        empty = cloth.vertex_groups.new(name='empty')
        # Create a vertex group for vertices picked by Hand1
        pick1 = cloth.vertex_groups.new(name='pick1')
        # If multi-hand interaction is enabled, create a group for Hand2
        pick2 = cloth.vertex_groups.new(name='pick2')
        # Create a general vertex group for all picked vertices
        pick = cloth.vertex_groups.new(name='pick')

        # Add the pick nodes to the corresponding vertex groups
        # Assign full weight to pick1 group
        pick1.add(pick_list_1, 1.0, 'REPLACE') 
        # Assign full weight to pick2 group 
        pick2.add(pick_list_2, 1.0, 'REPLACE')  
        # Assign full weight to the general pick group
        pick.add(pick_list, 1.0, 'REPLACE')  

        # Assign vertex groups to Hook modifiers
        cloth.modifiers['Hook1'].vertex_group = 'pick1'
        cloth.modifiers['Hook2'].vertex_group = 'pick2'

        # Set the Cloth modifier parameters
        cloth.modifiers["Cloth"].settings.vertex_group_mass = 'empty'  # Non-picked vertices are affected by cloth mass
        cloth.modifiers["VertexWeightMix"].vertex_group_a = 'empty'
        cloth.modifiers["VertexWeightMix"].vertex_group_b = 'pick'
        cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD'  # Add weights of the vertex groups
        cloth.modifiers["VertexWeightMix"].mix_set = 'OR'  # The OR mix set mode

        # ------------------------------------------------------------------------------
        # Initialize the hands position
        # ------------------------------------------------------------------------------
        # Add Hand1 and Hand2 as Empty objects in the scene
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand1'  # Name the first hand
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand1

        # If multi-hand interaction is enabled, add Hand2
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand2'
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand2

        # Assign Hand1 and Hand2 to the Hook modifiers
        cloth.modifiers["Hook1"].object = bpy.data.objects["Hand1"]
        cloth.modifiers["Hook2"].object = bpy.data.objects["Hand2"]

        # Set the stiffness for the pinning of cloth vertices
        cloth.modifiers["Cloth"].settings.pin_stiffness = 20

        # Clear any existing animation data for the hands
        hand1 = bpy.data.objects['Hand1']
        hand1.animation_data_clear()
        hand2 = bpy.data.objects['Hand2']
        hand2.animation_data_clear()

        # Initialize hand and cloth simulation (set initial positions)
        hand1.location = (hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2])
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        hand2.location = (hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2])
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 5

        # ------------------------------------------------------------------------------
        # Start generate the trajectories of the hands
        # ------------------------------------------------------------------------------
        # define the step size for the movement
        step = 0.01
        
        # generate the lift up trajectory from initial z to lift_height
        z1_positions = np.linspace(hand1_loc_init[2], lift_height1, int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        x1_positions = np.linspace(hand1_loc_init[0], hand1_loc_init[0], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        y1_positions = np.linspace(hand1_loc_init[1], hand1_loc_init[1], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        z2_positions = np.linspace(hand2_loc_init[2], lift_height2, int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        x2_positions = np.linspace(hand2_loc_init[0], hand2_loc_init[0], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        y2_positions = np.linspace(hand2_loc_init[1], hand2_loc_init[1], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the lifting action
        hand1_loc_lift, hand2_loc_lift, frame_num, camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi = True, move_camera = False)
        frame_num += 10

        # generation the horizontal movement trajectory for current x1, y1, x2, y2
        larger_move_length = max(move_length1, move_length2)
        x1_positions = np.linspace(hand1_loc_lift[0], hand1_loc_lift[0] + np_moves[0][0], int(np.abs(larger_move_length) / step) + 2)
        y1_positions = np.linspace(hand1_loc_lift[1], hand1_loc_lift[1] + np_moves[0][1], int(np.abs(larger_move_length) / step) + 2)
        z1_positions = np.linspace(hand1_loc_lift[2], hand1_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        x2_positions = np.linspace(hand2_loc_lift[0], hand2_loc_lift[0] + np_moves[1][0], int(np.abs(larger_move_length) / step) + 2)
        y2_positions = np.linspace(hand2_loc_lift[1], hand2_loc_lift[1] + np_moves[1][1], int(np.abs(larger_move_length) / step) + 2)
        z2_positions = np.linspace(hand2_loc_lift[2], hand2_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the horizontal movement
        hand1_loc_move, hand2_loc_move, frame_num , camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi = True, move_camera = True)
        
        # camera location fixed and wait for stabilization
        for i in range(10):
            camera.location = camera_location
            camera.keyframe_insert(data_path="location", frame=frame_num)
            frame_num += 1

        # generation of the drop trajectory 
        cloth.modifiers["VertexWeightMix"].mask_constant = 1
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # realease the fabric
        frame_num += 1
        release_at = frame_num 
        cloth.modifiers["VertexWeightMix"].mask_constant = 0
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # Wait for cloth to stabilize
        frame_num += 1

        styled_message(f"Finish trajectory generation. Total frames: {frame_num}")
        return frame_num

    def random_compress_fabric_dualarm(self):
        frame_num = 0
        cloth = self.cloth
        camera = self.get_existing_object(type = 'CAMERA')

        # get current mesh vertices
        self.meshes = self.convert_mesh_to_np(cloth.data)[0]
        if hasattr(self, 'keypoints') is False:
            self.keypoints = self.find_keypoints_lists_t_shirt(self.meshes, plot = True)  

        # ------------------------------------------------------------------------------
        # generate the grasping point inside the cloth given the keypoints
        # ------------------------------------------------------------------------------

        # random select a pick point from the keypoints
        if cloth.name == "T_Shirt_l1" or "T_Shirt_l2" or "T_Shirt_l3":
            np_pick_1 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
            
            np_pick_2 = np.array([np.random.uniform(self.keypoints_details[0]['arm_pit_left'][0],
                                                           self.keypoints_details[1]['arm_pit_right'][0]),
                                    np.random.uniform(self.keypoints_details[6]['bottom_hem_left'][1],
                                                      self.keypoints_details[2]['neckline_left'][1])])
        
        np_picks_both = np.column_stack((np_pick_1, np_pick_2)).T

        pick_list_1, pick_list_2, pick_list = self.find_pick_lists(np_picks_both, use_multi=True)
        
        if pick_list == [] or pick_list_1 == pick_list_2:
            pick_list_1 = [3]
            pick_list_2 = [208]
            pick_list = pick_list_1 + pick_list_2

        # ------------------------------------------------------------------------------
        # generate the motion of the hands including lifting, moving and dropping
        # ------------------------------------------------------------------------------
        # calculate the initial position of the hands using the pick points
        hand1_loc_init = np.array([self.meshes[pick_list_1[0]][0], self.meshes[pick_list_1[0]][1], self.meshes[pick_list_1[0]][2]])
        hand2_loc_init = np.array([self.meshes[pick_list_2[0]][0], self.meshes[pick_list_2[0]][1], self.meshes[pick_list_2[0]][2]])
        print(f"hand1_loc: {hand1_loc_init}")
        print(f"hand2_loc: {hand2_loc_init}")

        # # Generate random movement vectors for the pick points
        # np_moves = np.random.uniform(-self.move_range, self.move_range, (2, 2))
        # np_moves[1]  = np_moves[0] # Set both move vectors to the same
        # np.around(np_moves, 3)  # Round the move vectors to 3 decimal places
        # move_length1 = np.linalg.norm(np_moves[0])
        # move_length2 = np.linalg.norm(np_moves[1])
        # move_norm1 = np_moves[0] / move_length1 if move_length1 > 0 else np.array([1, 0])
        # move_norm2 = np_moves[1] / move_length2 if move_length2 > 0 else np.array([1, 0])

        vec_dual = hand1_loc_init - hand2_loc_init
        length_dual = np.linalg.norm(vec_dual)
        norm_dual = vec_dual / length_dual
        rand_r = np.random.uniform(length_dual/15, length_dual/3)
        delta = norm_dual * rand_r
        np_moves = np.zeros((2, 2))
        np_moves[0] = [-delta[0], -delta[1]]
        np_moves[1] = [delta[0], delta[1]]
        np.around(np_moves, 3)  # Round the move vectors to 3 decimal places
        move_length1 = np.linalg.norm(np_moves[0])
        move_length2 = np.linalg.norm(np_moves[1])

        # Generate random lift heights, ensuring a safe lifting height
        np_heights = np.zeros(2)
        np_heights[0] = np.amax(self.meshes, axis=0)[2] + 0.05  # lifting the cloth a little bit
        np_heights[1] = np_heights[0]  # Set both lift heights to the same
        np.around(np_heights, 3)  # Round the heights to 3 decimal places
        print('finish random movement vertor generation.')
        lift_height1 = np_heights[0]
        lift_height2 = np_heights[1]

        # ------------------------------------------------------------------------------
        # Initial setting of the fabric vertices
        # ------------------------------------------------------------------------------
        # Initialize frame number for simulation keyframes
        frame_num = 0

        # Initialize cloth vertex groups for the pick points
        # Create a vertex group for non-picked vertices
        empty = cloth.vertex_groups.new(name='empty')
        # Create a vertex group for vertices picked by Hand1
        pick1 = cloth.vertex_groups.new(name='pick1')
        # If multi-hand interaction is enabled, create a group for Hand2
        pick2 = cloth.vertex_groups.new(name='pick2')
        # Create a general vertex group for all picked vertices
        pick = cloth.vertex_groups.new(name='pick')

        # Add the pick nodes to the corresponding vertex groups
        # Assign full weight to pick1 group
        pick1.add(pick_list_1, 1.0, 'REPLACE') 
        # Assign full weight to pick2 group 
        pick2.add(pick_list_2, 1.0, 'REPLACE')  
        # Assign full weight to the general pick group
        pick.add(pick_list, 1.0, 'REPLACE')  

        # Assign vertex groups to Hook modifiers
        cloth.modifiers['Hook1'].vertex_group = 'pick1'
        cloth.modifiers['Hook2'].vertex_group = 'pick2'

        # Set the Cloth modifier parameters
        cloth.modifiers["Cloth"].settings.vertex_group_mass = 'empty'  # Non-picked vertices are affected by cloth mass
        cloth.modifiers["VertexWeightMix"].vertex_group_a = 'empty'
        cloth.modifiers["VertexWeightMix"].vertex_group_b = 'pick'
        cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD'  # Add weights of the vertex groups
        cloth.modifiers["VertexWeightMix"].mix_set = 'OR'  # The OR mix set mode

        # ------------------------------------------------------------------------------
        # Initialize the hands position
        # ------------------------------------------------------------------------------
        # Add Hand1 and Hand2 as Empty objects in the scene
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand1'  # Name the first hand
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand1

        # If multi-hand interaction is enabled, add Hand2
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2]), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand2'
        # bpy.context.object.scale = [1, 1, 0.2]  # Scale down Z axis for Hand2

        # Assign Hand1 and Hand2 to the Hook modifiers
        cloth.modifiers["Hook1"].object = bpy.data.objects["Hand1"]
        cloth.modifiers["Hook2"].object = bpy.data.objects["Hand2"]

        # Set the stiffness for the pinning of cloth vertices
        cloth.modifiers["Cloth"].settings.pin_stiffness = 20

        # Clear any existing animation data for the hands
        hand1 = bpy.data.objects['Hand1']
        hand1.animation_data_clear()
        hand2 = bpy.data.objects['Hand2']
        hand2.animation_data_clear()

        # Initialize hand and cloth simulation (set initial positions)
        hand1.location = (hand1_loc_init[0], hand1_loc_init[1], hand1_loc_init[2])
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        hand2.location = (hand2_loc_init[0], hand2_loc_init[1], hand2_loc_init[2])
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 5

        # ------------------------------------------------------------------------------
        # Start generate the trajectories of the hands
        # ------------------------------------------------------------------------------
        # define the step size for the movement
        step = 0.01
        
        # generate the lift up trajectory from initial z to lift_height
        z1_positions = np.linspace(hand1_loc_init[2], lift_height1, int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        x1_positions = np.linspace(hand1_loc_init[0], hand1_loc_init[0], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        y1_positions = np.linspace(hand1_loc_init[1], hand1_loc_init[1], int((lift_height1 - hand1_loc_init[2]) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        z2_positions = np.linspace(hand2_loc_init[2], lift_height2, int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        x2_positions = np.linspace(hand2_loc_init[0], hand2_loc_init[0], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        y2_positions = np.linspace(hand2_loc_init[1], hand2_loc_init[1], int((lift_height2 - hand2_loc_init[2]) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the lifting action
        hand1_loc_lift, hand2_loc_lift, frame_num, camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi = True, move_camera = False)
        frame_num += 10

        # generation the horizontal movement trajectory for current x1, y1, x2, y2
        larger_move_length = max(move_length1, move_length2)
        x1_positions = np.linspace(hand1_loc_lift[0], hand1_loc_lift[0] + np_moves[0][0], int(np.abs(larger_move_length) / step) + 2)
        y1_positions = np.linspace(hand1_loc_lift[1], hand1_loc_lift[1] + np_moves[0][1], int(np.abs(larger_move_length) / step) + 2)
        z1_positions = np.linspace(hand1_loc_lift[2], hand1_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand1_positions = np.column_stack((x1_positions, y1_positions, z1_positions))
        x2_positions = np.linspace(hand2_loc_lift[0], hand2_loc_lift[0] + np_moves[1][0], int(np.abs(larger_move_length) / step) + 2)
        y2_positions = np.linspace(hand2_loc_lift[1], hand2_loc_lift[1] + np_moves[1][1], int(np.abs(larger_move_length) / step) + 2)
        z2_positions = np.linspace(hand2_loc_lift[2], hand2_loc_lift[2], int(np.abs(larger_move_length) / step) + 2)
        hand2_positions = np.column_stack((x2_positions, y2_positions, z2_positions))

        # Simulate the horizontal movement
        hand1_loc_move, hand2_loc_move, frame_num , camera_location = self.sim_uniform_linear_movement(hand1, hand2, camera,
                           hand1_positions, hand2_positions, frame_num, use_multi = True, move_camera = True)
        
        # camera location fixed and wait for stabilization
        for i in range(10):
            camera.location = camera_location
            camera.keyframe_insert(data_path="location", frame=frame_num)
            frame_num += 1

        # generation of the drop trajectory 
        cloth.modifiers["VertexWeightMix"].mask_constant = 1
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # realease the fabric
        frame_num += 1
        release_at = frame_num 
        cloth.modifiers["VertexWeightMix"].mask_constant = 0
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
        
        # Wait for cloth to stabilize
        frame_num += 1

        styled_message(f"Finish trajectory generation. Total frames: {frame_num}")
        return frame_num

    def save_final_mesh_pcl(self, idx):
        # set the final mesh address
        final_pcl_address = os.path.join(self.database_dir, f'{num_data:06d}.simu_pcl.txt')
        final_mesh_address = os.path.join(self.database_dir, f'{num_data:06d}.simu_mesh.obj')
        
        # This gives access to the object's final state after applying all modifiers and simulations
        depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get the evaluated version of the cloth object
        evaluated_cloth = self.cloth.evaluated_get(depsgraph)

        # Extract the final mesh data after applying all modifiers and simulations (e.g., cloth physics)
        evaluated_mesh = evaluated_cloth.to_mesh()

        # convert the mesh to numpy array
        self.result_np_pcl, np_edge, np_face, np_face_tri= self.convert_mesh_to_np(evaluated_mesh)

        if self.result_np_pcl is not None:
            # Save the final point cloud as a txt file
            np.savetxt(final_pcl_address, self.result_np_pcl)
            styled_message(f"Final pcl saved to {final_pcl_address}", fg_color="32", bg_color="44")

            # save the final mesh as a .obj file including vertices and faces
            self.save_as_obj(self.result_np_pcl, np_face_tri, final_mesh_address)
            styled_message(f"Final mesh saved to {final_mesh_address}", fg_color="32", bg_color="44")
            
            # Free the evaluated mesh
            evaluated_cloth.to_mesh_clear()
            return True
            
        styled_message("Final mesh is None. Cannot save the final mesh.", fg_color="31", bg_color="33")
        return False

    def cloth_simulation(self, texture_dir, operation, use_multi, use_lift, idx):
        # --------------------------------------------------------------------- 
        # Initialize the simulation
        # ---------------------------------------------------------------------
        # delete all the objects in the scene
        self.ensure_obj_mode_and_del_existing_obj_datablock()

        # generate the ground plane, camera and sunlight
        self.generate_plane_camera_sun()

        # generate cloth model
        if cloth_style == 'T_Shirtv2':
            self.generate_t_shirtv2_model(texture_dir = texture_dir)    
        elif cloth_style == "T_Shirt_l1":
            self.generate_t_shirt_l1_model(texture_dir = texture_dir)
        elif cloth_style == "T_Shirt_l2":
            self.generate_t_shirt_l2_model(texture_dir = texture_dir)
        elif cloth_style == "T_Shirt_l3":
            self.generate_t_shirt_l3_model(texture_dir = texture_dir)

        # set the frame back to 0
        bpy.context.scene.frame_set(0)
        bpy.context.view_layer.update()

        # --------------------------------------------------------------------- 
        # Randomly generate the cloth manipulation
        # ---------------------------------------------------------------------
        # select the operation function
        if operation == 1 or operation == 2 or operation == 3 or operation == 4:
            frame_num = self.random_grasp_lift_move(use_multi, use_lift)
        elif operation == 5:
            frame_num = self.random_compress_fabric_dualarm()
        elif operation == 6:
            frame_num = self.random_shear_fabric_using_dualarm()

        # --------------------------------------------------------------------- 
        # Simulate the cloth manipulation
        # ---------------------------------------------------------------------
        styled_message(f'Start to simulate each frame.')  # Output simulation time
        t_start = time.time()
        
        # Method 1
        # Step through frames to simulate the cloth movement
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = frame_num

        # select the t shirt object  from the sence collection
        bpy.context.view_layer.objects.active = bpy.data.objects[self.cloth.name]
        bpy.context.object.select_set(True)
        bpy.context.object.modifiers["Cloth"].point_cache.frame_end = frame_num
        for i in range(frame_num + 1):
            t_frame = time.time()
            bpy.context.scene.frame_set(i)  # Set the scene frame
            bpy.context.view_layer.update()  # ensure new frame is fully evaluated

        # Method 2 
        # Use bake all to simulate the cloth movement
        # bpy.context.scene.frame_start = 1
        # bpy.context.scene.frame_end = frame_num
        # self.cloth.modifiers["Cloth"].point_cache.frame_end = frame_num
        # bpy.ops.ptcache.free_bake_all()
        # bpy.ops.ptcache.bake_all(bake=True)

        styled_message(f'simulation time: {time.time() - t_start}')  # Output simulation time

        # save the final mesh
        success_save_mesh = self.save_final_mesh_pcl(idx)
        if not success_save_mesh:
            return False
        
        # save the final image
        bpy.data.scenes["Scene"].frame_end = frame_num
        rgb_filename = os.path.join(self.database_dir, f'{num_data:06d}.simu_rgb.png')
        cloth_sim_model.scene.render.filepath = rgb_filename
        bpy.ops.render.render(write_still=True)

        styled_message(f"Final image saved to {rgb_filename}", fg_color="32", bg_color="44")

        return True


if __name__ == "__main__":

    # --------------------------------------------------------------------- #
    # ------------------------- Setup Directories ------------------------- #
    # --------------------------------------------------------------------- #

    PROJECT_PATH = '/home/ktang/ws/Mesh_GAN_cloth'

    cloth_style = 'T_Shirt_l3'

    if cloth_style == 'T_Shirtv2':
        # Path to your T-shirt OBJ file
        obj_file_path = os.path.join(PROJECT_PATH, 'CAD/t_shirtv2_generated_from_berkley_coarse.obj')
        # Base directory for saving state files
        database_dir = '/home/ktang/ws/data/Mesh_GAN_cloth/cloth_t_shirtv2_coarse_data/'
        # Texture directory
        texture_dir = '/home/ktang/ws/data/textures/'
    elif cloth_style == 'T_Shirt_l1':
        # Path to your T-shirt OBJ file
        obj_file_path = os.path.join(PROJECT_PATH, 'CAD/t_shirt_meshlab/t_shirt_l1.obj')
        # Base directory for saving state files
        database_dir = '/home/ktang/ws/data/Mesh_GAN_cloth/t_shirt_meshlab/t_shirt_l1/'
        # Texture directory
        texture_dir = '/home/ktang/ws/data/textures/'
    elif cloth_style == "T_Shirt_l2":
        # Path to your T-shirt OBJ file
        obj_file_path = os.path.join(PROJECT_PATH, 'CAD/t_shirt_meshlab/t_shirt_l2.obj')
        # Base directory for saving state files
        database_dir = '/home/ktang/ws/data/Mesh_GAN_cloth/t_shirt_meshlab/t_shirt_l2/'
        # Texture directory
        texture_dir = '/home/ktang/ws/data/textures/'
    elif cloth_style == "T_Shirt_l3":
        # Path to your T-shirt OBJ file
        obj_file_path = os.path.join(PROJECT_PATH, 'CAD/t_shirt_meshlab/t_shirt_l3.obj')
        # Base directory for saving state files
        database_dir = '/home/ktang/ws/data/Mesh_GAN_cloth/t_shirt_meshlab/t_shirt_l3/'
        # Texture directory
        texture_dir = '/home/ktang/ws/data/textures/'

    initial_state_dir = os.path.join(database_dir, 'start_states/start_state.txt')

    # create directory if not exist
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)

    first_time = False
    if not os.path.exists(initial_state_dir):
        first_time = True

    # --------------------------------------------------------------------- #
    # ------------- Initialize blender, cloth, and simulation ------------- #
    # --------------------------------------------------------------------- #
    
    if first_time:
        first_time_initialization(obj_file_path, database_dir, cloth_style)

    initial_image_index = 0
    num_data = initial_image_index
    total_num_data = 20000

    cloth_sim_model = ClothSimulation(obj_file_path, database_dir, cloth_style = cloth_style, multi = True, cam_move=True)

    for idx in range(total_num_data):
        
        # Randomly select the operation: 
        operation = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.05, 0.15, 0.15, 0.3, 0.3])
        # operation = 3
        if operation == 1:
            use_multi = True
            use_lift = False
            styled_message(f"Operation {operation}: Multi-hand operation without lifting.")
        elif operation == 2:
            use_multi = True
            use_lift = True
            styled_message(f"Operation {operation}: Multi-hand operation with lifting.")
        elif operation == 3:
            use_multi = False
            use_lift = False
            styled_message(f"Operation {operation}: Single-hand operation without lifting.")
        elif operation == 4:
            use_multi = False
            use_lift = True
            styled_message(f"Operation {operation}: Single-hand operation with lifting.")
        elif operation == 5:
            use_multi = True
            use_lift = False
            styled_message(f"Operation {operation}: compress the fabrics.")
        elif operation == 6:
            use_multi = True
            use_lift = False
            styled_message(f"Operation {operation}: shear the fabrics.")

        success = cloth_sim_model.cloth_simulation(texture_dir = texture_dir, 
                                                               operation = operation, 
                                                               use_multi=use_multi, 
                                                               use_lift=use_lift, 
                                                               idx = idx
                                                               )
                    
        if success:
            num_data += 1
            styled_message(f"Data {num_data} generated.")
        else:
            styled_message("Data generation failed.")