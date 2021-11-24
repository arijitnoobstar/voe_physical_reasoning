import bpy
import bpy
import bpycv
import boxx
import cv2
import json
import pandas as pd
import numpy as np
import mathutils
from mathutils import Vector
import random
import csv
import os
import shutil
import filecmp

# Number of trials (only applied if variation setting filename is not specified)
NUM_TRIALS = 5000

# Variation settings filename for upload. Set to None if it does not exist
variation_filename = './C_variation_settings.csv'
filter_10_percent = False

# Colour in HEX
red=0xF90707 
yellow=0xFFFF00 
blue=0x0000FF
lime=0x00FF00
orange = 0xFFA500
cyan=0x00FFFF
pink = 0xFFC0CB
green = 0x00FF00
silver= 0x808080
black = 0x000000
purple = 0x800080
white = 0xFFFFFF
metal=0x8A8A8A

# rendering settings
generate_data = True
starting_frame = 1 # THIS ONLY CHANGES RENDER FRAMES
ending_frame = 51 # THIS ONLY CHANGES RENDER FRAMES
delete_folder = True # change to False to not override data
render_frames = True
render_vis = False # set to False for actual dataset generation
render_videos = True

# dataset variation values
shape_list = ['Cube', 'Cylinder', 'Torus', 'Sphere', 'Cone', 'Side_Cylinder', 'Inverted_Cone']
container_list = ['box', 'mug']
obj_width_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
container_width_list = [0.7, 0.9, 1.1, 1.3, 1.5]
obj_height_list =  [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
container_height_list = [0.5, 0.7, 0.9, 1.1, 1.3]

# container scaling in x, y & z
container_scaler = {
    'box' : [3.5, 5.3, 10],
    'mug' : [8, 8, 5]
}

## comment this***
#shape_list = ['Cube']
#container_list = ['box']
#obj_width_list = [1]
#obj_height_list =  [1.6]
#container_width_list = [1.1]
#container_height_list = [0.7]

# Dataset Proportions
train_prop = 0.75
val_prop = 0.15
test_prop = 0.1

# determine the checkpoint trial for val and test
assert (train_prop + val_prop + test_prop) == 1.0
val_checkpoint = int(train_prop * NUM_TRIALS)
test_checkpoint = int((train_prop + val_prop) * NUM_TRIALS)

# make the data_filepath folders      
root_filepath = 'Data/C_container/'
filepath_extensions = ['train/surprising/', 'train/expected/', 'validation/surprising/', 'validation/expected/', \
'test/surprising/', 'test/expected/']

for extension in filepath_extensions:
    try:
        os.makedirs(root_filepath + extension)
    except:
        if delete_folder:
            shutil.rmtree(root_filepath + extension)
            os.makedirs(root_filepath + extension)
        else:
            pass
if variation_filename != None:
    variation_df = pd.read_csv(variation_filename)
    # filter out 10% for 10% detaset
    if filter_10_percent:
        variation_df = pd.concat([variation_df[:375],variation_df[3750:3900], variation_df[5250:5350]])
else:        
    data_segment = 'train'
    # determine variation settings for trials
    variation_settings_duplicate_checker = []
    variation_settings = []
    trial_count = 0
    while trial_count < NUM_TRIALS:
        
        variation = [trial_count + 1, random.choice(shape_list), random.choice(container_list),\
         random.choice(obj_width_list), random.choice(container_width_list),\
         random.choice(obj_height_list), random.choice(container_height_list), 'N.A.']
            
        # ensure no repeat of an existing trial variation (ignore random choice)
        if variation[1:-1] not in variation_settings_duplicate_checker:
            # add to duplicate checker
            variation_settings_duplicate_checker.append(variation[1:-1])
            # change data_segment based on trial count 
            if trial_count == val_checkpoint:
                data_segment = 'validation'
            elif trial_count == test_checkpoint:
                data_segment = 'test'
                
            # do not put non-fit scenes in expected - surprising pair (for now)
            if data_segment != 'train' and variation[3] > variation[4]:
                continue
            
            # gap between torus and box should be more (from manual observation) for val/test set
            if data_segment != 'train' and variation[1] == 'Torus' and \
            variation[2] == 'box' and (variation[3] - variation[4]) >= -0.11:
                continue

            # insert data_segment and violation as False (expected case) first
            variation.insert(1, data_segment)
            variation.insert(2, False)
            # add scene for expected case
            variation_settings.append(variation)
            
            # if validation or test set, change violation setting to True (surprising case)
            if data_segment != 'train':  
                variation_duplicate = variation[:]
                variation_duplicate[2] = True
                # replace rand_violation_choice's None value with random choice
                variation_duplicate[-1] = random.choice(['no_contain', 'no_protude'])
                # add scene for surprising case
                variation_settings.append(variation_duplicate)
            
            # increment trial count
            trial_count += 1

    # use pandas to store data into a csv
    variation_df = pd.DataFrame(variation_settings, columns = ['trial_num', 'data_segment', 'violation',\
    'shape', 'container', 'obj_width','container_width', 'obj_height','container_height','rand_violation_choice'])
    
    if filter_10_percent:
        variation_df = pd.concat([variation_df[:375],variation_df[3750:3900], variation_df[5250:5350]])
    
variation_df.to_csv(root_filepath + 'variation_settings.csv', index = False)

# lists to save outcome data from each scene
fit_list = []
protuded_list = []

# loop through each variation
# NOTE: iterate throught each setting, NOT trial_number as each trial can be made
# from 2 videos (expected and surprising)
for _, setting in variation_df.iterrows():
    
    # unpack variation settings for this trial
    trial_num, data_segment, violation, obj_shape, container, obj_width, container_width, \
    obj_height, container_height, rand_violation_choice = setting
    
    
    # Trial Specific Mappings & data_filepath
    if data_segment == 'train':
        data_filepath = root_filepath + 'train/'
    elif data_segment == 'validation':
        data_filepath = root_filepath + 'validation/'
    else:
        data_filepath = root_filepath + 'test/'
        
    if violation == False:
        data_filepath += 'expected/'
    else:
        data_filepath += 'surprising/'

    # Misc Settings
    instance_mapping = {
        'Ground': 0,
        'Object': 1,
        'Occluder' : 3, 
        'Container' : 4, # instance mapping is hardcoded consistently across all VIPAC A-E
    }

    # Colour mapping in a [string, hex] pair
    colour_mapping = {
        'Ground': ['white', white],
        'Container' : ['pink', pink],
        'Occluder' : ['cyan', cyan],
        'Cube' : ['red', red], 
        'Cylinder' : [ 'lime', lime], 
        'Torus' : [ 'orange', orange], 
        'Sphere' : [ 'yellow', yellow], 
        'Cone' : [ 'green', green], 
        'Side_Cylinder' : [ 'black', black], 
        'Inverted_Cone' : [ 'purple', purple]
    }
           
    # Define all helper functions
    def hex_to_rgb( hex_value ):
        """
        Converts HEX colour to RGB
        """
        b = (hex_value & 0xFF) / 255.0
        g = ((hex_value >> 8) & 0xFF) / 255.0
        r = ((hex_value >> 16) & 0xFF) / 255.0
        return r, g, b

    def add_material(obj, material_name, hue):
        """
        Adds material to object based on material name and hue
        """
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(material_name)
        material.use_nodes = True
        principled_bsdf = material.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            principled_bsdf.inputs[0].default_value = (*hex_to_rgb(hue), 1)  
        obj.active_material = material

    def create_light():
        """
        Create the lighting from the top
        """
        bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(7, 7, 7), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(0, -4.1934, 10.00991), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.resize(value=(1, 1.68457, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(0, 4.19906, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(-5.26466, -10, -0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # assign energy
        bpy.context.object.data.energy = 400
        
        # rename light accordingly
        bpy.context.object.name = 'Light'

        
    def create_ground(x_size, y_size, colour, rigid = True):
        """
        Create a ground of size (x_size, y_size, 1)
        """
        # add ground and size accordingly
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(x_size, y_size, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        if rigid:
            # make ground a right body in passive mode
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.type = 'PASSIVE'
        
        # rename plane to 'Ground'
        bpy.context.object.name = 'Ground'
        
        # add the material (colour) to the ground
        add_material(bpy.context.object, "Ground", colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Ground']
        
    def make_hidden_plane(height = 1, x = 1, y = 1):
        """
        Create a hidden plane inside container to force object ptotude out (in violation case)
        """
        # add ground and size accordingly
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(-1*2, -3.446*2, height))
        bpy.ops.transform.resize(value=(0.5 * x, 0.5 * y, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.type = 'PASSIVE'
        
        # rename plane to 'Ground'
        bpy.context.object.name = 'Hidden_Plane'
        
        # make it invisible to render
        bpy.context.object.hide_render = True
        
        # no instance id or colour made as it is hidden
        
        
    def object_spawn(shape, colour, x_scale = 1, y_scale = 1, z_scale = 1):
        """
        Spawn an object with specified shape, scales and colour above container
        """
        # spawn the object mesh
        if shape.lower() == "cube":
            bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        elif shape.lower() == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            bpy.ops.object.shade_smooth()
        elif shape.lower() == "cone":
            bpy.ops.mesh.primitive_cone_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))
            bpy.ops.object.shade_smooth()
        elif shape.lower() == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            bpy.ops.object.shade_smooth()
        elif shape.lower() == "torus":
            bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), major_radius=1, minor_radius=0.25, abso_major_rad=1.25, abso_minor_rad=0.75)
            bpy.ops.transform.rotate(value=1.57079632679, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
            bpy.ops.object.shade_smooth()
        elif shape.lower() == "side_cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            bpy.ops.transform.rotate(value=1.55643, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
            bpy.ops.transform.translate(value=(-0, -0, -1.51684), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
            bpy.ops.object.shade_smooth()
        elif shape.lower() == "inverted_cone":
            bpy.ops.mesh.primitive_cone_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))
            bpy.ops.transform.rotate(value=-3.12121, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
            bpy.ops.transform.translate(value=(-0, -0, -1.54691), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
            bpy.ops.object.shade_smooth()

        # size the object accordingly
        bpy.ops.transform.resize(value=(x_scale, y_scale, z_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # add material/colour to object
        add_material(bpy.context.object, shape, colour)
        
        # rename object name to name of shape
        bpy.context.object.name = shape
        

#        # give collision modifier
#        bpy.ops.object.modifier_add(type='COLLISION')
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Object']
        
        # position object abover container
        bpy.ops.transform.translate(value=(-1*2, -3.446*2, 6.25), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.restitution = 0.8
        
    def create_container(container, colour, container_scaler, rigid = True, x_scale = 1, \
    y_scale = 1, z_scale = 1, allow_containment = True):
        """
        Creates the container object
        """
        # scale fbx file to proper size
        x_scale *= container_scaler[0]
        y_scale *= container_scaler[1]
        z_scale *= container_scaler[2]
        
        if container == 'box':
            bpy.ops.import_scene.fbx( filepath = './3DModels/box.fbx')
            name = 'CardboardBox1'
        if container == 'mug':
            bpy.ops.import_scene.fbx( filepath = './3DModels/mug.fbx')
            name = 'Cylinder001'
            
        bpy.ops.transform.resize(value=(x_scale, y_scale, z_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(-1*2, -3.446*2, 0.05 + -1 * find_lowest_point(name)), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        ob = bpy.context.scene.objects[name]
        bpy.context.view_layer.objects.active = ob
        if rigid:
            # make it a rigid body and instantiate with mass to give it strong wall property
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.mass = 10000
            bpy.context.object.rigid_body.type = 'PASSIVE'
            if allow_containment:
                # use MESH collision shape to allow opening to have no collision causing mesh
                bpy.context.object.rigid_body.collision_shape = 'MESH'
            else:
                bpy.context.object.rigid_body.restitution = 1.0
        
        # name the wall accordingly
        bpy.context.object.name = 'Container'
        
        # add material/colour to wall
        add_material(bpy.context.object, 'Container', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Container'] 
        
    def create_occluder(colour, container):
        """
        Create the occluder
        """
        
        # determine the height adjust
        if container == 'mug':
            height_adjust = 1.3
        elif container == 'box':
            height_adjust = 1.2
            
        # initialise the occluder and orientate it
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.translate(value=(-2.9, -10.5, find_highest_point('Container') + height_adjust), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, True, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.resize(value=(2, 1, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, True, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.rotate(value=-1.56857, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        # name the occluder accordingly
        bpy.context.object.name = 'Occluder'
        
        # add material/colour to occluder
        add_material(bpy.context.object, 'Occluder', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Occluder']
        
    def occluder_move(to_height, start_frame, end_frame):
        """
        Move the occluder from from_height to to_height
        """
        occluder_mesh = bpy.context.scene.objects['Occluder']
#        occluder_mesh.location[2] = from_height
        occluder_mesh.keyframe_insert(data_path='location', frame=start_frame)
        occluder_mesh.location[2] = to_height
        occluder_mesh.keyframe_insert(data_path='location', frame=end_frame)
        
        
    def find_lowest_point(object):
        """
        Finds the lowest point of an object mesh
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return min(vectors, key=lambda item: item.z).z 
    
    def find_highest_point(object):
        """
        Finds the highest point of an object mesh
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return max(vectors, key=lambda item: item.z).z 
    
    def find_rightmost_point(object):
        """
        Finds the rightmost point of an object mesh (x-axis)
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return max(vectors, key=lambda item: item.x).x
    
    def find_leftmost_point(object):
        """
        Finds the leftmost point of an object mesh (x-axis)
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return min(vectors, key=lambda item: item.x).x 
    
    def find_backmost_point(object):
        """
        Finds the backmost point of an object mesh (y-axis)
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return max(vectors, key=lambda item: item.y).y
    
    def find_frontmost_point(object):
        """
        Finds the frontmost point of an object mesh (y-axis)
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        return min(vectors, key=lambda item: item.y).y 
    
    def find_height(object):
        """
        Finds the height of an object
        """
        return find_highest_point(object) - find_lowest_point(object)
    
    def find_width(object):
        """
        Finds the width of an object (x axis)
        """
        return find_rightmost_point(object) - find_leftmost_point(object)
        
    def find_depth(object):
        """
        Finds the depth of an object (y axis)
        """
        return find_backmost_point(object) - find_frontmost_point(object)
    
    def object_geometry_location(object):
        """
        finds the x,y,z of the object's geometry in world coordinates
        """
        mesh_obj = bpy.context.scene.objects[object]
        matrix_w = mesh_obj.matrix_world
        vectors = [matrix_w @ vertex.co for vertex in mesh_obj.data.vertices]
        x_coord = 0.5 * (max(vectors, key=lambda item: item.x).x + min(vectors, key=lambda item: item.x).x)
        y_coord = 0.5 * (max(vectors, key=lambda item: item.y).y + min(vectors, key=lambda item: item.y).y)
        z_coord = 0.5 * (max(vectors, key=lambda item: item.z).z + min(vectors, key=lambda item: item.z).z)
        
        return x_coord, y_coord, z_coord
        
    
    def frames_to_video(scene_filepath, type, delete_frame = False, do_npz = False, extension = 'jpg', start = starting_frame, end = ending_frame):
        """
        converts frames into avi video
        """
        # gather all frames
        frame_arr = []
        for frame_num in range(1, end - start + 1):
            if do_npz:
                if frame_num == 1:
                    npz_frames = cv2.imread('{}/{}_{}.{}'.format(scene_filepath, type, frame_num, extension))[...,0]
                else:
                    npz_frames = np.dstack((npz_frames, cv2.imread('{}/{}_{}.{}'.format(scene_filepath, type, frame_num, extension))[...,0]))
            else:
                frame_arr.append(cv2.imread('{}/{}_{}.{}'.format(scene_filepath, type, frame_num, extension)))
            if delete_frame:
                os.remove('{}/{}_{}.{}'.format(scene_filepath, type, frame_num, extension))
                
        if do_npz:
            np.savez_compressed('{}{}.npz'.format(scene_filepath, type), npz_frames)
        else:
            height, width, _ = frame_arr[0].shape
            size = (width, height)
            
            video = cv2.VideoWriter('{}/{}.avi'.format(scene_filepath, type) ,cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
            
            for frame in frame_arr:
                video.write(frame)
            video.release()
            
    # start scene operations        
            
    bpy.ops.object.select_all(action='DESELECT')
    # Check through objects of previous scene and delete everything except the Camera
    for obj in bpy.context.scene.objects:
        if obj.name != 'Camera':
           obj.select_set(True)
           bpy.ops.object.delete()        

    # create light, plane and container and spawn object above container
    create_light()
    colour_mapping['Object'] = [colour_mapping[obj_shape][0], colour_mapping[obj_shape][1]]
    object_spawn(obj_shape, colour_mapping['Object'][1], obj_width, obj_width, obj_height)

    
    if violation:
        # surprising case
        if not fit:
            # not fit, then only violation is to 'contain'
            create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = False)
            create_container(container, colour = colour_mapping['Container'][1], rigid = False,\
            x_scale = container_width, y_scale = container_width, z_scale = container_height,\
            container_scaler = container_scaler[container], allow_containment = True)
        elif protuded:
            # if protuded, violation can either not protude or not even contain at all (choose with 50% probability)
            if rand_violation_choice == 'no_protude':
                create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = False)
                create_container(container, colour = colour_mapping['Container'][1], rigid = False,\
                x_scale = container_width, y_scale = container_width, z_scale = container_height,\
                container_scaler = container_scaler[container], allow_containment = True)
            else:
                create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = True)
                create_container(container, colour = colour_mapping['Container'][1], rigid = True,\
                x_scale = container_width, y_scale = container_width, z_scale = container_height,\
                container_scaler = container_scaler[container], allow_containment = False)
        else:
             # if not protuded, violation can either protude or not even contain at all (choose with 50% probability)
            if rand_violation_choice == 'protude':
                create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = True)
                create_container(container, colour = colour_mapping['Container'][1], rigid = True,\
                x_scale = container_width, y_scale = container_width, z_scale = container_height,\
                container_scaler = container_scaler[container], allow_containment = True)
                # make hidden plane to force object to protude out
                make_hidden_plane(height = find_highest_point('Container') - (0.7 * find_height(obj_shape)),\
                x = find_width(obj_shape), y = find_depth(obj_shape))
            else:
                create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = True)
                create_container(container, colour = colour_mapping['Container'][1], rigid = True,\
                x_scale = container_width, y_scale = container_width, z_scale = container_height,\
                container_scaler = container_scaler[container], allow_containment = False)
    else:
        # expected case
        create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = True)
        create_container(container, colour = colour_mapping['Container'][1], rigid = True,\
        x_scale = container_width, y_scale = container_width, z_scale = container_height,\
        container_scaler = container_scaler[container], allow_containment = True)
        
        
    create_occluder(colour = colour_mapping['Occluder'][1], container = container)
    occluder_move(object_geometry_location('Occluder')[2] - 6, 35, 50)
    
    # bake object physics to keyframes
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects[obj_shape]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=ending_frame, step=1)

    # animate
    bpy.ops.screen.animation_play()
    
    # render the frames and metadata
    if generate_data:
        
        # make folder for scene
        scene_filepath = data_filepath + 'trial_{}/'.format(trial_num)
        try:
            os.mkdir(scene_filepath)
        except:
            if delete_folder:
                shutil.rmtree(scene_filepath)
                os.mkdir(scene_filepath)
            else:
                pass
        
        # initialise dictionary for scene_data json file
        scene_data = {}  
        # general data for scene_data json
        scene_data['trial_num'] = trial_num
        scene_data['dataset_segment'] = data_segment
        scene_data['violation'] = violation
        scene_data['frame'] = {}
             
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'JPEG' # set output format to .jpg
               
        # start rendering from 'starting_frame' till 'ending_frame'
        for frame_num in range(starting_frame, ending_frame):
            # set blender frame and normalise frame_num to start from 1 onwards
            scene.frame_set(frame_num)
            frame_num -= (starting_frame - 1)

            if render_frames:
                # render using bpycv           
                result = bpycv.render_data()
                # save raw rgb, depth (non-vis) and instance segmentation (non-vis) images
                cv2.imwrite(scene_filepath + "rgb_{}.jpg".format(frame_num), result["image"][..., ::-1])  
                cv2.imwrite(scene_filepath + "depth_raw_{}.jpg".format(frame_num), np.uint16(result["depth"]))
                cv2.imwrite(scene_filepath + "inst_raw_{}.jpg".format(frame_num), np.uint16(result["inst"]))
                
            # and condition needed as videos cannot be rendered without frames
            if render_vis and render_frames:
                # convert instance segmentation raw into a visual image using the boxx python package
                inst = result["inst"]
                if frame_num  == 1:
                    unique, _ = np.unique(inst, return_inverse=True)
                    instance_colour_mapping = boxx.getDefaultColorList(4, includeBackGround=True)
                # map unique instance ids to a visual image
                height, width = inst.shape
                inst_vis = np.zeros(shape = (height, width, 3))            
                for i in range(height):
                    for j in range(width):
                        # hardcoded part for VIPAC C as indexes are not numbered nicely
                        if inst[i,j] == 4:
                            index = 3
                        elif inst[i,j] == 3:
                            index = 2
                        else:
                            index = inst[i,j]
                        inst_vis[i,j,:] = instance_colour_mapping[index]
                inst_vis = inst_vis.astype(int) * 255
                cv2.imwrite(scene_filepath + "inst_vis_{}.jpg".format(frame_num), inst_vis[..., ::-1])

                # convert depth raw into a visual image by excluding background and normalising in grayscale
                depth = result["depth"]
                # mask background away and find min and max distances
                ground_mask = np.where(inst == instance_mapping['Ground'], False, True)
                depth_filtered = np.multiply(depth, ground_mask)
                # ensure that there is at least one pixel of non-ground, else all values are 0
                if (np.sum(depth_filtered) != 0):
                    depth_min = depth[ground_mask].min()
                    depth_max = depth[ground_mask].max()
                    # normalise and reverse so that nearer objects are brighter
                    depth_normalised = ( depth_filtered - depth_min )/( depth_max - depth_min )
                    depth_vis = 255 * np.multiply(1 - depth_normalised, ground_mask)
                else:
                    # all pixels are ground
                    depth_vis = np.copy(depth_filtered)
                cv2.imwrite(scene_filepath + "depth_vis_{}.jpg".format(frame_num), np.uint16(depth_vis))
            
            # frame data for scene_data json
            scene_data['frame'][str(frame_num)] = {}
            for entity in instance_mapping.keys():
                if entity == 'Ground' or entity == 'Hidden_Plane':
                    # no need for pose/orientation data for ground and hidden_plane
                    continue
                elif entity == 'Object':
                    # object has name specific to object shape
                    obj = bpy.context.scene.objects[obj_shape]
                else:
                    obj = bpy.context.scene.objects[entity]
                    
                scene_data['frame'][str(frame_num)][entity] = {}

                # Pose data
                scene_data['frame'][str(frame_num)][entity]['Pose'] = {}
                scene_data['frame'][str(frame_num)][entity]['Pose']['x'] = obj.location[0]
                scene_data['frame'][str(frame_num)][entity]['Pose']['y'] = obj.location[1]
                scene_data['frame'][str(frame_num)][entity]['Pose']['z'] = obj.location[2]
                    
                # Orientation Data
                scene_data['frame'][str(frame_num)][entity]['Orientation'] = {}
                scene_data['frame'][str(frame_num)][entity]['Orientation']['x'] = obj.rotation_euler[0]
                scene_data['frame'][str(frame_num)][entity]['Orientation']['y'] = obj.rotation_euler[1]
                scene_data['frame'][str(frame_num)][entity]['Orientation']['z'] = obj.rotation_euler[2] 

        # instance data for scene_data json
        scene_data['instance'] = {}
        for entity in instance_mapping.keys():
            scene_data['instance'][instance_mapping[entity]] = \
            {'type' : entity.lower(), 'colour' : colour_mapping[entity][0],
            'colour_hex' : "0x" + format(colour_mapping[entity][1], "06X")}

        with open(scene_filepath + '/scene_data.json', 'w') as scene_data_file:
            json.dump(scene_data, scene_data_file, indent = 2)
            
        ### HARDCODED FOR SUB-DATASET C: Containment ###
        
        # initialise dictionary for physical_data json file
        physical_data = {'features' : {}, 'prior_rules' : {}, 'posterior_rules' : {}}
        
        # Feature #1 & #2: Object height and container height
        physical_data['features']['object_height'] = obj_height
        physical_data['features']['container_height'] = container_height
        # Feature #3 & #4: Object Width & container width
        physical_data['features']['object_width'] = obj_width
        physical_data['features']['container_width'] = container_height
        # Feature #5 & #6: Object shape and container shape
        physical_data['features']['object_shape'] = obj_shape
        physical_data['features']['container_shape'] = container
        
        # Prior Rule #1: object height vs container height
        if find_highest_point('Container') <= find_height(obj_shape):
            physical_data['prior_rules']['is_object_taller_than_container?'] = False
        else:
            physical_data['prior_rules']['is_object_taller_than_container?'] = True
        
        # checking if object went in
        obj = bpy.context.scene.objects[obj_shape]
        if object_geometry_location(obj_shape)[2] < find_highest_point('Container'):
            fit = True
        elif find_lowest_point(obj_shape) - find_lowest_point('Container') < 0.4:
            fit = True
        else:
            fit = False
        
        # Prior Rule #2: is the object thinner than the container opening (use getting in criteria to determine)
        physical_data['prior_rules']['is_object_thinner_than_container?'] = fit
        
        # Posterior Rule #1: Did the object fit?
        physical_data['posterior_rules']['did_the_object_fit?'] = fit
        
        # Posterior Rule #2: Does the object protude out of the container ?
        if find_highest_point(obj_shape) > find_highest_point('Container'):
            protuded = True
        else:
            protuded = False
        physical_data['posterior_rules']['did_the_object_protude?'] = protuded
        
        if not violation:
            fit_list.append(fit)
            protuded_list.append(protuded)
    
        with open(scene_filepath + '/physical_data.json', 'w') as physical_data_file:
            json.dump(physical_data, physical_data_file, indent = 2)
        
        if render_videos and render_frames:
            # Convert all frames to video
            frames_to_video(scene_filepath, 'rgb', True, False, 'jpg', starting_frame, ending_frame)
            frames_to_video(scene_filepath, 'inst_raw', True, True, 'jpg', starting_frame, ending_frame)
            frames_to_video(scene_filepath, 'depth_raw', True, True, 'jpg', starting_frame, ending_frame)
            if render_vis:
                frames_to_video(scene_filepath, 'inst_vis', 'jpg', starting_frame, ending_frame)
                frames_to_video(scene_filepath, 'depth_vis', 'jpg', starting_frame, ending_frame)
        
if generate_data:               
    outcome_df = pd.DataFrame(fit_list, columns = ['fit'])
    outcome_df['protuded'] = protuded_list
    outcome_df.to_csv(root_filepath + 'outcome.csv', index = False)
    # visualise dataset balance
    print('fit')
    print(outcome_df['fit'].value_counts()) # in terms of trials
    print('protuded')
    print(outcome_df['protuded'].value_counts()) # in terms of trials
    
# visualise some stats
print('obj_height')
print(variation_df['obj_height'].value_counts()) # in terms of scenes
print('container_height')
print(variation_df['container_height'].value_counts()) # in terms of scenes
print('obj_width')
print(variation_df['obj_width'].value_counts()) # in terms of scenes
print('container_width')
print(variation_df['container_width'].value_counts()) # in terms of scenes
print('shape')
print(variation_df['shape'].value_counts()) # in terms of scenes
print('container')
print(variation_df['container'].value_counts()) # in terms of scenes