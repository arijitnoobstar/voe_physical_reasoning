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
import time
import csv
import os
import shutil
import filecmp

# Number of trials
NUM_TRIALS = 5000

# Variation settings filename for upload. Set to None if it does not exist
variation_filename = './E_variation_settings.csv'
filter_10_percent = False

# Colour in HEX
red=0xF90707 
yellow=0xFFFF00 
blue=0x0000FF
lime=0x00FF00
orange = 0xFFA500
cyan=0x00FFFF
green = 0x00FF00
silver= 0x808080
black = 0x000000
purple = 0x800080
white = 0xFFFFFF
metal=0x8A8A8A

# Occluder heights
occluder_top = 10.3
occluder_bottom = 3.3

# rendering settings
generate_data = True
starting_frame = 20 # THIS ONLY CHANGES RENDER FRAMES
ending_frame = 70 # THIS ONLY CHANGES RENDER FRAMES
delete_folder = True # change to False to not override data
render_frames = True
render_vis = False # set to False for actual dataset generation
render_videos = True

# dataset variation values
shape_list = ['Cube', 'Cylinder', 'Torus', 'Sphere', 'Cone', 'Side_Cylinder', 'Inverted_Cone']
wall_list = ['Normal', 'Open', 'Soft']
obj_width_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
opening_width_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
obj_height_list =  [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
opening_height_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

## comment this***
#shape_list = ['Sphere']
#wall_list = ['Open']
#obj_width_list = [0.4]
#obj_height_list =  [0.4]
#opening_width_list = [1.4]
#opening_height_list = [1.4]

# Dataset Proportions
train_prop = 0.75
val_prop = 0.15
test_prop = 0.1

# determine the checkpoint trial for val and test
assert (train_prop + val_prop + test_prop) == 1.0
val_checkpoint = int(train_prop * NUM_TRIALS)
test_checkpoint = int((train_prop + val_prop) * NUM_TRIALS)

# make the data_filepath folders      
root_filepath = 'Data/E_barrier/'
filepath_extensions = ['train/expected/', 'validation/surprising/', 'validation/expected/', \
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
    
    # generate all random variations first
    while trial_count < NUM_TRIALS:
        variation = [random.choice(shape_list),random.choice(wall_list)\
        ,random.choice(obj_width_list),random.choice(obj_height_list)]
        
        # if wall type is not open, then the opening width and height are 0
        if variation[1] == 'Open':
            variation.extend([random.choice(opening_width_list), random.choice(opening_height_list)])
        else:
            variation.extend([0,0])
            
        # ensure no repeat of an existing trial variation
        if variation not in variation_settings_duplicate_checker:
            # add to duplicate checker
            variation_settings_duplicate_checker.append(variation)
            variation_settings.append(variation)
            # increment trial count
            trial_count += 1
            
    # randomly shuffle variations to even out distribution among train, val & test
    random.shuffle(variation_settings)
    # assign variations to train, val and test and add violation scene for val & test
    trial_count, scene_count = 0, 0
    while trial_count < NUM_TRIALS:
        # change data_segment based on trial count 
        if trial_count == val_checkpoint:
            data_segment = 'validation'
        elif trial_count == test_checkpoint:
            data_segment = 'test'
        # insert data_segment and violation as False (expected case) first
        variation_settings[scene_count].insert(0, trial_count + 1)
        variation_settings[scene_count].insert(1, data_segment)
        variation_settings[scene_count].insert(2, False)
        
        # if validation or test set, change violation setting to True (surprising case)
        if data_segment != 'train':  
            variation_duplicate = variation_settings[scene_count][:]
            variation_duplicate[2] = True
            # add scene for surprising case
            variation_settings.insert(scene_count + 1, variation_duplicate)
            scene_count += 2
        else:
            scene_count += 1
            
        trial_count += 1

    # use pandas to store data into a csv
    variation_df = pd.DataFrame(variation_settings, columns = ['trial_num', 'data_segment', 'violation',\
    'shape','wall_type','obj_width', 'obj_height','opening_width','opening_height'])
    
    if filter_10_percent:
        variation_df = pd.concat([variation_df[:375],variation_df[3750:3900], variation_df[5250:5350]])
    
variation_df.to_csv(root_filepath + 'variation_settings.csv', index = False)
# lists to save outcome data from each scene
passed_through_wall_list = []

# loop through each variation
# NOTE: iterate throught each setting, NOT trial_number as each trial can be made
# from 2 videos (expected and surprising)
for _, setting in variation_df.iterrows():
    
    # unpack variation settings for this trial
    trial_num, data_segment, violation, obj_shape, wall_type, obj_width, obj_height, \
    opening_width, opening_height = setting
    # scale opening width & height according to 3D mesh scale,\
    # so that scale of 1 is the same for both obj and opening
    opening_width *= 1.345
    opening_height *= 1
    
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
        wall_type + '_Wall' : 2,
        'Occluder' : 3
    }

    # Colour mapping in a [string, hex] pair
    colour_mapping = {
        'Ground': ['white', white],
        wall_type + '_Wall' : ['blue', blue],
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
        bpy.ops.transform.translate(value=(-5.26466, -5, -0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # assign energy
        bpy.context.object.data.energy = 400
        
        # rename light accordingly
        bpy.context.object.name = 'Light'

        
    def create_ground(x_size, y_size, colour):
        """
        Create a ground of size (x_size, y_size, 1)
        """
        # add ground and size accordingly
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(x_size, y_size, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # make ground a right body in passive mode
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.type = 'PASSIVE'
        
        # rename plane to 'Ground'
        bpy.context.object.name = 'Ground'
        
        # add the material (colour) to the ground
        add_material(bpy.context.object, "Ground", colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Ground']
        
        
    def create_object(shape, colour, x_scale = 1, y_scale = 1, z_scale = 1):
        """
        Create an object with specified shape, scales and colour
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

        # give collision modifier
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Object']

    def create_cloth_support(colour, wall_type):
        """
        Create a solid wall of specified height (default: 1.54478)
        """
        # create pivot_rod and resize + translate
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(0.1, 5.5, 0.1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        # position it properly
        bpy.ops.transform.translate(value=(0.2, 0, 5.15), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        # name the pivot_rod accordingly
        bpy.context.object.name = 'pivot_rod'
        
        # add material/colour to pivot_rod
        add_material(bpy.context.object, wall_type + 'pivot_rod', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[wall_type + '_Wall']

        # create support_rod and resize + translate
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(0.1, 0.1, 5.2), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        # position it properly
        bpy.ops.transform.translate(value=(0.2, 2.8, 5.2/2), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        # name the support_rod accordingly
        bpy.context.object.name = 'support_rod'
        
        # add material/colour to support_rod
        add_material(bpy.context.object, wall_type + 'pivot_rod', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[wall_type + '_Wall']

    def create_wall(colour, rigid = True):
        """
        Create a solid wall of specified height (default: 1.54478)
        """
        # create wall and resize + translate
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.resize(value=(1, 4, 2.15), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        # position it properly
        bpy.ops.transform.translate(value=(0, 0, -1 * find_lowest_point('Cube')), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        if rigid:
            # make it a rigid body and instantiate with high mass to make it into an obstacle
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.mass = 10000
        
        # name the wall accordingly
        bpy.context.object.name = wall_type + '_Wall'
        
        # add material/colour to wall
        add_material(bpy.context.object, wall_type + '_Wall', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[wall_type + '_Wall']
    
    def create_open_wall(colour, rigid = True, width_scale = 1, height_scale = 1, allow_through = True):
        """
        Create a solid wall of specified height (default: 1) and a centre opening
        """
        bpy.ops.import_scene.fbx( filepath = './3DModels/wall_with_opening.fbx')
        # make it a rigid body and instantiate with high mass to make it into an obstacle
        bpy.ops.transform.resize(value=(2.5, width_scale, height_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(0, 0, 5), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(0, 0, -1 * find_lowest_point('Cube')), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        
        ob = bpy.context.scene.objects['Cube']
        bpy.context.view_layer.objects.active = ob
        if rigid:
            # make it a rigid body and instantiate with mass to give it strong wall property
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.mass = 100
            # use MESH collision shape to allow opening to have no collision causing mesh
            if allow_through:
                bpy.context.object.rigid_body.collision_shape = 'MESH'

        # name the wall accordingly
        bpy.context.object.name = wall_type + '_Wall'
        
        # position it properly
#        bpy.ops.transform.translate(value=(0, 0, -1 * find_lowest_point(wall_type + '_Wall')), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    
        # add material/colour to wall
        add_material(bpy.context.object, wall_type + '_Wall', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[wall_type + '_Wall']    
               
    def create_cloth_wall(colour):
        """
        Make cloth wall
        """
        # create wall and resize and orientate
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0.2))
        bpy.ops.transform.rotate(value=-1.51376, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(0, 0, 2.36049), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.resize(value=(2.54746, 2.54746, 2.54746), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # edit mesh to have cuts and modify to change to softbody cloth and edit quality and tension stiffness
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.subdivide(number_cuts=20)
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.modifier_add(type='CLOTH')
        bpy.context.object.modifiers["Cloth"].settings.quality = 8
        bpy.context.object.modifiers["Cloth"].settings.tension_stiffness = 12
        
        # Next, pin the designated vertices.
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = bpy.context.object
        gripped_group = bpy.context.object.vertex_groups.new(name='Pinned')
        pinned_vertices = [1,3]
#        pinned_vertices.extend([x for x in range(44,64,8)]) # optional to add more vertices
        gripped_group.add(pinned_vertices, 1.0, 'ADD')
        bpy.context.object.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'

        # make cloth a passive body and change collision shape to mesh
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.type = 'PASSIVE'
        bpy.context.object.rigid_body.collision_shape = 'MESH'
        bpy.context.object.rigid_body.mesh_source = 'BASE'
        
        
        # name the cloth wall accordingly
        bpy.context.object.name = wall_type + '_Wall'
        
        # add material/colour to wall
        add_material(bpy.context.object, wall_type + '_Wall', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[wall_type + '_Wall']
        
#        # find group of vertices not attached to the pivot rod
#        ungripped_group = bpy.context.object.vertex_groups.new(name='unpinned')
#        unpinned_vertices = [x.index for x in bpy.context.object.data.vertices if (x.index!=1 and x.index!=3)]
#        gripped_group.add(unpinned_vertices, 1.0, 'ADD')
        
        # make the cloth wave
        bpy.ops.object.modifier_add(type='WAVE')
#        bpy.context.object.modifiers["Wave"].vertex_group = "unpinned"
        bpy.context.object.modifiers["Wave"].width = 1
        bpy.context.object.modifiers["Wave"].narrowness = 0.5
        bpy.context.object.modifiers["Wave"].use_x = False
        bpy.context.object.modifiers["Wave"].start_position_x = -4
        bpy.context.object.modifiers["Wave"].speed = 0.1
        

    def create_occluder(colour, height_above_ground = 8):
        """
        Create the occluder. Default height_above_ground: 8
        """
        # initialise the occluder and orientate it
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        bpy.ops.transform.translate(value=(-0, -5, height_above_ground), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, True, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.resize(value=(4.5, 3.2, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, True, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.rotate(value=-1.56857, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        
        # name the occluder accordingly
        bpy.context.object.name = 'Occluder'
        
        # add material/colour to occluder
        add_material(bpy.context.object, 'Occluder', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Occluder']
        
    def occluder_move(from_height, to_height, start_frame, end_frame):
        """
        Move the occluder from from_height to to_height
        """
        occluder_mesh = bpy.context.scene.objects['Occluder']
        occluder_mesh.location[2] = from_height
        occluder_mesh.keyframe_insert(data_path='location', frame=start_frame)
        occluder_mesh.location[2] = to_height
        occluder_mesh.keyframe_insert(data_path='location', frame=end_frame)
        
                
    def object_move(object,x1,y1,x2,y2,rigid = True, stop_kinematic = True, start_frame = 20, end_frame = 40):
        """
        Moves object from (x1,y1) to (x2,y2) from start_frame to end_frame
        """
        
        # Spawn object right on top of the ground surface
        object_mesh = bpy.context.scene.objects[object]
        spawn_height = object_mesh.location[2] - find_lowest_point(object) + \
        find_lowest_point('Ground')

        # set rigid body setting
        if rigid:
            bpy.context.view_layer.objects.active = object_mesh
            bpy.ops.rigidbody.object_add()
            
        # initialise the object location at start and end
        object_mesh.location = [x1,y1,spawn_height]
        object_mesh.keyframe_insert(data_path='location', frame=start_frame)
        object_mesh.location = [x2,y2,spawn_height]
        object_mesh.keyframe_insert(data_path='location', frame=end_frame)
        
        # set kinematic settings for rigid body
        object_mesh.rigid_body.kinematic = True
        object_mesh.keyframe_insert(data_path='rigid_body.kinematic', frame=start_frame)
        if stop_kinematic:
            object_mesh.rigid_body.kinematic = False
            object_mesh.keyframe_insert(data_path='rigid_body.kinematic', frame=end_frame)

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

    # create light, plane and occluder
    create_light()
    create_ground(100, 100, colour = colour_mapping['Ground'][1])
    create_occluder(colour = colour_mapping['Occluder'][1], height_above_ground = occluder_top)
    stop_kinematic = True
    obj_move_end_frame = 50
    x1, x2 = -11.5, -4.5

    #### create wall & account for violation settings ####
    # normal wall
    if wall_type == 'Normal':
        if violation:
            set_rigid = False
        else:
            set_rigid = True
        create_wall(colour = colour_mapping[wall_type + '_Wall'][1], rigid = set_rigid)
        
    # open wall
    elif wall_type == 'Open':
        if violation:
            if passed_through_wall:
                allow_through = False
                set_rigid = True
            else:
                allow_through = True
                set_rigid = False
        else:
            set_rigid = True
            allow_through = True
        create_open_wall(colour = colour_mapping[wall_type + '_Wall'][1],\
        rigid = set_rigid, width_scale = opening_width, height_scale = opening_height, allow_through = allow_through)
    
            
    # soft wall
    elif wall_type == 'Soft':
        if violation:
            pass
        else:
            # update x2 and obj_move_end_frame
            x2 = 1.5
            obj_move_end_frame = 58
        create_cloth_wall(colour = colour_mapping[wall_type + '_Wall'][1])
        create_cloth_support(colour = black, wall_type = wall_type)
        
    # create object
    colour_mapping['Object'] = [colour_mapping[obj_shape][0], colour_mapping[obj_shape][1]]
    create_object(obj_shape, colour_mapping['Object'][1], 1, obj_width, obj_height)
        
    # move object and occluder
    occluder_move(occluder_top, occluder_bottom, 30, 40)   
    object_move(obj_shape,x1,0,x2,0, True, stop_kinematic, 40, obj_move_end_frame)
    occluder_move(occluder_bottom, occluder_top, 60, 70)
    
    # bake object physics to keyframes
    bpy.context.view_layer.objects.active =  bpy.context.scene.objects[obj_shape]
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
        
        for frame_num in range(starting_frame, ending_frame):
            # set blender frame and normalise frame_num to start from 1 onwards
            scene.frame_set(frame_num)
            
        # start rendering from 'starting_frame' till 'ending_frame'
        for frame_num in range(starting_frame, ending_frame):
            # set blender frame and normalise frame_num to start from 1 onwards
            scene.frame_set(frame_num)
            frame_num -= (starting_frame - 1)

            if render_frames:         
                result = bpycv.render_data(render_annotation = True)
#                # save raw rgb, depth (non-vis) and instance segmentation (non-vis) images 
                cv2.imwrite(scene_filepath + "rgb_{}.jpg".format(frame_num), result["image"][..., ::-1])  
                cv2.imwrite(scene_filepath + "depth_raw_{}.jpg".format(frame_num), np.uint16(result["depth"]))
                cv2.imwrite(scene_filepath + "inst_raw_{}.jpg".format(frame_num), np.uint16(result["inst"]))
                
            # and condition needed as videos cannot be rendered without frames
            if render_vis and render_frames:
                # convert instance segmentation raw into a visual image using the boxx python package
                inst = result["inst"]
                if frame_num  == 1:
                    unique, _ = np.unique(inst, return_inverse=True)
                # hardcode to 4 items for AVoE E as object is not available in first frame
                    instance_colour_mapping = boxx.getDefaultColorList(4, includeBackGround=True)
                # map unique instance ids to a visual image
                height, width = inst.shape
                inst_vis = np.zeros(shape = (height, width, 3))            
                for i in range(height):
                    for j in range(width):
                        inst_vis[i,j,:] = instance_colour_mapping[inst[i,j]]
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
                if entity == 'Ground':
                    # no need for pose/orientation data for ground
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
            
        ### HARDCODED FOR SUB-DATASET E: barrier ###
        
        # initialise dictionary for physical_data json file
        physical_data = {'features' : {}, 'prior_rules' : {}, 'posterior_rules' : {}}
        
        # Feature #1: Wall Opening
        if wall_type == "Open":    
            physical_data['features']['wall_opening'] = True 
        else:
            physical_data['features']['wall_opening'] = False
        # Feature #2: Wall Soft
        if wall_type == "Soft":
            physical_data['features']['wall_soft'] = True
        else:
            physical_data['features']['wall_soft'] = False
        # Feature #3 & #4: Object Width & Opening Width
        physical_data['features']['object_width'] = obj_width
        physical_data['features']['opening_width'] = opening_width/1.345
        # Feature #5 & #6: Object Height & Opening Height
        physical_data['features']['object_height'] = obj_height
        physical_data['features']['opening_height'] = opening_height
        # Feature #7: Object Shape
        physical_data['features']['object_shape'] = obj_shape
        
        # Prior Rule #1: Is there an opening?
        physical_data['prior_rules']['is_there_an_opening?'] = \
        physical_data['features']['wall_opening']
        
        # Prior Rule #2: Is there an opening?
        physical_data['prior_rules']['is_the_blocker_soft?'] = \
        physical_data['features']['wall_soft']
        
        # Prior Rule #3: Is the object thinner than the opening?
        if obj_width < opening_width:
            physical_data['prior_rules']['is_the_object_thinner_than_the_opening?'] = True
        else:
            physical_data['prior_rules']['is_the_object_thinner_than_the_opening?'] = False

        # Prior Rule #4: Is the object shorter than the opening?
        if obj_height < opening_height:
            physical_data['prior_rules']['is_the_object_shorter_than_the_opening?'] = True
        else:
            physical_data['prior_rules']['is_the_object_shorter_than_the_opening?'] = False
                        
        # Posterior Rule #1: Did the object pass through the wall?
        obj = bpy.context.scene.objects[obj_shape]
        # passed_through_wall bool used to determine outcome of violation case
        if obj.location[0] > 0.8: # right side of wall
            passed_through_wall = True
            physical_data['posterior_rules']['did_the_object_pass_through_the_wall?'] = True
        else:
            passed_through_wall = False
            physical_data['posterior_rules']['did_the_object_pass_through_the_wall?'] = False
    
        with open(scene_filepath + '/physical_data.json', 'w') as physical_data_file:
            json.dump(physical_data, physical_data_file, indent = 2)
            
        if not violation:
            passed_through_wall_list.append(passed_through_wall)
        
        if render_videos and render_frames:
            # Convert all frames to video
            frames_to_video(scene_filepath, 'rgb', True, False, 'jpg', starting_frame, ending_frame)
            frames_to_video(scene_filepath, 'inst_raw', True, True, 'jpg', starting_frame, ending_frame)
            frames_to_video(scene_filepath, 'depth_raw', True, True, 'jpg', starting_frame, ending_frame)
            if render_vis:
                frames_to_video(scene_filepath, 'inst_vis', 'jpg', starting_frame, ending_frame)
                frames_to_video(scene_filepath, 'depth_vis', 'jpg', starting_frame, ending_frame)
        
if generate_data:               
    outcome_df = pd.DataFrame(passed_through_wall_list, columns = ['passed_through_wall'])
    outcome_df.to_csv(root_filepath + 'outcome.csv', index = False)
    # visualise dataset balance
    print('passed_through_wall')
    print(outcome_df['passed_through_wall'].value_counts()) # in terms of trials

# visualise some stats
print('obj_height')
print(variation_df['obj_height'].value_counts()) # in terms of scenes
print('opening_height')
print(variation_df['opening_height'].value_counts()) # in terms of scenes
print('obj_width')
print(variation_df['obj_width'].value_counts()) # in terms of scenes
print('opening_width')
print(variation_df['opening_width'].value_counts()) # in terms of scenes
print('shape')
print(variation_df['shape'].value_counts()) # in terms of scenes
print('wall_type')
print(variation_df['wall_type'].value_counts()) # in terms of scenes