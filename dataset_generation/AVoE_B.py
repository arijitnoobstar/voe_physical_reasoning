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

# Number of trials
NUM_TRIALS = 5000

# Variation settings filename for upload. Set to None if it does not exist
variation_filename = './B_train_surprising.csv'
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

# occluder settings
occluder_top = 6

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
obj_width_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6]
obj_height_list =  [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6]
occluder_height_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


## comment this***
#shape_list = ['Sphere']
#obj_width_list = [1.4]
#obj_height_list =  [0.5]
#occluder_height_list = [0.5]

# Dataset Proportions
train_prop = 0.75
val_prop = 0.15
test_prop = 0.1

# determine the checkpoint trial for val and test
assert (train_prop + val_prop + test_prop) == 1.0
val_checkpoint = int(train_prop * NUM_TRIALS)
test_checkpoint = int((train_prop + val_prop) * NUM_TRIALS)

# make the data_filepath folders      
root_filepath = 'Data/B_occlusion/'
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
        variation = [trial_count + 1, random.choice(shape_list),\
        random.choice(obj_width_list),random.choice(obj_height_list),\
         random.choice(occluder_height_list)]
            
        # ensure no repeat of an existing trial variation
        if variation[1:] not in variation_settings_duplicate_checker:
            # add to duplicate checker
            variation_settings_duplicate_checker.append(variation[1:])
            # change data_segment based on trial count 
            if trial_count == val_checkpoint:
                data_segment = 'validation'
            elif trial_count == test_checkpoint:
                data_segment = 'test'
                
            # insert data_segment and violation as False (expected case) first
            variation.insert(1, data_segment)
            variation.insert(2, False)
            # add scene for expected case
            variation_settings.append(variation)
            
            # if validation or test set, change violation setting to True (surprising case)
            if data_segment != 'train':  
                variation_duplicate = variation[:]
                variation_duplicate[2] = True
                # add scene for surprising case
                variation_settings.append(variation_duplicate)
            
            # increment trial count
            trial_count += 1

    # use pandas to store data into a csv
    variation_df = pd.DataFrame(variation_settings, columns = ['trial_num', 'data_segment', 'violation',\
    'shape', 'obj_width','obj_height','occluder_height'])
    
    if filter_10_percent:
        variation_df = pd.concat([variation_df[:375],variation_df[3750:3900], variation_df[5250:5350]])

variation_df.to_csv(root_filepath + 'variation_settings.csv', index = False)
# lists to save outcome data from each scene
seen_through_middle_list = []
object_on_right_list = []


# loop through each variation
# NOTE: iterate throught each setting, NOT trial_number as each trial can be made
# from 2 videos (expected and surprising)

for _ , setting in variation_df.iterrows():
    
    # unpack variation settings for this trial
    trial_num, data_segment, violation, obj_shape, obj_width, \
    obj_height, occluder_height = setting
    
    
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
        'Occluder' : 3,
        'Object': 6,
    }

    # Colour mapping in a [string, hex] pair
    colour_mapping = {
        'Ground': ['white', white],
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
        
        
    def create_object(shape, colour, height = 1, direction = 'left'):
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
        bpy.ops.transform.resize(value=(height, height, height), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        # add material/colour to object
        add_material(bpy.context.object, shape, colour)
        
        bpy.context.object.name = shape
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Object']
    
               
    def create_occluder(colour, middle_height = 0.5):
        """
        Create the occluder. Default height_above_ground: 8
        """
        # initialise the occluder and orientate it
        
        # the chunk of code below creates the occluder with middle segment in the range 0 < middle_height < 1
        scale = (5 - 5*middle_height)/(5*middle_height)
        bpy.ops.mesh.primitive_plane_add(size = 5/scale, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 5), "orient_type":'NORMAL', "orient_matrix":((-0, 0, 1), (-1, -0, 0), (0, -1, 0)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":True, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 8/scale), "orient_type":'NORMAL', "orient_matrix":((0, -0, 1), (0, 1, 0), (-1, 0, 0)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":True, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, -5), "orient_type":'NORMAL', "orient_matrix":((-0, 0, 1), (-1, -0, 0), (0, -1, 0)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":True, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.object.editmode_toggle()
        bpy.ops.transform.rotate(value= 3.141592653 / 2, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.resize(value=(scale, 1, 0.5 * ((2*scale)/(1 + scale))), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.transform.translate(value=(4, -2, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        

        # name the occluder accordingly
        obj = bpy.context.scene.objects['Plane']
        bpy.context.view_layer.objects.active = obj
        bpy.context.object.name = 'Occluder'
        
        # add material/colour to occluder
        add_material(bpy.context.object, 'Occluder', colour)
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping['Occluder']
        
    def occluder_move(mode, start_frame, end_frame):
        """
        Move the occluder from from_height to to_height
        """
        occluder_mesh = bpy.context.scene.objects['Occluder']
        # difference between object centre Z location and bottom most point
        difference = occluder_mesh.location[2] - find_lowest_point('Occluder')
        
        if mode == 'down':
            occluder_mesh.location[2] = occluder_top + difference
            occluder_mesh.keyframe_insert(data_path='location', frame=start_frame)
            occluder_mesh.location[2] = difference
            occluder_mesh.keyframe_insert(data_path='location', frame=end_frame)
        else:
            # mode up
            occluder_mesh.location[2] = difference
            occluder_mesh.keyframe_insert(data_path='location', frame=start_frame)
            occluder_mesh.location[2] = occluder_top + difference
            occluder_mesh.keyframe_insert(data_path='location', frame=end_frame)
        
                
    def object_move(object,x1,z1,x2,z2,rigid = True, stop_kinematic = True, start_frame = 20, end_frame = 40):
        """
        Moves object from (x1,z1) to (x2,z2) from start_frame to end_frame
        """
        
        # Spawn object right on top of the ground surface or to z height specified
        object_mesh = bpy.context.scene.objects[object]
        if z1 == 0:
            z1 = object_mesh.location[2] - find_lowest_point(object)
        if z2 == 0:
            z2 = object_mesh.location[2] - find_lowest_point(object)


        # set rigid body setting
        if rigid:
            bpy.context.view_layer.objects.active = object_mesh
            bpy.ops.rigidbody.object_add()
            
        # initialise the object location at start and end
        object_mesh.location = [x1,0,z1]
        object_mesh.keyframe_insert(data_path='location', frame=start_frame)
        object_mesh.location = [x2,0,z2]
        object_mesh.keyframe_insert(data_path='location', frame=end_frame)
        
        if rigid:
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
    
    def find_height(object):
        """
        Finds the height of an object
        """
        return find_highest_point(object) - find_lowest_point(object)
    
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
    create_occluder(colour = colour_mapping['Occluder'][1], middle_height = occluder_height)
    occluder_move('down', 1, 20)
    
    colour_mapping['Object'] = [colour_mapping[obj_shape][0], colour_mapping[obj_shape][1]]
    create_object(obj_shape, colour_mapping['Object'][1], obj_height, direction = 'left')

    # this segment of code has been strictly hardcoded just for AVoE B for 2 expected scenarios and 3 violation scenarios
        

    if violation:
        if seen_through_middle:
            # displacement is amount from centre of object to bottom of object
            displacement = bpy.context.scene.objects[obj_shape].location[2] - find_lowest_point(obj_shape)
            # fake top height is the height for the middle portion
            fake_bottom_height = -5
            object_move(obj_shape,-12.5,displacement,-4,displacement,rigid = True, stop_kinematic = False, start_frame = 15, end_frame = 24)
            object_move(obj_shape,-4,displacement,-4,fake_bottom_height,rigid = True, stop_kinematic = False, start_frame = 25, end_frame = 25)
            object_move(obj_shape,-4,fake_bottom_height,4.75,fake_bottom_height,rigid = True, stop_kinematic = False, start_frame = 26, end_frame = 34)
            object_move(obj_shape,4.75,fake_bottom_height,4.75,displacement,rigid = True, stop_kinematic = False, start_frame = 35, end_frame = 35)
            object_move(obj_shape,4.75,displacement,10,displacement,rigid = True, stop_kinematic = False, start_frame = 36, end_frame = 40)
        else:
            # displacement is amount from centre of object to bottom of object
            displacement = bpy.context.scene.objects[obj_shape].location[2] - find_lowest_point(obj_shape)
            # fake top height is the height for the middle portion
            fake_top_height = occluder_height * 5 - displacement * 1.5
            object_move(obj_shape,-12.5,displacement,-4,displacement,rigid = True, stop_kinematic = False, start_frame = 15, end_frame = 24)
            object_move(obj_shape,-4,displacement,-4,fake_top_height,rigid = True, stop_kinematic = False, start_frame = 25, end_frame = 25)
            object_move(obj_shape,-4,fake_top_height,4.75,fake_top_height,rigid = True, stop_kinematic = False, start_frame = 26, end_frame = 34)
            object_move(obj_shape,4.75,fake_top_height,4.75,displacement,rigid = True, stop_kinematic = False, start_frame = 35, end_frame = 35)
            object_move(obj_shape,4.75,displacement,10,displacement,rigid = True, stop_kinematic = False, start_frame = 36, end_frame = 40)
    else:
        object_move(obj_shape,-12.5,0,10,0,rigid = True, stop_kinematic = False, start_frame = 15, end_frame = 40)

    # bake object physics to keyframes
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects[obj_shape]
    obj.select_set(True)
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
        
        # SPECIFIC to AVoE B, we need a obj_presence list to determine if object went throught the middle    
        obj_presence = []
        
        # start rendering from 'starting_frame' till 'ending_frame'
        for frame_num in range(starting_frame, ending_frame):
            # set blender frame and normalise frame_num to start from 1 onwards
            scene.frame_set(frame_num)
            frame_num -= (starting_frame - 1)

            if render_frames:
                # render using bpycv           
                result = bpycv.render_data(render_annotation = True)
                # save raw rgb, depth (non-vis) and instance segmentation (non-vis) images
                cv2.imwrite(scene_filepath + "rgb_{}.jpg".format(frame_num), result["image"][..., ::-1])  
                cv2.imwrite(scene_filepath + "depth_raw_{}.jpg".format(frame_num), np.uint16(result["depth"]))
                cv2.imwrite(scene_filepath + "inst_raw_{}.jpg".format(frame_num), np.uint16(result["inst"]))
                
                # AVoE B hardcoded part (necessary to determine middle portion outcome)
                inst = result["inst"]
                unique, _ = np.unique(inst, return_inverse=True)
                if 6 in unique:
                    obj_presence.append(True)
                else:
                    obj_presence.append(False)
                
                
            # and condition needed as videos cannot be rendered without frames
            if render_vis and render_frames:
                # convert instance segmentation raw into a visual image using the boxx python package
                inst = result["inst"]
                if frame_num  == 1:
                    # hardcode to 4 for AVoE B as not all objects will appear at start
                    instance_colour_mapping = boxx.getDefaultColorList(4, includeBackGround=True)
                    
                # map unique instance ids to a visual image
                height, width = inst.shape
                inst_vis = np.zeros(shape = (height, width, 3))            
                for i in range(height):
                    for j in range(width):
                       # hardcoded part for AVoE B as indexes are not numbered nicely
                        if inst[i,j] == 3:
                            index = 1
                        if inst[i,j] == 6:
                            index = 2
                        elif inst[i,j] == 7:
                            index = 3
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
                if entity == 'Ground':
                    # no need for pose/orientation data for ground
                    continue
                elif entity == 'Object':
                    # object has name specific to object shape
                    obj = bpy.context.scene.objects[obj_shape]
                else:
                    if entity in bpy.context.scene.objects:
                        obj = bpy.context.scene.objects[entity]
                    else:
                        continue
                    
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
            
            
        ### HARDCODED FOR SUB-DATASET B: occlusion ###
        
        # initialise dictionary for physical_data json file
        physical_data = {'features' : {}, 'prior_rules' : {}, 'posterior_rules' : {}}
        
        # Feature #1 & #2: Object Height & Object Width
        physical_data['features']['object_height'] = obj_height
        physical_data['features']['object_width'] = obj_width
        # Feature #3: Object Shape
        physical_data['features']['object_shape'] = obj_shape
        # Feature #4: Middle Segment Height
        physical_data['features']['middle_segment_height'] = occluder_height
        
        if render_frames:
            # find out if object appeared in the middle
            count = 0 # object always shows up in the middle
            for i in range(0, len(obj_presence) - 1):
                if (obj_presence[i] != obj_presence[i+1]):
                    count += 1
            if count >= 4:
                seen_through_middle = True
            else:
                seen_through_middle = False
        else:
            # just a placeholder value when rendering is not done
            seen_through_middle = False

        # find out if any object appeared on the right of the occluder
        if object_geometry_location(obj_shape)[0] > find_rightmost_point('Occluder'):
            object_on_right = True
        else:
            object_on_right = False
            
        if not violation and render_frames:
            seen_through_middle_list.append(seen_through_middle)
        if not violation:
            object_on_right_list.append(object_on_right)
        
        # Prior Rule #1: is the object taller
        if violation:
            physical_data['prior_rules']['is_object_taller_than_middle?'] = not seen_through_middle
        else:
            physical_data['prior_rules']['is_object_taller_than_middle?'] = seen_through_middle
                        
        # Posterior Rule #1: see object in middle?
        physical_data['posterior_rules']['see_object_in_middle?'] = seen_through_middle
    
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
        
if generate_data and render_frames: # need the second condition as seen_through_middle needs rendered frame access  
    outcome_df = pd.DataFrame(seen_through_middle_list, columns = ['seen_through_middle'])
    outcome_df['object_on_right'] = object_on_right_list
    outcome_df.to_csv(root_filepath + 'outcome.csv', index = False)
    # visualise dataset balance
    print('seen_through_middle')
    print(outcome_df['seen_through_middle'].value_counts()) # in terms of trials
    print('object_on_right')
    print(outcome_df['object_on_right'].value_counts()) # in terms of trials


# visualise some stats
print('obj_width')
print(variation_df['obj_width'].value_counts()) # in terms of scenes
print('obj_height')
print(variation_df['obj_height'].value_counts()) # in terms of scenes
print('shape')
print(variation_df['shape'].value_counts()) # in terms of scenes
print('middle_segment_height')
print(variation_df['occluder_height'].value_counts()) # in terms of scenes