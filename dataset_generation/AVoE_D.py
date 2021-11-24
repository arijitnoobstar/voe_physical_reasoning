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
variation_filename = './D_variation_settings.csv'
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
render_vis = False
render_videos = True # set to False for actual dataset generation

# dataset variation values
shape_list = ['Cube', 'Cylinder', 'Sphere', 'Cone', 'Side_Cylinder', 'Inverted_Cone']
left_height_list = [0.5, 0.75, 1, 1.25, 1.5]
right_height_list = [0.5, 0.75, 1, 1.25, 1.5]
left_speed_list = [0.5, 1, 1.5, 2, 2.5]
right_speed_list =  [0.5, 1, 1.5, 2, 2.5]

# frame for pre-collision object speedup
speedup_frame = 5

## comment this***
#shape_list = ['Cylinder']
#left_height_list = [1]
#right_height_list = [1]
#left_speed_list = [1]
#right_speed_list =  [1]

# Dataset Proportions
train_prop = 0.75
val_prop = 0.15
test_prop = 0.1

# determine the checkpoint trial for val and test
assert (train_prop + val_prop + test_prop) == 1.0
val_checkpoint = int(train_prop * NUM_TRIALS)
test_checkpoint = int((train_prop + val_prop) * NUM_TRIALS)


# make the data_filepath folders      
root_filepath = 'Data/D_collision/'
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
    while trial_count < NUM_TRIALS:
        
        variation = [trial_count + 1, random.choice(shape_list), random.choice(shape_list),\
         random.choice(left_height_list), random.choice(right_height_list),\
         random.choice(left_speed_list), random.choice(right_speed_list), 'N.A.']
         
    #    # to hardcode right object
    #    variation[2] = 'Cube' 
            
        # ensure no repeat of an existing trial variation
        if variation[1:-1] not in variation_settings_duplicate_checker:
            # add to duplicate checker
            variation_settings_duplicate_checker.append(variation[1:-1])
            # change data_segment based on trial count 
            if trial_count == val_checkpoint:
                data_segment = 'validation'
            elif trial_count == test_checkpoint:
                data_segment = 'test'
                
            # can consider for next version of AVoE
#            # no violation case for bigger object being slower
#            if data_segment != 'train':
#                if (variation[3] > variation[4] and variation[5] < variation[6]) \
#                or (variation[3] < variation[4] and variation[5] > variation[6]):
#                    continue
            if (variation[3] != variation[4] and variation[5] != variation[6]):
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
                # set rand_violation_choice when both objects have the same speed and size
                if (variation[5] == variation[6] and variation[7] == variation[8]):
                    variation_duplicate[-1] = random.choice(['left', 'right'])
                # add scene for surprising case
                variation_settings.append(variation_duplicate)
            
            # increment trial count
            trial_count += 1

    # use pandas to store data into a csv
    variation_df = pd.DataFrame(variation_settings, columns = ['trial_num', 'data_segment', 'violation',\
    'left_shape', 'right_shape', 'left_height','right_height', 'left_speed','right_speed', 'rand_violation_choice'])
    
    if filter_10_percent:
        variation_df = pd.concat([variation_df[:375],variation_df[3750:3900], variation_df[5250:5350]])
    
variation_df.to_csv(root_filepath + 'variation_settings.csv', index = False)

# lists to save outcome data from each scene
left_direction_change_list = []
right_direction_change_list = []
left_magnitude_higher_list = []
right_magnitude_higher_list = []

# loop through each variation
# NOTE: iterate throught each setting, NOT trial_number as each trial can be made
# from 2 videos (expected and surprising)
for _, setting in variation_df.iterrows():
    
    # unpack variation settings for this trial
    trial_num, data_segment, violation, left_shape, right_shape, left_height, right_height, left_speed,\
    right_speed, rand_violation_choice = setting
    
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
        'Left_Object': 6,
        'Right_Object' : 7, # instance mapping is hardcoded consistently across all AVoE A-E
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
        bpy.ops.transform.translate(value=(-1.26466, -0, -0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

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
        
        # rename object name to name of shape
        if direction == 'left':
            label = 'Left_Object'
        else:
            label = 'Right_Object'
        
        bpy.context.object.name = label
        
        # create object instance id
        bpy.context.object["inst_id"] = instance_mapping[label]
        
    def object_move(object,x1,y1,x2,y2,rigid = True, stop_kinematic = True, start_frame = 1, end_frame = 30, mass = 1):
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
            bpy.context.object.rigid_body.mass = mass
            # no friction for collision
            bpy.context.object.rigid_body.friction = 0
            
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
    
    def find_collision_frame():
        """ 
        find the frame in which the objects collide
        """
              
        for frame_num in range(starting_frame, ending_frame):
            bpy.context.scene.frame_set(frame_num)
            if frame_num == starting_frame:
                gap = find_leftmost_point('Right_Object') - find_rightmost_point('Left_Object')
                gap_diff = 0.001
            else:
                gap_diff = gap - (find_leftmost_point('Right_Object') - find_rightmost_point('Left_Object'))
                gap = gap - gap_diff
                
            if gap_diff <= 0 or gap < 0:
                collision_frame = frame_num
                break
            
        return collision_frame
    
    def return_direction(scene, object, start_frame, end_frame, current_frame):
        """
        returns the vectorised direction of movement based on the object positions between frames
        """
        obj = bpy.context.scene.objects[object]
        scene.frame_set(start_frame)
        x1 = obj.location[0]
        scene.frame_set(end_frame)
        x2 = obj.location[0]
        scene.frame_set(current_frame)
        
        if x1 - x2 > 0.01:
            return 'left'
        elif x2 - x1 > 0.01:
            return 'right'
        else:
            return 'stationary'

    def return_vel(scene, object, start_frame, end_frame, current_frame):
        """
        returns an arbitrarily scaled magnitude of object velocity between frames
        """
        obj = bpy.context.scene.objects[object]
        scene.frame_set(start_frame)
        x1 = obj.location[0]
        scene.frame_set(end_frame)
        x2 = obj.location[0]
        scene.frame_set(current_frame)
        
        return x2 - x1
        
    
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

    # create light and ground
    create_light()
    create_ground(100, 100, colour = colour_mapping['Ground'][1], rigid = True)
    # create left object
    colour_mapping['Left_Object'] = [colour_mapping[left_shape][0], colour_mapping[left_shape][1]]
    create_object(left_shape, colour_mapping['Left_Object'][1], left_height, direction = 'left')
    # create right object
    colour_mapping['Right_Object'] = [colour_mapping[right_shape][0], colour_mapping[right_shape][1]]
    create_object(right_shape, colour_mapping['Right_Object'][1], right_height, direction = 'right')
    
    left_displacement = 3 * left_speed/4
    right_displacement = 3 * right_speed/4
    if violation:
        # scenario B & C violations (swap masses)
        if (left_height > right_height and left_speed >= right_speed) or \
        (left_height < right_height and left_speed <= right_speed):
            object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
             start_frame = 1, end_frame = speedup_frame, mass = right_height * right_height * right_height)
            object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
             start_frame = 1, end_frame = speedup_frame, mass = left_height * left_height * left_height)
        # scenario D left faster case violation
        elif (left_height == right_height and left_speed > right_speed):
            object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
             start_frame = 1, end_frame = speedup_frame, mass = left_height * left_height * left_height)
            object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
             start_frame = 1, end_frame = speedup_frame, mass = 100 * right_height * right_height * right_height)
        # scenario D right faster case violation
        elif (left_height == right_height and left_speed < right_speed):
            object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
            start_frame = 1, end_frame = speedup_frame, mass = 100 * left_height * left_height * left_height)
            object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
            start_frame = 1, end_frame = speedup_frame, mass = right_height * right_height * right_height)
        # scenario E violation case (choose left or right randomly)
        elif (left_height == right_height and left_speed == right_speed):
            
            # left becomes heavier (50% chance)
            if rand_violation_choice == 'left':
                object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
                start_frame = 1, end_frame = speedup_frame, mass = 100 * left_height * left_height * left_height)
                object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
                start_frame = 1, end_frame = speedup_frame, mass = right_height * right_height * right_height)
                
                # make heavier obect stop suddenly during contact
                collision_frame = find_collision_frame()             
                object_mesh = bpy.context.scene.objects['Left_Object']
                bpy.ops.object.select_all(action='DESELECT')
                object_mesh.select_set(True)
                bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=collision_frame, step=1)
                bpy.context.scene.frame_set(collision_frame-1)
                bpy.context.view_layer.objects.active = object_mesh
                bpy.ops.rigidbody.object_add() 
                object_mesh.keyframe_insert(data_path='location', frame=collision_frame)
                object_mesh.keyframe_insert(data_path='location', frame=ending_frame)
                object_mesh.rigid_body.kinematic = True
                object_mesh.keyframe_insert(data_path='rigid_body.kinematic', frame=speedup_frame)
                
                

            # right becomes heavier (50% chance)
            else:
                object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
                start_frame = 1, end_frame = speedup_frame, mass = left_height * left_height * left_height)
                object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
                start_frame = 1, end_frame = speedup_frame, mass = 100 * right_height * right_height * right_height)
                
                # make heavier obect stop suddenly during contact
                collision_frame = find_collision_frame()
                object_mesh = bpy.context.scene.objects['Right_Object']
                bpy.ops.object.select_all(action='DESELECT')
                object_mesh.select_set(True)
                bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=collision_frame, step=1)
                bpy.context.scene.frame_set(collision_frame-1)
                bpy.context.view_layer.objects.active = object_mesh
                bpy.ops.rigidbody.object_add() 
                object_mesh.keyframe_insert(data_path='location', frame=collision_frame)
                object_mesh.keyframe_insert(data_path='location', frame=ending_frame)
                object_mesh.rigid_body.kinematic = True
                object_mesh.keyframe_insert(data_path='rigid_body.kinematic', frame=speedup_frame)
                
    else:
        # expected case
        object_move('Left_Object',-5, 0, -5 + left_displacement, 0, rigid = True, stop_kinematic = True,\
        start_frame = 1, end_frame = speedup_frame, mass = left_height * left_height * left_height)
        object_move('Right_Object',5, 0, 5 - right_displacement, 0, rigid = True, stop_kinematic = True,\
        start_frame = 1, end_frame = speedup_frame, mass = right_height * right_height * right_height)

    # bake object physics to keyframes
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects['Left_Object']
    obj.select_set(True)
    obj = bpy.context.scene.objects['Right_Object']
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
               
        # start rendering from 'starting_frame' till 'ending_frame'
        for frame_num in range(starting_frame, ending_frame):
            
            # set blender frame and normalise frame_num to start from 1 onwards
            scene.frame_set(frame_num)
            actual_frame_num = frame_num # needed to determine object speeds
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
                    unique, _ = np.unique(inst, return_inverse = True)
                    instance_colour_mapping = boxx.getDefaultColorList(len(unique), includeBackGround=True)
                # map unique instance ids to a visual image
                height, width = inst.shape
                inst_vis = np.zeros(shape = (height, width, 3))            
                for i in range(height):
                    for j in range(width):
                        # hardcoded part for AVoE D as indexes are not numbered nicely
                        if inst[i,j] == 6:
                            index = 1
                        elif inst[i,j] == 7:
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
                if entity == 'Ground':
                    # no need for pose/orientation data for ground
                    continue
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
            
        ### HARDCODED FOR SUB-DATASET D: Collision ###
        
        # initialise dictionary for physical_data json file
        physical_data = {'features' : {}, 'prior_rules' : {}, 'posterior_rules' : {}}
        
        # Feature #1 & #2: Object sizes (in terms of height)
        physical_data['features']['left_size'] = left_height
        physical_data['features']['right_size'] = right_height
        # Feature #3 & #4: 'Prior Object speeds (arbitrary scale)
        physical_data['features']['left_prior_velocity'] = left_speed
        physical_data['features']['right_prior_velocity'] = -1 * right_speed
        # determine relative prior speeds based on frame data (direction already accounted for) 
        prior_left_relative_speed = return_vel(scene, 'Left_Object', speedup_frame, speedup_frame + 1,\
        actual_frame_num)
        prior_right_relative_speed = return_vel(scene, 'Right_Object', speedup_frame, speedup_frame + 1,\
        actual_frame_num)
        # determine relative posterior speeds
        post_left_relative_speed = return_vel(scene, 'Left_Object', actual_frame_num - 1, \
        actual_frame_num, actual_frame_num)
        post_right_relative_speed = return_vel(scene, 'Right_Object', actual_frame_num -1, \
        actual_frame_num, actual_frame_num)
        
        # Feature #5 & #6: Posterior Object speeds (arbitrary scale)
        physical_data['features']['left_posterior_velocity'] = left_speed * \
        post_left_relative_speed / prior_left_relative_speed
        physical_data['features']['right_posterior_velocity'] = -1 * right_speed * \
        post_right_relative_speed / prior_right_relative_speed
        
        # Feature #7 & #8: Left and Right object shapes
        physical_data['features']['left_shape'] = left_shape
        physical_data['features']['right_shape'] = right_shape
        
        # Prior Rule #1: is the rightmost object larger?
        if right_height > left_height:
            physical_data['prior_rules']['is_right_object_larger?'] = True
        else:
            physical_data['prior_rules']['is_right_object_larger?'] = False
    
        # Prior Rule #2: are both objects the same size
        if right_height == left_height:
            physical_data['prior_rules']['are_both_objects_same_size?'] = True
        else:
            physical_data['prior_rules']['are_both_objects_same_size?'] = False
        
        # Prior Rule #3: is the rightmost object faster?
        if right_speed > left_speed:
            physical_data['prior_rules']['is_right_object_faster?'] = True
        else:
            physical_data['prior_rules']['is_right_object_faster?'] = False
    
        # Prior Rule #4: are both objects the same size
        if right_speed == left_speed:
            physical_data['prior_rules']['are_both_objects_same_speed?'] = True
        else:
            physical_data['prior_rules']['are_both_objects_same_speed?'] = False
            
        # NOTE: moving to stationary is considered as 'direction changed'
        # Posterior Rule #1: did right object change direction??
        if return_direction(scene, 'Right_Object', speedup_frame, speedup_frame + 1, actual_frame_num) != \
        return_direction(scene, 'Right_Object', actual_frame_num - 1, actual_frame_num, actual_frame_num):
            physical_data['posterior_rules']['did_right_object_change_direction?'] = True
        else:
            physical_data['posterior_rules']['did_right_object_change_direction?'] = False
    
        # NOTE: moving to stationary is considered as 'direction changed'
        # Posterior Rule #2: did left object change direction??
        if return_direction(scene, 'Left_Object', speedup_frame, speedup_frame + 1, actual_frame_num) != \
        return_direction(scene, 'Left_Object', actual_frame_num - 1, actual_frame_num, actual_frame_num):
            physical_data['posterior_rules']['did_left_object_change_direction?'] = True
        else:
            physical_data['posterior_rules']['did_left_object_change_direction?'] = False
        
        # Posterior Rule #3: did right object increase speed magnitude?
        if abs(post_right_relative_speed) > abs(prior_right_relative_speed):
            physical_data['posterior_rules']['did_right_object_increase_speed_magnitude?'] = True
        else:
            physical_data['posterior_rules']['did_right_object_increase_speed_magnitude?'] = False
    
        # Posterior Rule #4: did left object increase speed magnitude?
        if abs(post_left_relative_speed) > abs(prior_left_relative_speed):
            physical_data['posterior_rules']['did_left_object_increase_speed_magnitude?'] = True
        else:
            physical_data['posterior_rules']['did_left_object_increase_speed_magnitude?'] = False
            
        if not violation:
            left_direction_change_list.append(physical_data['posterior_rules']['did_left_object_change_direction?'])
            right_direction_change_list.append(physical_data['posterior_rules']['did_right_object_change_direction?'])
            left_magnitude_higher_list.append(physical_data['posterior_rules']['did_left_object_increase_speed_magnitude?'])
            right_magnitude_higher_list.append(physical_data['posterior_rules']['did_right_object_increase_speed_magnitude?'])
    
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
    outcome_df = pd.DataFrame(left_direction_change_list, columns = ['left_direction_change'])
    outcome_df['right_direction_change'] = right_direction_change_list
    outcome_df['left_magnitude_higher'] = left_magnitude_higher_list
    outcome_df['right_magnitude_higher'] = right_magnitude_higher_list
    outcome_df.to_csv(root_filepath + 'outcome.csv', index = False)
    # visualise dataset balance
    print('left_direction_change')
    print(outcome_df['left_direction_change'].value_counts()) # in terms of trials
    print('right_direction_change')
    print(outcome_df['right_direction_change'].value_counts()) # in terms of trials
    print('left_magnitude_higher')
    print(outcome_df['left_magnitude_higher'].value_counts()) # in terms of trials
    print('right_magnitude_higher')
    print(outcome_df['right_magnitude_higher'].value_counts()) # in terms of trials
    
# visualise some stats
print('left_size')
print(variation_df['left_height'].value_counts()) # in terms of scenes
print('right_size')
print(variation_df['right_height'].value_counts()) # in terms of scenes
print('left_speed')
print(variation_df['left_speed'].value_counts()) # in terms of scenes
print('right_speed')
print(variation_df['right_speed'].value_counts()) # in terms of scenes
print('left_shape')
print(variation_df['left_shape'].value_counts()) # in terms of scenes
print('right_shape')
print(variation_df['right_shape'].value_counts()) # in terms of scenes