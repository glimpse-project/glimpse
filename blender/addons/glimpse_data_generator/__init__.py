# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8-80 compliant>

bl_info = {
    "name": "Glimpse Training Data Generator",
    "description": "Tool to help generate skeleton tracking training data",
    "author": "Robert Bragg",
    "version": (0, 1, 1),
    "blender": (2, 78, 0),
    #"location": "",
    "warning": "",
    "support": 'COMMUNITY',
    "wiki_url": "",
    "category": "Mesh"}

# Copyright (C) 2017: Robert Bragg <robert@impossible.com>

import math
import mathutils
import os
import json
import ntpath
import numpy

import bpy
from bpy.props import (
        CollectionProperty,
        StringProperty,
        BoolProperty,
        IntProperty,
        EnumProperty,
        FloatProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        orientation_helper_factory,
        axis_conversion,
        )
import bmesh


def add_clothing(op, context, clothing_name):

    if clothing_name + "_reference" not in bpy.data.objects:
        return

    mhclo_relpath = os.path.join(clothing_name, clothing_name + ".mhclo")
    ref_clothing_obj = bpy.data.objects[clothing_name + "_reference"]

    # XXX: we don't reference context.active_object because for some reason
    # (bug or important misunderstanding about blender's state management)
    # then changes to context.scene.objects.active aren't immediately
    # reflected in context.active_object (and not a question of calling
    # scene.update() either)
    helper_mesh = context.scene.objects.active
    human_mesh_name = helper_mesh.name[:-len("BodyHelperMeshObject")]

    mhclo_file = bpy.path.abspath(context.scene.GlimpseClothesRoot + mhclo_relpath)
    bpy.ops.mhclo.test_clothes(filepath = mhclo_file)

    clothing = context.selected_objects[0]
    context.scene.objects.active = clothing

    clothing.data.materials.append(bpy.data.materials.get("JointLabelsMaterial"))

    bpy.ops.object.modifier_add(type='DATA_TRANSFER')
    clothing.modifiers['DataTransfer'].object = ref_clothing_obj

    clothing.modifiers['DataTransfer'].use_vert_data = True
    clothing.modifiers['DataTransfer'].data_types_verts = {'VGROUP_WEIGHTS'}
    clothing.modifiers['DataTransfer'].vert_mapping = 'TOPOLOGY'

    clothing.modifiers['DataTransfer'].use_loop_data = True
    clothing.modifiers['DataTransfer'].data_types_loops = {'VCOL'}
    clothing.modifiers['DataTransfer'].loop_mapping = 'TOPOLOGY'

    context.scene.objects.active = clothing
    bpy.ops.object.datalayout_transfer(modifier="DataTransfer")
    bpy.ops.object.modifier_apply(modifier='DataTransfer')

    bpy.ops.object.select_all(action='DESELECT')

    clothing.select = True
    bpy.data.objects[human_mesh_name + 'PoseObject'].select = True
    context.scene.objects.active = bpy.data.objects[human_mesh_name + 'PoseObject']

    bpy.ops.object.parent_set(type='ARMATURE_NAME')


class AddClothingOperator(bpy.types.Operator):
    """Adds clothing to the active body"""

    bl_idname = "glimpse.add_clothing"
    bl_label = "Add Clothing"

    clothing_name = StringProperty(
            name="Clothing Name",
            description="Name of clothing register_module"
            )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMeshObject" in context.active_object.name

    def execute(self, context):
        if self.clothing_name + "_reference" not in bpy.data.objects:
            return {'FINISHED'}
        add_clothing(self, context, self.clothing_name)

        return {'FINISHED'}


class AddHoodieOperator(bpy.types.Operator):
    """Adds a hoodie to the active body"""

    bl_idname = "glimpse.add_hoodie"
    bl_label = "Add Hoodie"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMeshObject" in context.active_object.name

    def execute(self, context):
        add_clothing(self, context, "hooded_cardigan")
        return {'FINISHED'}

class AddMTrousers01Operator(bpy.types.Operator):
    """Adds m_trousers_01 clothing to the active body"""

    bl_idname = "glimpse.add_m_trousers_01"
    bl_label = "Add M Trousers 01"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMeshObject" in context.active_object.name

    def execute(self, context):
        add_clothing(self, context, "m_trousers_01")
        return {'FINISHED'}


class AddPatrolCapOperator(bpy.types.Operator):
    """Adds a patrol cap to the active body"""

    bl_idname = "glimpse.add_patrol_cap"
    bl_label = "Add Patrol Cap"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMeshObject" in context.active_object.name

    def execute(self, context):
        add_clothing(self, context, "patrol_cap")
        return {'FINISHED'}


class AddGlassesOperator(bpy.types.Operator):
    """Adds a glasses to the active body"""

    bl_idname = "glimpse.add_glasses"
    bl_label = "Add Glasses"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMeshObject" in context.active_object.name

    def execute(self, context):
        add_clothing(self, context, "glasses")
        return {'FINISHED'}


all_bodies = [ 'Man0', 'Woman0', 'Man1', 'Woman1' ]

hat_choices = [ 'none', 'knitted_hat_01', 'newsboy_cap', 'patrol_cap' ]
hat_probabilities = [ 0.4, 0.2, 0.2, 0.2 ]

glasses_choices = [ 'none', 'glasses' ]
glasses_probabilities = [ 0.7, 0.3 ]

top_choices = [ 'none', 'hooded_cardigan' ]
top_probabilities = [ 0.8, 0.2 ]

trouser_choices = [ 'none', 'm_trousers_01' ]
trouser_probabilities = [ 0.5, 0.5 ]

shoe_choices = []
shoe_probabilities = []


# The data generator operator expects that each body has previously had all
# possible clothing items added to the body. This avoids the cost of loading
# the clothing at runtime and also clothing that's not applicable to a
# particular body can be excluded and the generator will handle that gracefully
class AddClothingLibraryOperator(bpy.types.Operator):
    """Adds the full library of known clothing items to active body"""

    bl_idname = "glimpse.add_clothes_library"
    bl_label = "Add Clothes Library To Body"

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):

        bpy.ops.object.select_all(action='DESELECT')

        all_clothes = []
        all_clothes.extend(hat_choices)
        all_clothes.extend(glasses_choices)
        all_clothes.extend(top_choices)
        all_clothes.extend(trouser_choices)
        all_clothes.extend(shoe_choices)

        # The clothing sub-lists have 'none' entries we want to ignore here
        all_clothes = [ clothing for clothing in all_clothes if clothing != 'none' ]


        for body in all_bodies:
            pose_obj = bpy.data.objects[body + 'PoseObject']
            if pose_obj == None:
                self.report({'ERROR'}, "didn't find pose mesh for " + body)
                continue
            helper_obj = bpy.data.objects[body + 'BodyHelperMeshObject']
            if helper_obj == None:
                self.report({'ERROR'}, "didn't find helper mesh for " + body)
                continue

            for child in pose_obj.children:
                self.report({'INFO'}, body + " child:" + child.name)

                if body + "Clothes:" in child.name:
                    self.report({'INFO'}, "removing " + child.name + " from " + body)
                    context.scene.objects.active = bpy.data.objects[child.name]
                    bpy.data.objects[child.name].select = True
                    bpy.ops.object.delete()

            for clothing in all_clothes:
                bpy.ops.object.select_all(action='DESELECT')
                context.scene.objects.active = helper_obj
                add_clothing(self, context, clothing)

                for child in pose_obj.children:
                    if child.name == clothing:
                        child.name = body + "Clothes:" + clothing
                        child.layers = helper_obj.layers
                        child.hide = True
                        break

        return {'FINISHED'}


class RigGeneratorOperator(bpy.types.Operator):
    """Generates Glimpse training data"""

    bl_idname = "glimpse.generate_data"
    bl_label = "Generate Glimpse training data"

    my_debug_bool = BoolProperty(
            name="Debug Boolean",
            description="Debug Knob",
            default=True,
            )

    my_debug_float = FloatProperty(
            name="Debug float",
            min=0.01, max=1000.0,
            default=1.0,
            )

    @classmethod
    def poll(cls, context):
        return len(bvh_index) > 0

    def execute(self, context):

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.objects.active = bpy.data.objects['BasePoseObject']

        render_layers = []
        for i in range(0, 20):
            render_layers.append(False)
        render_layers[0] = True

        # Do all our rendering from layer zero, pulling in other objects
        # as needed
        context.scene.layers = render_layers

        for body in all_bodies:
            pose_obj = bpy.data.objects[body + 'PoseObject']
            for child in pose_obj.children:
                child.layers = pose_obj.layers

        render_objects = []

        self.report({'INFO'}, "Rendering MoCap indices from " + str(bpy.context.scene.GlimpseBvhGenFrom) + " to " + str(bpy.context.scene.GlimpseBvhGenTo))
        for idx in range(bpy.context.scene.GlimpseBvhGenFrom, bpy.context.scene.GlimpseBvhGenTo):
            bvh = bvh_index[idx]

            if bvh['blacklist']:
                self.report({'INFO'}, "skipping blacklisted")
                continue

            # Note we always load the mocap animations with the base the base
            # mesh, and then associate the animation_data.action with all the
            # other armatures we have
            bpy.context.scene.objects.active = bpy.data.objects['BasePoseObject']
            self.report({'INFO'}, "loading")
            load_bvh_file(bvh, interactive_mode=False)

            for body in all_bodies:
                pose_obj = bpy.data.objects[body + 'PoseObject']
                if pose_obj.animation_data == None:
                    pose_obj.animation_data_create()
                pose_obj.animation_data.action = bpy.data.objects['BasePoseObject'].animation_data.action
                bpy.data.armatures[body + 'Pose'].pose_position = 'POSE'

            bvh_name = os.path.basename(bvh['file'])
            self.report({'INFO'}, "rendering " + bvh_name)

            randomization_step = 5

            # extend the range end beyond the length in case it's not a multiple
            # of the randomization_step...
            range_end = bvh['end'] + randomization_step - 1

            # Hit some errors with the range bounds not being integer and I
            # guess that comes from the json library loading our mocap index
            # may make some numeric feilds float so bvh['start'] or bvh['end']
            # might be float
            for start_frame in range(int(bvh['start']), int(range_end), randomization_step):

                self.report({'INFO'}, "> randomizing " + bvh_name + " render")

                for obj in render_objects:
                    obj.layers[0] = False
                    if "Clothes:" in obj.name:
                        obj.hide = True
                        obj.layers = obj.parent.layers
                render_objects = []
                #for body in all_bodies:
                #    bpy.data.objects[body + "BodyMeshObject"].layers[0] = False

                # randomize the model
                body = numpy.random.choice(all_bodies)
                hat = numpy.random.choice(hat_choices, p=hat_probabilities)
                trousers = numpy.random.choice(trouser_choices, p=trouser_probabilities)
                top = numpy.random.choice(top_choices, p=top_probabilities)
                glasses = numpy.random.choice(glasses_choices, p=glasses_probabilities)

                body_obj = bpy.data.objects[body + "BodyMeshObject"]
                body_obj.layers[0] = True
                render_objects.append(body_obj)

                body_pose = bpy.data.objects[body + "PoseObject"]
                bpy.data.objects['Camera'].constraints['Track To'].target = body_pose

                if hat != 'none':
                    hat_obj = bpy.data.objects[body + "Clothes:" + hat]
                    if hat_obj:
                        hat_obj.hide = False
                        hat_obj.layers[0] = True
                        render_objects.append(hat_obj)

                if trousers != 'none':
                    trouser_obj = bpy.data.objects[body + "Clothes:" + trousers]
                    if trouser_obj:
                        trouser_obj.hide = False
                        trouser_obj.layers[0] = True
                        render_objects.append(trouser_obj)

                    #add_clothing(self, context, trousers)
                    #trouser_obj.layers[0] = True
                    #render_objects.append(trouser_obj)

                if top != 'none':
                    top_obj = bpy.data.objects[body + "Clothes:" + top]
                    if top_obj:
                        top_obj.hide = False
                        top_obj.layers[0] = True
                        render_objects.append(top_obj)

                    #add_clothing(self, context, top)
                    #top_obj.layers[0] = True
                    #render_objects.append(top_obj)

                if glasses != 'none':
                    glasses_obj = bpy.data.objects[body + "Clothes:" + glasses]
                    if glasses_obj:
                        glasses_obj.hide = False
                        glasses_obj.layers[0] = True
                        render_objects.append(glasses_obj)

                    #add_clothing(self, context, glasses)
                    #glasses_obj.layers[0] = True
                    #render_objects.append(glasses_obj)

                # NB. the range extends beyond the length of animation in case
                # it's not a multiple of the randomization step, so clip end here:
                end_frame = min(start_frame + randomization_step, bvh['end'])

                dirname =  str(start_frame) + "_" + str(end_frame-1) + "_" + body + "_hat_" + hat + "_trousers_" + trousers + "_top_" + top + "_glasses_" + glasses
                context.scene.node_tree.nodes['LabelOutput'].base_path = bpy.path.abspath(bpy.context.scene.GlimpseDataRoot + os.path.join("color-test", bvh_name[:-4], dirname))
                context.scene.node_tree.nodes['DepthOutput'].base_path = bpy.path.abspath(bpy.context.scene.GlimpseDataRoot + os.path.join("depth-test", bvh_name[:-4], dirname))
                context.scene.update()

                context.scene.layers = render_layers

                for frame in range(int(start_frame), int(end_frame)):
                    bpy.context.scene.frame_current = frame

                    pose_cam_vec = body_pose.pose.bones['pelvis'].head - bpy.data.objects['Camera'].location
                    if pose_cam_vec.length < 1.5:
                        self.report({'INFO'}, "> skipping " + bvh_name + " frame " + str(frame) + ": pose too close to camera")
                        continue

                    self.report({'INFO'}, "> render " + bvh_name + " frame " + str(frame) + "to " + bpy.context.scene.node_tree.nodes['LabelOutput'].base_path)
                    bpy.ops.render.render(write_still=True)

        return {'FINISHED'}


    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "my_debug_bool")
        row.prop(self, "my_debug_float")


class GeneratePanel(bpy.types.Panel):
    bl_category = "Glimpse Generate"
    bl_label = "GlimpseGenerate v %d.%d.%d: Main" % bl_info["version"]
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"

    def draw(self, context):
        layout = self.layout
        ob = context.object
        scn = context.scene
        
        layout.separator()
        layout.prop(scn, "GlimpseDataRoot", text="Output")
        layout.separator()
        layout.prop(scn, "GlimpseClothesRoot", text="Clothes")
        layout.separator()
        #layout.prop(scn, "GlimpseDebug")
        #layout.separator()
        row = layout.row()
        row.prop(scn, "GlimpseBvhGenFrom", text="From")
        row.prop(scn, "GlimpseBvhGenTo", text="To")
        layout.separator()
        layout.operator("glimpse.generate_data")


# Is there a better place to track this state?
bvh_index = []
bvh_file_index = {}
bvh_index_pos = 0


def get_bvh_index_pos(self):
    return bvh_index_pos


def set_bvh_index_pos(self, value):
    global bvh_index_pos

    if value >= 0 and value < len(bvh_index) and value != bvh_index_pos:
        update_current_bvh_state(None)
        bvh_index_pos = value
        load_current_bvh_file(None)


# NB: sometimes called with no op
def update_current_bvh_state(ignore_op):
    if bvh_index_pos >= len(bvh_index):
        if ignore_op != None:
            op.report({'ERROR'}, "Invalid Mo-cap index")
        return

    bvh_state = bvh_index[bvh_index_pos]

    bvh_state['start'] = bpy.context.scene.frame_start
    bvh_state['end'] = bpy.context.scene.frame_end

    camera = bpy.data.objects['Camera']
    cam_pos = camera.location.xyz
    cam_rot = camera.rotation_quaternion

    vertical_fov = bpy.data.cameras['Camera'].angle

    bvh_state['camera'] = { 'location': [cam_pos[0], cam_pos[1], cam_pos[2]],
                            'rotation': [cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3]],
                            'vertical_fov': vertical_fov }


def load_bvh_file(bvh_state, interactive_mode=False):

    pose_obj = bpy.context.scene.objects.active

    bpy.context.scene.McpStartFrame = 1
    bpy.context.scene.McpEndFrame = 1000
    bpy.ops.mcp.load_and_retarget(filepath=bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot + bvh_state['file']))

    # It's a nasty gotcha but the MalkWalk addon will set the fake user
    # reference on the imported action which makes the .blend file eventually
    # balloon to ridiculous proportions
    pose_obj.animation_data.action.use_fake_user = False

    bpy.context.scene.frame_start = bvh_state['start']
    if 'end' in bvh_state:
        bpy.context.scene.frame_end = bvh_state['end']
    else:
        if bpy.context.object.animation_data:
            frame_end = bpy.context.object.animation_data.action.frame_range[1]
        else:
            frame_end = 1000
        bpy.context.scene.frame_end = frame_end
        bvh_state['end'] = frame_end

    # While interactively reviewing mocap files then it seems more likely than
    # not that the camera state that was good for the previous file could be
    # good default for this one if we haven't got camera state in the index.
    #
    # We want to make a more deterministic default while running
    # non-interactively
    #
    if not interactive_mode:
        cam_pos = bpy.data.objects['Camera'].location.xyz
        cam_rot = bpy.data.objects['Camera'].rotation_quaternion
    else:
        cam_pos = [0, -2, 1.4]
        cam_rot = [1, 0, 0, 0]
    cam_fov = 83 # (vertical)

    if 'camera' in bvh_state:
        if 'location' in bvh_state['camera']:
            cam_pos = bvh_state['camera']['location']

        if 'quaternion' in bvh_state['camera']:
            cam_rot = bvh_state['camera']['quaternion']

        # Originally it was fixed (part of the .blend file) that the camera
        # had a vertical fov of 49.13 degrees, but since that doesn't really
        # match the typical fov of a phone camera well we now track the fov
        # of the camera as part of the index (at time of writing we switched
        # to an fov of 83 degrees)
        #
        if 'vertical_fov' in bvh_state:
            cam_fov = bvh_state['camera']['vertical_fov']
        else:
            # instead of keeping the old FOV we will instead use the new
            # default but move the camera closer
            cam_pos[1] = cam_pos[1] / 2
    else:
        bvh_state['camera'] = {}

    bpy.data.objects['Camera'].location.xyz = cam_pos
    bpy.data.objects['Camera'].rotation_quaternion = cam_rot
    bpy.data.cameras['Camera'].angle = math.radians(cam_fov)

    bvh_state['camera']['location'] = cam_pos
    bvh_state['camera']['quaternion'] = cam_rot
    bvh_state['camera']['vertical_fov'] = cam_fov

    #if 'blacklist' in bvh_state:
    #    bpy.context.scene.GlimpseMoCapBlacklist = bvh_state['blacklist']
    #else:
    #    bpy.context.scene.GlimpseMoCapBlacklist = False


# NB: sometimes called with no op
def load_current_bvh_file(ignore_op):
    if bvh_index_pos >= len(bvh_index):
        if ignore_op != None:
            op.report({'ERROR'}, "Invalid Mo-cap index")
        return

    load_bvh_file(bvh_index[bvh_index_pos], interactive_mode=True)


def load_mocap_index():

    bvh_index = []

    try:
        with open(bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot + "index.json")) as fp:
            bvh_index = json.load(fp)

        # early version might have indexed non-bvh files...
        keep = [bvh for bvh in bvh_index if bvh['file'][-4:] == '.bvh']
        bvh_index = keep

        bpy.types.Scene.GlimpseBvhIndexPos[1]['max'] = max(0, len(bvh_index) - 1)

        for bvh in bvh_index:
            bvh_file_index[bvh['file']] = bvh

            # normalize so we don't have to consider that it's left unspecified
            if 'blacklist' not in bvh:
                bvh['blacklist'] = False

    except IOError as e:
        self.report({'INFO'}, str(e))

    return bvh_index


class VIEW3D_MoCap_OpenBvhIndexButton(bpy.types.Operator):
    bl_idname = "glimpse.open_bvh_index"
    bl_label = "Load Index"

    def execute(self, context):
        global bvh_index
        global bvh_file_index
        global bvh_index_pos

        bvh_index = []
        bvh_file_index = {}
        bvh_index_pos = 0

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        bpy.ops.object.select_all(action='DESELECT')

        pose_obj = bpy.data.objects['Man0PoseObject']
        pose_obj.select=True
        context.scene.layers = pose_obj.layers
        context.scene.objects.active = pose_obj

        bvh_index = load_mocap_index()

        if len(bvh_index) > 0:
            if bvh_index[0]['blacklist']:
                skip_to_next_bvh(self)
            load_current_bvh_file(self)

        return {"FINISHED"}


class VIEW3D_MoCap_BvhScanButton(bpy.types.Operator):
    bl_idname = "glimpse.scan_bvh_files"
    bl_label = "Scan for un-indexed .bvh files"

    def execute(self, context):
        global bvh_index
        global bvh_index_pos
        
        for root, dirs, files in os.walk(bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot)):
            relroot = os.path.relpath(root, bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot))
            for file in files:
                if file[-4:] != ".bvh":
                    continue

                filename = os.path.join(relroot, file)

                # no matter what OS we're using we want consistent filename
                # indexing conventions
                filename = ntpath.normpath(filename)
                filename = ntpath.normcase(filename)
                if filename not in bvh_file_index:
                    new_bvh_state = { 'file': filename, 'start': 1 }
                    bvh_index.append(new_bvh_state)
                    bvh_file_index[filename] = new_bvh_state

                    self.report({'INFO'}, "ADD: " + filename)

        return {"FINISHED"}


class VIEW3D_MoCap_OpenBvhPrev(bpy.types.Operator):
    bl_idname = "glimpse.open_bvh_prev"
    bl_label = "Prev"

    @classmethod
    def poll(cls, context):
        return bvh_index_pos > 0

    def execute(self, context):

        global bvh_index
        global bvh_index_pos

        update_current_bvh_state(self)

        while bvh_index_pos > 0:
            bvh_index_pos = bvh_index_pos - 1

            if bvh_index[bvh_index_pos]['blacklist'] == False:
                break

        load_current_bvh_file(self)

        return {"FINISHED"}


def skip_to_next_bvh(op):
    global bvh_index_pos

    while bvh_index_pos < len(bvh_index) - 1:
        bvh_index_pos = bvh_index_pos + 1
        if bvh_index[bvh_index_pos]['blacklist'] == False:
            break


class VIEW3D_MainPanel_OpenBvhNext(bpy.types.Operator):
    bl_idname = "glimpse.open_bvh_next"
    bl_label = "Next"

    @classmethod
    def poll(cls, context):
        return bvh_index_pos < len(bvh_index) - 1

    def execute(self, context):
        global bvh_index
        global bvh_index_pos

        update_current_bvh_state(self)

        skip_to_next_bvh(self)

        load_current_bvh_file(self)

        return {"FINISHED"}


def get_mocap_blacklist(self):
    if bvh_index_pos < len(bvh_index):
        return bvh_index[bvh_index_pos]['blacklist']
    else:
        return False


def set_mocap_blacklist(self, value):
    if bvh_index_pos < len(bvh_index):
        bvh_index[bvh_index_pos]['blacklist'] = value


class VIEW3D_MoCap_BlacklistButton(bpy.types.Operator):
    bl_idname = "glimpse.scan_bvh_files"
    bl_label = "Index BVH Files"

    def execute(self, context):
        bvh_index[bvh_index_pos]['blacklist': True]


class VIEW3D_MoCap_SaveBvhIndexButton(bpy.types.Operator):
    bl_idname = "glimpse.save_bvh_index"
    bl_label = "Save Index"

    def execute(self, context):

        if len(bvh_index):
            update_current_bvh_state(self)

            try: 
                with open(bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot + "index.json"), "w", encoding="utf-8") as fp:
                    json.dump(bvh_index, fp)
            except IOError as e:
                self.report({'ERROR'}, str(e))
        else:
            self.report({'ERROR'}, "No Mo-cap data to save")

        return {"FINISHED"}


class MoCapPanel(bpy.types.Panel):
    bl_label = "Motion Capture"
    bl_category = "Glimpse Generate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"

    #@classmethod
    #def poll(self, context):
    #    return (context.object and context.object.type == 'ARMATURE')

    def draw(self, context):
        layout = self.layout
        ob = context.object
        scn = context.scene

        layout.prop(scn, "GlimpseBvhRoot", text="")
        layout.separator()
        layout.operator("glimpse.scan_bvh_files")
        layout.separator()
        layout.operator("glimpse.open_bvh_index")
        row = layout.row()
        row.operator("glimpse.open_bvh_prev")
        row.operator("glimpse.open_bvh_next")
        row.prop(scn, "GlimpseBvhIndexPos", text="")
        row.label("/ " + str(len(bvh_index)))
        layout.separator()
        layout.operator("glimpse.save_bvh_index")


class MoCapFilePanel(bpy.types.Panel):
    bl_label = "Motion Capture File"
    bl_category = "Glimpse Generate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"

    @classmethod
    def poll(self, context):
        return (context.object and context.object.type == 'ARMATURE') and len(bvh_index) > 0

    def draw(self, context):
        layout = self.layout
        ob = context.object
        scn = context.scene

        if bvh_index_pos < len(bvh_index):
            layout.label("File: " + bvh_index[bvh_index_pos]['file'])
        else:
            layout.label("File: None")

        layout.separator()
        layout.prop(scn, "GlimpseMoCapBlacklist")



def register():
    bpy.types.Scene.GlimpseDataRoot = StringProperty(
            name="Training Directory",
            description="Root directory for training data",
            subtype='DIR_PATH',
            )

    bpy.types.Scene.GlimpseClothesRoot = StringProperty(
            name="Clothes Directory",
            description="Root directory for makehuman clothes",
            subtype='DIR_PATH',
            )

    bpy.types.Scene.GlimpseBvhRoot = StringProperty(
            name="BVH Directory",
            description="Root directory for .bvh motion capture files",
            subtype='DIR_PATH',
            )

    bpy.types.Scene.GlimpseDebug = BoolProperty(
            name="Debug",
            description="Enable Debugging",
            default=True,
            )

    bpy.types.Scene.GlimpseMoCapBlacklist = BoolProperty(
            name="Blacklist",
            description="Blacklist this Mo-cap file",
            default=False,
            get=get_mocap_blacklist,
            set=set_mocap_blacklist
            )

    bpy.types.Scene.GlimpseBvhIndexPos = IntProperty(
            name="Index",
            description="Current BVH state index",
            default=0,
            min=0,
            max=5000,
            get=get_bvh_index_pos,
            set=set_bvh_index_pos
            )

    bpy.types.Scene.GlimpseBvhGenFrom = IntProperty(
            name="Index",
            description="From",
            default=0,
            min=0)

    bpy.types.Scene.GlimpseBvhGenTo = IntProperty(
            name="Index",
            description="To",
            default=0,
            min=0)

    bpy.utils.register_module(__name__)



def unregister():
    bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
    register()
