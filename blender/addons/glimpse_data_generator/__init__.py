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


def add_clothing(op, context, mhclo_relpath, ref_clothing_obj):
    human_mesh_name = context.active_object.name[:-len("BodyHelperMesh")]
    helper_mesh = context.active_object

    mhclo_file = bpy.path.abspath(bpy.context.scene.GlimpseClothesRoot + mhclo_relpath)
    bpy.ops.mhclo.test_clothes(filepath = mhclo_file)

    clothing = bpy.context.selected_objects[0]
    bpy.context.scene.objects.active = clothing

    clothing.data.materials.append(bpy.data.materials.get("JointLabelsMaterial"))

    bpy.ops.object.modifier_add(type='DATA_TRANSFER')
    clothing.modifiers['DataTransfer'].object = ref_clothing_obj

    clothing.modifiers['DataTransfer'].use_vert_data = True
    clothing.modifiers['DataTransfer'].data_types_verts = {'VGROUP_WEIGHTS'}
    clothing.modifiers['DataTransfer'].vert_mapping = 'TOPOLOGY'

    clothing.modifiers['DataTransfer'].use_loop_data = True
    clothing.modifiers['DataTransfer'].data_types_loops = {'VCOL'}
    clothing.modifiers['DataTransfer'].loop_mapping = 'TOPOLOGY'

    bpy.context.scene.objects.active = clothing
    bpy.ops.object.datalayout_transfer(modifier="DataTransfer")
    bpy.ops.object.modifier_apply(modifier='DataTransfer')

    bpy.ops.object.select_all(action='DESELECT')

    clothing.select = True
    bpy.data.objects[human_mesh_name + 'PoseObject'].select = True
    bpy.context.scene.objects.active = bpy.data.objects[human_mesh_name + 'PoseObject']

    bpy.ops.object.parent_set(type='ARMATURE_NAME')


class AddHoodieOperator(bpy.types.Operator):
    """Adds an item of clothing to the active object"""

    bl_idname = "glimpse.add_hoodie"
    bl_label = "Add Hoodie"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "BodyHelperMesh" in context.active_object.name

    def execute(self, context):
        mhclo_relpath = os.path.join("hoodie", "hooded_cardigan.mhclo")
        ref_obj = bpy.data.objects['hooded_cardigan_reference']
        add_clothing(self, context, mhclo_relpath, ref_obj)

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
        return context.active_object is not None

    def execute(self, context):
        self.report({'INFO'}, "TODO: Generate Data")

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
        layout.prop(scn, "GlimpseClothesRoot", text="Clothes")
        layout.separator()
        layout.prop(scn, "GlimpseDebug")
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
        updateCurrentBvhState(None)
        bvh_index_pos = value
        loadCurrentBvhFile(None)


# NB: sometimes called with no op
def updateCurrentBvhState(ignore_op):
    if bvh_index_pos >= len(bvh_index):
        if ignore_op != None:
            op.report({'ERROR'}, "Invalid Mo-cap index")
        return

    bvh_state = bvh_index[bvh_index_pos]

    bvh_state['start'] = bpy.context.scene.frame_start
    bvh_state['end'] = bpy.context.scene.frame_end

    cam_pos = bpy.data.objects['Camera'].location.xyz
    cam_rot = bpy.data.objects['Camera'].rotation_quaternion
    bvh_state['camera'] = { 'location': [cam_pos[0], cam_pos[1], cam_pos[2]],
                            'rotation': [cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3]] }


# NB: sometimes called with no op
def loadCurrentBvhFile(ignore_op):
    if bvh_index_pos >= len(bvh_index):
        if ignore_op != None:
            op.report({'ERROR'}, "Invalid Mo-cap index")
        return

    bvh_state = bvh_index[bvh_index_pos]

    bpy.context.scene.McpStartFrame = 1
    bpy.context.scene.McpEndFrame = 1000
    bpy.ops.mcp.load_and_retarget(filepath=bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot + bvh_state['file']))

    bpy.context.scene.frame_start = bvh_state['start']
    if 'end' in bvh_state:
        bpy.context.scene.frame_end = bvh_state['end']
    else:
        bpy.context.scene.frame_end = 1000

    if 'camera' in bvh_state:
        if 'location' in bvh_state['camera']:
            cam_pos = bvh_state['camera']['location']
            bpy.data.objects['Camera'].location.xyz = cam_pos
        if 'quaternion' in bvh_state['camera']:
            cam_rot = bvh_state['camera']['quaternion']
            bpy.data.objects['Camera'].rotation_quaternion = cam_rot

    #if 'blacklist' in bvh_state:
    #    bpy.context.scene.GlimpseMoCapBlacklist = bvh_state['blacklist']
    #else:
    #    bpy.context.scene.GlimpseMoCapBlacklist = False


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

        try:
            with open(bpy.path.abspath(bpy.context.scene.GlimpseBvhRoot + "index.json")) as fp:
                bvh_index = json.load(fp)

            # early version might have indexed non-bvh files...
            keep = [bvh for bvh in bvh_index if bvh['file'][-4:] == '.bvh']
            bvh_index = keep

            bpy.types.Scene.GlimpseBvhIndexPos[1]['max'] = max(0, len(bvh_index) - 1)

            for bvh in bvh_index:
                bvh_file_index[bvh['file']] = bvh

            loadCurrentBvhFile(self)
        except IOError as e:
            self.report({'INFO'}, str(e))

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

        updateCurrentBvhState(self)

        while bvh_index_pos > 0:
            bvh_index_pos = bvh_index_pos - 1

            if 'blacklist' not in bvh_index[bvh_index_pos] or bvh_index[bvh_index_pos]['blacklist'] == False:
                break

        loadCurrentBvhFile(self)

        return {"FINISHED"}


class VIEW3D_MainPanel_OpenBvhNext(bpy.types.Operator):
    bl_idname = "glimpse.open_bvh_next"
    bl_label = "Next"

    @classmethod
    def poll(cls, context):
        return bvh_index_pos < len(bvh_index) - 1

    def execute(self, context):
        global bvh_index
        global bvh_index_pos

        updateCurrentBvhState(self)

        while bvh_index_pos < len(bvh_index) - 1:
            bvh_index_pos = bvh_index_pos + 1
            if 'blacklist' not in bvh_index[bvh_index_pos] or bvh_index[bvh_index_pos]['blacklist'] == False:
                break

        loadCurrentBvhFile(self)

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
            updateCurrentBvhState(self)

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

    @classmethod
    def poll(self, context):
        return (context.object and context.object.type == 'ARMATURE')

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
            max=100,
            get=get_bvh_index_pos,
            set=set_bvh_index_pos
            )

    bpy.utils.register_module(__name__)



def unregister():
    bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
    register()
