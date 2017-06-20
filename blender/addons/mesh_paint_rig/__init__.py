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
    "name": "Glimpse Rig Paint",
    "description": "Paint mesh vertices for use with Glimpse' skeleton tracking training",
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
import os

import bpy
from bpy.props import (
        CollectionProperty,
        StringProperty,
        BoolProperty,
        EnumProperty,
        FloatProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        orientation_helper_factory,
        axis_conversion,
        )

class PaintRigOperator(bpy.types.Operator):
    """Paint Rig"""
    bl_idname = "object.paint_rig_operator"
    bl_label = "Rig Paint Operator"

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
        self.report({'INFO'}, "Debug 2")

        pose_obj = None
        mesh_obj = None
        for obj in bpy.context.selected_objects:
            if obj.type == 'ARMATURE':
                pose_obj = obj
            elif obj.type == 'MESH':
                mesh_obj = obj

        if pose_obj == None or mesh_obj == None:
            self.report({'ERROR'}, "Need to select a mesh and an armature")
            return {'FINISHED'}

        if mesh_obj.data.vertex_colors:
            vcol_layer = mesh_obj.data.vertex_colors.active
        else:
            vcol_layer = mesh_obj.data.vertex_colors.new()

        # set default color
        for poly in mesh_obj.data.polygons:
            for loop_index in poly.loop_indices:
                vert_index = mesh_obj.data.loops[loop_index].vertex_index
                vcol_layer.data[loop_index].color = (1.0, 1.0, 1.0)

        boneheads = {
            'head': {
                'color': (1.0, 0.0, 0.0),
            },
            'neck_01': {
                'color': (0.0, 1.0, 0.0),
            },
            'upperarm_l': {
                'color': (0.3, 0.7, 0.0),
            },
            'upperarm_r': { 
                'color': (0.0, 0.7, 0.5),
            }, 
            'thigh_l': {
                'color': (0.1, 0.2, 0.9),
            },
            'thigh_r': {
                'color': (0.7, 0.1, 0.5),
            }
        }

        for t in range(0, 500, 5):
            thresh = (1/1000.0) * t

            for bone in pose_obj.pose.bones:
                if bone.name in boneheads:
                    bone_data = boneheads[bone.name]

                    self.report({'INFO'}, "joint " + bone.name)

                    bonehead_obj_pos = bone.head.xyz
                    bonehead_world_pos = pose_obj.matrix_world * bonehead_obj_pos

                    for poly in mesh_obj.data.polygons:
                        for loop_index in poly.loop_indices:
                            vert_index = mesh_obj.data.loops[loop_index].vertex_index

                            vert_obj_pos = mesh_obj.data.vertices[vert_index].co
                            vert_world_pos = mesh_obj.matrix_world * vert_obj_pos

                            dx = vert_world_pos[0] - bonehead_world_pos[0]
                            dy = vert_world_pos[1] - bonehead_world_pos[1]
                            dz = vert_world_pos[2] - bonehead_world_pos[2]
                            dist = math.sqrt(dx**2 + dy**2 + dz**2)

                            current_col = vcol_layer.data[loop_index].color
                            bone_col = bone_data['color']
                            if dist < thresh and current_col[0] == 1.0 and current_col[1] == 1.0 and current_col[2] == 1.0:
                                vcol_layer.data[loop_index].color = bone_col


        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "my_debug_bool")
        row.prop(self, "my_debug_float")

classes = (
    PaintRigOperator,
    )


def register():
    bpy.utils.register_class(PaintRigOperator)


def unregister():
    bpy.utils.unregister_class(PaintRigOperator)

if __name__ == "__main__":
    register()
