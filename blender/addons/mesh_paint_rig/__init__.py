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

def hex_to_rgb(hex):
    red = ((hex & 0xff0000)>>16) / 255
    green = ((hex & 0xff00)>>8) / 255
    blue = (hex & 0xff) / 255

    return (red, green, blue)


# TODO: factor out into separate module...
#
# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002
#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228 (PSF license)
#
from priodict import priorityDictionary

def Dijkstra(G,start,end=None):
	"""
	Find shortest paths from the  start vertex to all vertices nearer than or equal to the end.

	The input graph G is assumed to have the following representation:
	A vertex can be any object that can be used as an index into a dictionary.
	G is a dictionary, indexed by vertices.  For any vertex v, G[v] is itself a dictionary,
	indexed by the neighbors of v.  For any edge v->w, G[v][w] is the length of the edge.
	This is related to the representation in <http://www.python.org/doc/essays/graphs.html>
	where Guido van Rossum suggests representing graphs as dictionaries mapping vertices
	to lists of outgoing edges, however dictionaries of edges have many advantages over lists:
	they can store extra information (here, the lengths), they support fast existence tests,
	and they allow easy modification of the graph structure by edge insertion and removal.
	Such modifications are not needed here but are important in many other graph algorithms.
	Since dictionaries obey iterator protocol, a graph represented as described here could
	be handed without modification to an algorithm expecting Guido's graph representation.

	Of course, G and G[v] need not be actual Python dict objects, they can be any other
	type of object that obeys dict protocol, for instance one could use a wrapper in which vertices
	are URLs of web pages and a call to G[v] loads the web page and finds its outgoing links.

	The output is a pair (D,P) where D[v] is the distance from start to v and P[v] is the
	predecessor of v along the shortest path from s to v.

	Dijkstra's algorithm is only guaranteed to work correctly when all edge lengths are positive.
	This code does not verify this property for all edges (only the edges examined until the end
	vertex is reached), but will correctly compute shortest paths even for some graphs with negative
	edges, and will raise an exception if it discovers that a negative edge has caused it to make a mistake.
	"""

	D = {}	# dictionary of final distances
	P = {}	# dictionary of predecessors
	Q = priorityDictionary()	# estimated distances of non-final vertices
	Q[start] = 0

	for v in Q:
		D[v] = Q[v]
		if v == end: break

		for w in G[v]:
			vwLength = D[v] + G[v][w]
			if w in D:
				if vwLength < D[w]:
					raise ValueError("Dijkstra: found better path to already-final vertex")
			elif w not in Q or vwLength < Q[w]:
				Q[w] = vwLength
				P[w] = v

	return (D,P)


def shortestPath(G,start,end):
	"""
	Find a single shortest path from the given start vertex to the given end vertex.
	The input has the same conventions as Dijkstra().
	The output is a list of the vertices in order along the shortest path.
	"""

	D,P = Dijkstra(G,start,end)
	Path = []
	while 1:
		Path.append(end)
		if end == start: break
		end = P[end]
	Path.reverse()
	return Path


# example, CLR p.528
#G = {'s': {'u':10, 'x':5},
#     'u': {'v':1, 'x':2},
#     'v': {'y':4},
#     'x': {'u':3,'v':9,'y':2},
#     'y': {'s':7,'v':6}}
#
#print(Dijkstra(G,'s'))
#print(shortestPath(G,'s','u'))

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
                'color': hex_to_rgb(0xff0000),
            },
            'neck_01': {
                'color': hex_to_rgb(0xff9100),
            },
            'upperarm_l': {
                'color': hex_to_rgb(0xffea00),
            },
            'upperarm_r': { 
                'color': hex_to_rgb(0xaaff00),
            },
            'lowerarm_l': {
                'color': hex_to_rgb(0x00ff9d),
            },
            'lowerarm_r': {
                'color': hex_to_rgb(0x00fffb),
            },
            'hand_l': {
                'color': hex_to_rgb(0x00a6ff),
            },
            'hand_r': {
                'color': hex_to_rgb(0x0026ff),
            }, 
            'thigh_l': {
                'color': hex_to_rgb(0x8c00ff),
            },
            'thigh_r': {
                'color': hex_to_rgb(0xfb00ff),
            },
            'calf_l': {
                'color': hex_to_rgb(0x4d3d28),
            },
            'calf_r': {
                'color': hex_to_rgb(0x4d323b),
            },
            'foot_l': {
                'color': hex_to_rgb(0xe193ad),
            },
            'foot_r': {
                'color': hex_to_rgb(0x0e560e),
            },
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
