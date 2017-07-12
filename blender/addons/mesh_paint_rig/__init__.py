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
import mathutils
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
import bmesh

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

    debug = True

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
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


        # Each bone can have a sequence of paints that are tested in
        # order for whether they should be applied to the current face
        #
        # 'color' is the rgb color of the paint
        #
        # Alternatively 'color_left' and 'color_right' can be used to
        #   define separate colors for the left and right side of the
        #   mesh
        #
        # 'rel_threshold_limit' determines (relative to the length of
        #   the bone) the distance from the bone head that the paint
        #   should be applied
        #
        # 'speed' affects the relative fluidity of the paint, or how
        #   quickly it flows/spreads compared to the pain for other
        #   bones.
        #
        # 'obj_y+only': True means that the paint should only flow
        #   along the positive 'y' direction of the bone (i.e. the
        #   direction from the head->tail)
        #
        # 'global_z-only': True means that the paint should only flow
        #   down (in global, world-space coordinates)
        #
        boneheads = {
            'head': {
                'paints': [{
                    'color_left': hex_to_rgb(0xc3b2ff),
                    'color_right': hex_to_rgb(0xffa9ca),
                    'rel_threshold_limit': 0.8,
                    'id_left': 1,
                    'id_left': 2
                },{
                    'color_left': hex_to_rgb(0x68c0d1),
                    'color_right': hex_to_rgb(0x70d19f),
                    'obj_y+only': True,
                    'id_left': 3,
                    'id_right': 4,
                }],
            },
            'neck_01': {
                'paints': [{
                    'color': hex_to_rgb(0xff9100),
                    'id': 5,
                }],
            },
            'clavicle_l': {
                'paints': [{
                    'color': hex_to_rgb(0xd73ce0),
                    'speed': 1.2,
                    'id': 6,
                }],
            },
            'clavicle_r': {
                'paints': [{
                    'color': hex_to_rgb(0x291773),
                    'speed': 1.2,
                    'id': 7,
                }],
            },
            'upperarm_l': {
                'paints': [{
                    'color': hex_to_rgb(0xffea00),
                    'rel_threshold_limit': 0.5,
                    'id': 8,
                },{
                    'color': hex_to_rgb(0xd19795),
                    'obj_y+only': True,
                    'speed': 0.75,
                    'id': 9,
                }],
            },
            'upperarm_r': { 
                'paints': [{
                    'color': hex_to_rgb(0xaaff00),
                    'rel_threshold_limit': 0.5,
                    'id': 10,
                },{
                    'color': hex_to_rgb(0xb3d166),
                    'obj_y+only': True,
                    'speed': 0.75,
                    'id': 11,
                }],
            },
            'lowerarm_l': {
                'paints': [{
                    'color': hex_to_rgb(0x00ff9d),
                    'rel_threshold_limit': 0.33,
                    'id': 12,
                },{
                    'color': hex_to_rgb(0xa1d6ae),
                    'obj_y+only': True,
                    'id': 13,
                }],
            },
            'lowerarm_r': {
                'paints': [{
                    'color': hex_to_rgb(0x00fffb),
                    'rel_threshold_limit': 0.33,
                    'id': 14,
                },{
                    'color': hex_to_rgb(0xd6d6d6),
                    'obj_y+only': True,
                    'id': 15,
                }],
            },
            'hand_l': {
                'paints': [{
                    'color': hex_to_rgb(0xd6c56f),
                    'rel_threshold_limit': 0.6,
                    'id': 16,
                },{
                    'color': hex_to_rgb(0x00a6ff),
                    'obj_y+only': True,
                    'id': 17,
                }],
            },
            'hand_r': {
                'paints': [{
                    'color': hex_to_rgb(0x35a29b),
                    'rel_threshold_limit': 0.6,
                    'id': 18,
                },{
                    'color': hex_to_rgb(0x0026ff),
                    'obj_y+only': True,
                    'id': 19,
                }],
            }, 
            'thigh_l': {
                'paints': [{
                    'color': hex_to_rgb(0x8c00ff),
                    'rel_threshold_limit': 0.45,
                    'id': 20,
                },{
                    'color': hex_to_rgb(0x4d274c),
                    'obj_y+only': True,
                    'id': 21,
                }],
            },
            'thigh_r': {
                'paints': [{
                    'color': hex_to_rgb(0xfb00ff),
                    'rel_threshold_limit': 0.45,
                    'id': 22,
                },{
                    'color': hex_to_rgb(0xd78469),
                    'obj_y+only': True,
                    'id': 23,
                }],
            },
            'calf_l': {
                'paints': [{
                    'color': hex_to_rgb(0x7d3d28),
                    'rel_threshold_limit': 0.33,
                    'id': 24,
                },{
                    'color': hex_to_rgb(0xf28eea),
                    'obj_y+only': True,
                    'id': 25,
                }],
            },
            'calf_r': {
                'paints': [{
                    'color': hex_to_rgb(0x6d723b),
                    'rel_threshold_limit': 0.33,
                    'id': 26,
                },{
                    'color': hex_to_rgb(0xffbb19),
                    'obj_y+only': True,
                    'id': 27,
                }],
            },
            'foot_l': {
                'paints': [{
                    'color': hex_to_rgb(0x0e560e),
                    'rel_threshold_limit': 0.65,
                    'id': 28,
                },{
                    'color': hex_to_rgb(0xffe394),
                    'obj_y+only': True,
                    'global_z-only': True,
                    'id': 29,
                }],
            },
            'foot_r': {
                'paints': [{
                    'color': hex_to_rgb(0xe193ad),
                    'rel_threshold_limit': 0.65,
                    'id': 30,
                },{
                    'color': hex_to_rgb(0x0a821d),
                    'obj_y+only': True,
                    'global_z-only': True,
                    'id': 31,
                }],
            },
            'spine_03': {
                'paints': [{
                    'color_left': hex_to_rgb(0xb28ee9),
                    'color_right': hex_to_rgb(0xe9706d),
                    'id_left': 32,
                    'id_right': 33,
                }],
            },
        }

        G = {}

        # Build up graph that's useable with the Dijkstra's algorithm
        # implementation above
        #
        # TODO: tweak the representation supported above so we can
        # attach more data to the vertices and link this structure with
        # blender's mesh data
        #
        for poly in mesh_obj.data.polygons:

            prev_vert_index = mesh_obj.data.loops[poly.loop_start + poly.loop_total - 1].vertex_index
            prev_vert_obj_pos = mesh_obj.data.vertices[prev_vert_index].co
            prev_vert_world_pos = mesh_obj.matrix_world * prev_vert_obj_pos

            for loop_index in poly.loop_indices:
                vert_index = mesh_obj.data.loops[loop_index].vertex_index
                vert_obj_pos = mesh_obj.data.vertices[vert_index].co
                vert_world_pos = mesh_obj.matrix_world * vert_obj_pos

                dx = vert_world_pos[0] - prev_vert_world_pos[0]
                dy = vert_world_pos[1] - prev_vert_world_pos[1]
                dz = vert_world_pos[2] - prev_vert_world_pos[2]

                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                # NB: need something immutable to use as a dictionary key
                vert0_key = (prev_vert_world_pos[0], prev_vert_world_pos[1], prev_vert_world_pos[1])
                vert1_key = (vert_world_pos[0], vert_world_pos[1], vert_world_pos[1])

                if vert0_key not in G:
                    G[vert0_key] = {}

                G[vert0_key][vert1_key] = dist

                prev_vert_world_pos = vert_world_pos


        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.faces.ensure_lookup_table()

        #bm2 = bm.copy()
        #bm2.faces.ensure_lookup_table()

        bm_col_layer = bm.loops.layers.color.verify()
        #bm2_col_layer = bm.loops.layers.color.verify()

        for t in range(0, 500, 5):
            base_thresh = (1/1000.0) * t

            for bone in pose_obj.pose.bones:
                if bone.name not in boneheads:
                    continue

                bone_data = boneheads[bone.name]

                for paint in bone_data['paints']:
                    if 'speed' in paint:
                        paint_thresh = base_thresh * paint['speed']
                    else:
                        paint_thresh = base_thresh

                    if 'rel_threshold_limit' in paint:
                        limit = bone.length * paint['rel_threshold_limit']

                        if paint_thresh > limit:
                            paint['threshold'] = limit
                        else:
                            paint['threshold'] = paint_thresh
                    else:
                        paint['threshold'] = paint_thresh

                self.report({'INFO'}, "joint " + bone.name)

                bonehead_obj_pos = bone.head.xyz
                bonehead_world_pos = pose_obj.matrix_world * bonehead_obj_pos

                bone_world_mat_inv = mesh_obj.matrix_world * bone.matrix
                bone_world_mat_inv.invert()

                for face in bm.faces:

                    current_col = face.loops[0][bm_col_layer]
                    if current_col[0] != 1.0 or current_col[1] != 1.0 or current_col[2] != 1.0:
                        continue

                    n_poly_verts = 0
                    x_tot = 0;
                    y_tot = 0;
                    z_tot = 0;

                    for v in face.verts:
                        n_poly_verts = n_poly_verts + 1

                        vert_obj_pos = v.co
                        vert_world_pos = mesh_obj.matrix_world * vert_obj_pos
                        x_tot = x_tot + vert_world_pos[0]
                        y_tot = y_tot + vert_world_pos[1]
                        z_tot = z_tot + vert_world_pos[2]

                    x_avg = x_tot / n_poly_verts
                    y_avg = y_tot / n_poly_verts
                    z_avg = z_tot / n_poly_verts

                    dx = x_avg - bonehead_world_pos[0]
                    dy = y_avg - bonehead_world_pos[1]
                    dz = z_avg - bonehead_world_pos[2]

                    dist = math.sqrt(dx**2 + dy**2 + dz**2)

                    for paint in bone_data['paints']:

                        in_bounds = False
                        if dist < paint['threshold']:
                            in_bounds = True

                            if 'obj_y+only' in paint:
                                bone_space_pos = bone_world_mat_inv * mathutils.Vector((x_avg, y_avg, z_avg))

                                if bone_space_pos[1] < 0:
                                    in_bounds = False

                            if 'global_z-only' in paint:
                                if dz > 0:
                                    in_bounds = False


                        if in_bounds:
                            if self.debug == True:
                                if 'color' in paint:
                                    bone_col = paint['color']
                                elif 'color_left' in paint and x_avg >= 0:
                                    bone_col = paint['color_left']
                                elif 'color_right' in paint and x_avg < 0:
                                    bone_col = paint['color_right']
                            else:
                                if 'id' in paint:
                                    grey_id = paint['id']
                                elif 'id_left' in paint and x_avg >= 0:
                                    grey_id = paint['id_left']
                                elif 'id_right' in paint and x_avg < 0:
                                    grey_id = paint['id_right']
                                bone_col = (grey_id * (1/255), grey_id * (1/255), grey_id * (1/255))

                            for loop in face.loops:
                                loop[bm_col_layer] = bone_col

                            break

        bm.to_mesh(mesh_obj.data)
        bm.free()

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
