import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

class SceneGraphSim:
    def __init__(self, full_sg_path=None) -> None:
        self._load_scene_graph(full_sg_path=full_sg_path)
        self._current_graph = nx.Graph()
        self._explored_rooms = []
        self._current_room = ''
    
    def _load_scene_graph(self, full_sg_path=None):
        if full_sg_path:
            with open(full_sg_path, "r") as f:
                self.full_sg_data = json.load(f) 
        else:
            with open("/home/saumyas/Projects/semnav/3DSceneGraph/3dsg_kNN_stairs.json", "r") as f:
                self.full_sg_data = json.load(f)
        
        self.netx_sg = json_graph.node_link_graph(self.full_sg_data)
        self.nodes_all = list(self.netx_sg.nodes)
        self.rooms_all = [n for n in self.nodes_all if 'room' in n]

    def sg_to_string(self, sg):
        return json.dumps(sg)
    
    def get_current_state(self):
        return self.netx_sg.nodes[self._current_room].get('Room_name') # TODO: should it return room or more stuff
    
    def start(self, init_room=None):
        if init_room:
            self._init_room = init_room
        else:
            self._init_room = np.random.choice(self.rooms_all)
        self._current_room = self._init_room

        self._current_nodes = [s for s in self.netx_sg.successors(self._init_room) if 'object' in s]
        # Include the original node
        self._current_nodes.append(self._init_room)
        # Create the subgraph
        self._current_graph = nx.subgraph(self.netx_sg, self._current_nodes)
        self._explored_rooms.append(self._current_room)
        
        return json.dumps(nx.node_link_data(self._current_graph))
    
    def _sample_next_room(self, room_idx=-1): # TODO: if current successors are explored goto nearest other unexplored successors
        next_rooms = [s for s in self.netx_sg.successors(self._explored_rooms[room_idx]) if 'room' in s]
        next_rooms_unexplored = [i for i in next_rooms if i not in self._explored_rooms]
        
        if len(next_rooms_unexplored) > 0:
            return np.random.choice(next_rooms_unexplored)
        else:
            return self._sample_next_room(room_idx=room_idx-1)

    def explore_next_room(self):
        if self._explored_rooms == self.rooms_all: # All rooms explored
            return json.dumps(nx.node_link_data(self._current_graph)), True
        
        next_room = self._sample_next_room(room_idx=-1) # TODO: does a VLM choose this?
        self._current_room = next_room.copy()

        successors = [s for s in self.netx_sg.successors(next_room) if 'object' in s] # objects
        successors.append(next_room)

        self._current_nodes.extend(successors)

        self._current_graph = nx.subgraph(self.netx_sg, self._current_nodes)
        self._explored_rooms.append(self._current_room)

        return json.dumps(nx.node_link_data(self._current_graph)), False
    
    def goto_room(self, room_id):
        if room_id in self._explored_rooms:
            self._current_room = room_id

if __name__=="__main__":
    scene_graph_sim = SceneGraphSim()
    sg_data = scene_graph_sim.start()
    sg_data, _ = scene_graph_sim.explore_next_room()