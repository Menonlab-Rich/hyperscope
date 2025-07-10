import networkx as nx
import numpy as np
from rsklearn import clustering
from joblib import hash as jlhash, Parallel, delayed, memory
from metavision_core.event_io import EventsIterator
from tqdm import tqdm
import itertools

class EventGraph():
    def __init__(self, events: EventsIterator, jobs: int = 4, connect_edges: bool = True):
        self.graph = nx.Graph()
        self.jobs = jobs # Store jobs for use in other methods
        dbscan = clustering.DBScan(eps=13, min_samples=5, metric='Euclidean')
        nodes = [] 
        print("Loading events...")
        for evts_buffer in tqdm(events, desc="Event Buffers"):
            coords = [(e['x'], e['y']) for e in evts_buffer]
            if not len(coords):
                print('no coords')
                continue
            coords = np.array(coords, dtype=np.float32)
            labels = dbscan.fit(coords)
            uniq_labels = np.unique(labels)
            for label in uniq_labels:
                if label == -1:
                    continue
                points = coords[labels == label]
                centroid = np.mean(points, axis=0)
                xyt = np.append(centroid, evts_buffer[-1]['t'])
                hash, attrs = self._prepare_event_data(xyt)
                nodes.append((hash, attrs))
        if len(nodes) < 2:
            raise(ValueError('Not Enough Nodes to Form a Graph'))
        self.graph.add_nodes_from(nodes)
        print(f'Added {len(nodes)} to graph')
        # Cleanup nodes here to help memory because graph now contains them all
        del nodes
        self._add_edges()


    def _get_node_pairs(self):
        nodes = list(self.graph.nodes)
        pairs = list(itertools.combinations(nodes, 2))
        return pairs

    def _prepare_event_data(self, evt):
        event_hash = jlhash(evt) 
        attributes = {'t': int(evt[2]), 'x': int(evt[0]), 'y': int(evt[1])}
        return event_hash, attributes

    def _calculate_edge_weight(self, hash1, hash2):
        attrs1 = self.graph.nodes[hash1]
        attrs2 = self.graph.nodes[hash2]
        dist_sq = (attrs1['t'] - attrs2['t'])**2 + \
                  (attrs1['x'] - attrs2['x'])**2 + \
                  (attrs1['y'] - attrs2['y'])**2
        distance = np.sqrt(dist_sq)
        return distance

    def _add_edges(self):
        pairs = self._get_node_pairs()
        edges = [(n1, n2, self._calculate_edge_weight(n1, n2)) for n1, n2 in pairs]
        self.graph.add_weighted_edges_from(edges)
            




# Example Usage (assuming EventsIterator is defined and works)
if __name__ == '__main__':
    raw_file_path = '../../data/raw/metavision/combo_nomod.raw' 
    
    # Test with a small number of events first
    # To test the N < 2 case for edges:
    # mv_iterator_test_few = EventsIterator("test_path_few_events", max_events=1)
    # event_graph_instance_few = EventGraph(mv_iterator_test_few, jobs=2, connect_edges=True)
    # print(f"Few Events - Nodes: {event_graph_instance_few.graph.number_of_nodes()}, Edges: {event_graph_instance_few.graph.number_of_edges()}")


    mv_iterator = EventsIterator(raw_file_path) 
    
    num_jobs = -1 # Or 1 to test serial path, or 0/-1 for all cores
    print(f"Initializing EventGraph with jobs={num_jobs}")
    event_graph_instance = EventGraph(mv_iterator, jobs=num_jobs, connect_edges=True) 
    
    print(f"Number of nodes in graph: {event_graph_instance.graph.number_of_nodes()}")
    print(f"Number of edges in graph: {event_graph_instance.graph.number_of_edges()}")
