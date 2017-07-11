# python3

'''
Why are we stuck with the crappy API?

1. We want fast lookup of 'reverse' edges in the graph (using the XOR 1 trick)
    - Maybe solved by storing edge indices in dicts
2. We might have multiple edges linking two nodes, or even edges from a node
   to itself.
    - This means we cannot simply say add_flow(from, to, flow) and hope
     to hit the correct edge. (We might, but it would be by luck)
    - How about storing the graph indices as:
        [{1: #{1, 3, 5}, 2: #{7, 8, 10}},
         {0: #{...}, 2: #{...}},
         ...]

Other OCD unhappiness:
    
1. Does MaxMatching really need to be a class?
Calling g = m.build_graph() feels really weird.
I'd like to kill the class and coordinate the call graph myself from another
(extra-class) function.
       
'''

import collections


class Edge:

    def __init__(self, from_, to, capacity):
        self.from_ = from_
        self.to = to
        self.capacity = capacity
        self.flow = 0


class FlowGraph:
    '''This class implements a bit unusual scheme for storing edges of the graph,
    in order to retrieve the backward edge for a given edge quickly.'''
    
    '''Thoughts:
        API:
            make_edge
            make_bipartite_graph
            find_path
            add_flow
        
        I think I can handle add_flow with (from, to, flow) as params
        If so, find_path can be simplified nicely
    '''

    def __init__(self, n):
        # List of all - forward and backward - edges
        self.edges = []
        # These adjacency lists store only indices of edges in the edges list
        # Consider using a map from vertex (e.to)
        # to edge index (edges[i]) - Dave
        # self.graph = [[] for _ in range(n)]
        self.graph = [{} for _ in range(n)]

    def add_edge(self, from_, to, capacity):
        # Note that we first append a forward edge and then a backward edge,
        # so all forward edges are stored at even indices (starting from 0),
        # whereas backward edges are stored at odd indices.
        forward_edge = Edge(from_, to, capacity)
        backward_edge = Edge(to, from_, 0)
        # self.graph[from_].append(len(self.edges))
        self.graph[from_][to] = len(self.edges)
        self.edges.append(forward_edge)
        # self.graph[to].append(len(self.edges))
        self.graph[to][from_] = len(self.edges)
        self.edges.append(backward_edge)

    def num_vertices(self):
        return len(self.graph)

    def num_edges(self):
        return len(self.edges)

    def get_edge_indices(self, from_):
        return self.graph[from_].items()

    def get_edge(self, id):
        return self.edges[id]

    def add_flow(self, id, flow):
        '''To get a backward edge for a true forward edge (i.e id is even),
        we should get id + 1 due to the described above scheme.
        On the other hand, when we have to get a "backward" edge
        for a backward edge (i.e. get a forward edge for backward - id is odd),
        id - 1 should be taken.

        It turns out that id ^ 1 works for both cases. Think this through!'''
        self.edges[id].flow += flow
        self.edges[id ^ 1].flow -= flow
    
        '''
        # aspirational API
        # doesn't handle multiple edges
        def add_flow(_, from, to, flow):
            idx = self.graph[from_][to]
            self.edges[idx].flow += flow
            self.edges[idx ^ 1].flow -= flow
        '''


def breadth_first_search(graph, from_, to):
    '''Make a graph
    >>> f = FlowGraph(4)
    >>> f.add_edge(0, 1, 1)
    >>> f.add_edge(0, 2, 1)
    >>> f.add_edge(1, 3, 1)
    >>> f.add_edge(2, 3, 1)
    
    Then do a BFS
    >>> breadth_first_search(f, 0, 3)
    (1, [0, 4])
    '''
    predecessor = [-1 for _ in range(graph.num_edges())]
    visited = [False for _ in range(graph.num_vertices())]
    visited[from_] = True

    q = collections.deque()
    for (dest, idx) in graph.get_edge_indices(from_):
        e = graph.edges[idx]
        if (e.capacity - e.flow > 0):
            q.append(idx)
            predecessor[dest] = from_

    while len(q) > 0:
        i = q.pop()
        e1 = graph.edges[i]
        # print('examing edge: {0} --> {1}'.format(e1.from_, e1.to))
        for (k, v) in graph.get_edge_indices(e1.to):
            e2 = graph.edges[v]
            # print('examing inner edge: {0} --> {1}'.format(e2.from_, e2.to))
            if (not visited[k]) and (e2.capacity - e2.flow > 0):
                # print('added inner edge to queue')
                predecessor[v] = i
                visited[k] = True
                q.append(v)

        # print(
            # 'about to consider whether {0} --> {1} leads to the end'
            # .format(e1.from_, e1.to))

        if e1.to == to:
            # print('It does!')
            min_flow = e1.capacity - e1.flow
            path = []
            while True:
                path.append(i)
                edge_cap = graph.edges[i].capacity - graph.edges[i].flow
                min_flow = min(min_flow, edge_cap)
                if graph.edges[i].from_ == from_:
                    # print('returning the path: {}'.format(path))
                    return (min_flow, path[::-1])
                i = predecessor[i]

    # print('returning empty path')
    return (0, [])


def max_flow(graph, from_, to):
    '''Returns a tuple: (total flow, modified graph)'''
    total_flow = 0
    while True:
        (flow, path) = breadth_first_search(graph, from_, to)
        # print('flow: {0} path: {1}'.format(flow, path))
        if flow == 0:
            break
        for idx in path:
            graph.add_flow(idx, flow)
        total_flow += flow
    return (total_flow, graph)


class MaxMatching:
    def read_data(self):
        n, m = map(int, input().split())
        adj_matrix = [list(map(int, input().split())) for i in range(n)]
        return adj_matrix

    def write_response(self, matching):
        line = [str(-1 if x == -1 else x) for x in matching]
        print(' '.join(line))

    def build_graph(self, adj_matrix):
        if not len(adj_matrix) > 0:
            return
        # build graph
        num_flights = len(adj_matrix)
        num_crews = len(adj_matrix[0])
        g = FlowGraph(num_crews + num_flights + 2)
        
        # Add edges from crews to flights
        for (f_idx, flight) in enumerate(adj_matrix, num_crews + 1):
            for (c_idx, able) in enumerate(flight, 1):
                if able:
                    g.add_edge(c_idx, f_idx, 1)
                    # print('added edge from {0} to {1}'.format(c_idx, f_idx))

        # Hook up to source and sink
        source_idx = 0
        for idx in range(1, num_crews + 1):
            g.add_edge(source_idx, idx, 1)
            # print('added edge from {0} to {1}'.format(source_idx, idx))

        sink_idx = g.num_vertices() - 1
        #                /- first of flights  /- last idx of flights
        for idx in range(num_crews + 1, sink_idx):
            g.add_edge(idx, sink_idx, 1)
            # print('added edge from {0} to {1}'.format(idx, sink_idx))
        
        return g
    
    def find_matching(_, graph, bipartition):
        # call max flow on graph
        (flow, g_prime) = max_flow(graph, 0, graph.num_vertices())
        # print('flow of {}'.format(flow))
        (l, r) = bipartition
        # assume that crews are on left, flights on right
        matches = []
        # iterate over flight vertices
        for vertex in range(l + 1, l + r + 1):
            for (dest, idx) in graph.get_edge_indices(vertex):
                # check if edge ends inside crews partition
                # and has capacity
                # print('got an edge ending at {}'.format(dest))
                e = graph.edges[idx]
                if (e.capacity - e.flow > 0) and (1 <= dest <= l):
                    # print('adding it!')
                    matches.append(dest)
                    break
            else:
                matches.append(-1)

        return matches

    def solve(self):
        adj_matrix = self.read_data()
        graph = self.build_graph(adj_matrix)
        n_flights = len(adj_matrix)
        n_crews = len(adj_matrix[0])
        (_, g_prime) = max_flow(graph, 0, n_crews + n_flights + 1)
        matching = self.find_matching(g_prime, (n_crews, n_flights))
        self.write_response(matching)

if __name__ == '__main__':
    max_matching = MaxMatching()
    max_matching.solve()

