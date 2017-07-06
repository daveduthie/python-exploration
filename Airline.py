# python3

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

    def get_edges_from(self, from_):
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

    def breadth_first_search(self, from_, to):
        '''Make a graph
        >>> f = FlowGraph(4)
        >>> f.add_edge(0, 1, 1)
        >>> f.add_edge(0, 2, 1)
        >>> f.add_edge(1, 3, 1)
        >>> f.add_edge(2, 3, 1)
        
        Then do a BFS
        >>> f.breadth_first_search(0, 3)
        [0, 4]
        '''
        predecessor = [-1 for _ in range(self.num_edges())]
        visited = [False for _ in range(self.num_vertices())]
        visited[from_] = True

        q = collections.deque()
        q.append(from_)

        while len(q) > 0:
            i = q.pop()
            e1 = self.edges[i]
            for (k, v) in self.get_edges_from(e1.to):
                e2 = self.edges[v]
                if (not visited[k]) and (e2.capacity - e2.flow > 0):
                    predecessor[v] = i
                    visited[k] = True
                    q.append(v)

            # print(
                # 'about to consider whether {0} --> {1} leads to the end'
                # .format(e1.from_, e1.to))

            if e1.to == to:
                path = []
                while True:
                    path.append(i)
                    # print('value of path: {}'.format(path))
                    if self.edges[i].from_ == from_:
                        # print('breaking')
                        break
                    i = predecessor[i]

                # print('returning the path: {}'.format(path))
                return path[::-1]

# Maybe I won't need this
'''
def read_data():
    vertex_count, edge_count = map(int, input().split())
    graph = FlowGraph(vertex_count)
    for _ in range(edge_count):
        u, v, capacity = map(int, input().split())
        graph.add_edge(u - 1, v - 1, capacity)
    return graph
'''


def max_flow(graph, from_, to):
    flow = 0
    # your code goes here
    return flow


class MaxMatching:
    def read_data(self):
        n, m = map(int, input().split())
        adj_matrix = [list(map(int, input().split())) for i in range(n)]
        return adj_matrix

    def write_response(self, matching):
        line = [str(-1 if x == -1 else x + 1) for x in matching]
        print(' '.join(line))

    def find_matching(self, adj_matrix):
        # Replace this code with an algorithm that finds the maximum
        # matching correctly in all cases.
        n = len(adj_matrix)
        m = len(adj_matrix[0])
        matching = [-1] * n
        busy_right = [False] * m
        for i in range(n):
            for j in range(m):
                if (adj_matrix[i][j] and
                   matching[i] == -1 and
                   (not busy_right[j])):
                        matching[i] = j
                        busy_right[j] = True
        return matching

    def solve(self):
        adj_matrix = self.read_data()
        matching = self.find_matching(adj_matrix)
        self.write_response(matching)

if __name__ == '__main__':
    max_matching = MaxMatching()
    max_matching.solve()

