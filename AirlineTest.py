import unittest
import Airline


def resolve_edges(graph, indices):
        edges = []
        for idx in indices:
            e = graph.edges[idx]
            edges.append((e.from_, e.to))
        return edges


class TestBreadthFirstSearch(unittest.TestCase):
    
    def test_breadth_first_search(self):
        f = Airline.FlowGraph(4)
        f.add_edge(0, 1, 1)
        f.add_edge(0, 2, 1)
        f.add_edge(1, 3, 1)
        f.add_edge(2, 3, 1)

        (flow, path) = Airline.breadth_first_search(f, 0, 3)
        
        self.assertEqual(flow, 1)
        edges = resolve_edges(f, path)
        expect = [(0, 2), (2, 3)]
        self.assertEqual(edges, expect)
    
    def test_another_breadth_first_search(self):
        f = Airline.FlowGraph(4)
        f.add_edge(0, 1, 10000)
        f.add_edge(0, 2, 10000)
        f.add_edge(1, 2, 1)
        f.add_edge(1, 3, 10000)
        f.add_edge(2, 3, 10000)
        
        (flow, path) = Airline.breadth_first_search(f, 0, 3)
        
        self.assertEqual(flow, 10000)
        edges = resolve_edges(f, path)
        expect = [(0, 2), (2, 3)]
        self.assertEqual(edges, expect)


class TestMaxMatching(unittest.TestCase):
    
    def test_easy_matching(self):
        print('\n\n\n-= BEGIN MATCH TEST =-')
        m = Airline.MaxMatching()
        test_matrix = [(1,)]
        g = m.build_graph(test_matrix)
        (flow, g_prime) = Airline.max_flow(g, 0, 3)
        self.assertEqual(flow, 1)
    
    def test_medium_matching(self):
        print('\n\n\n-= BEGIN MATCH TEST TWO =-')
        m = Airline.MaxMatching()
        test_matrix = [(1, 0, 0, 1),
                       (0, 1, 0, 1),
                       (0, 0, 1, 1)]
        g = m.build_graph(test_matrix)
        (flow, g_prime) = Airline.max_flow(g, 0, 8)
        self.assertEqual(flow, 3)
        matching = m.find_matching(g_prime, (4, 3))
        self.assertEqual(matching, [1, 2, 3])
        
    def test_eg_one(self):
        m = Airline.MaxMatching()
        test_matrix = [(1, 1, 0, 1),
                       (0, 1, 0, 0),
                       (0, 0, 0, 0)]
        g = m.build_graph(test_matrix)
        (flow, g_prime) = Airline.max_flow(g, 0, 8)
        self.assertEqual(flow, 2)
        matching = m.find_matching(g_prime, (4, 3))
        self.assertEqual(matching, [1, 2, -1])

if __name__ == '__main__':
    unittest.main()

