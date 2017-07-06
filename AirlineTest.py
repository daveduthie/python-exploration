import unittest
import Airline


class TestMaxMatching(unittest.TestCase):
    def test_breadth_first_search(self):
        f = Airline.FlowGraph(4)
        f.add_edge(0, 1, 1)
        f.add_edge(0, 2, 1)
        f.add_edge(1, 3, 1)
        f.add_edge(2, 3, 1)

        p = f.breadth_first_search(0, 3)
        edges = []
        for idx in p:
            e = f.edges[idx]
            edges.append((e.from_, e.to))

        expect = [(0, 1), (1, 3)]

        self.assertEqual(edges, expect)

    def test_easy_matching(self):
        m = Airline.MaxMatching()
        test_matrix = [(0, 1, 0),
                       (1, 0, 1),
                       (0, 1, 0)]

        result = m.find_matching(test_matrix)
        self.assertEqual([1, 0, -1], result)

if __name__ == '__main__':
    unittest.main()

