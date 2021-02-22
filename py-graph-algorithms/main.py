#Author: Mykola Halynsky
from Graph import Graph

my_graph = Graph("dir/data.txt", sep=',', is_directed=True)

# Select any method.
# my_graph.dijkstra_search(start=8, finish=7)
# my_graph.prim_search(4)
# my_graph.search_in_width(0)
# my_graph.search_in_depth(5)
# my_graph.a_star(7, 8)
# my_graph.ford_falkerson(0, 6)
# my_graph.bidirectional_dijkstra(1, 2)
# my_graph.bidirectional_a_star(7, 4)

my_graph.draw()
