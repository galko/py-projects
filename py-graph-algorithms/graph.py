#Author: Mykola Halynskyi
import math

import networkx as nx
import matplotlib.pyplot as plt
import time


class Graph:
    labels = dict()
    text = ''
    color_map_node = []
    color_map_edge = []

    def __init__(self, location, sep=' ', is_directed=False):
        if is_directed:
            self.G = nx.read_edgelist(location, delimiter=sep, nodetype=int, edgetype=int, data=(('weight', float),),
                                      create_using=nx.DiGraph())
        else:
            self.G = nx.read_edgelist(location, delimiter=sep, nodetype=int, edgetype=int, data=(('weight', float),), )
        self.labels = nx.get_edge_attributes(self.G, 'weight')
        for _ in self.G.nodes:
            self.color_map_node.append('red')
        for _ in self.G.edges:
            self.color_map_edge.append('black')
        self.draw()

    def change_color(self, node_index, color):
        if self.color_map_node[node_index - 1] != 'green':
            self.color_map_node[node_index - 1] = color

    def animation_helper(self, start):
        self.text += str(start) + '->'
        self.change_color(start, "green")
        self.draw()
        time.sleep(0.5)

    def draw(self):
        plt.figure(figsize=(16, 9))
        text = self.text[:-2]
        plt.text(-1, 1.1, text, fontweight='bold', fontsize=16)

        nx.draw_circular(self.G, with_labels=True, node_color=self.color_map_node,
                         font_weight='bold', width=3, edge_color=self.color_map_edge,
                         font_size=20, node_size=550, arrowsize=20)

        layout = nx.circular_layout(self.G)

        nx.draw_networkx_edge_labels(self.G, layout, edge_labels=self.labels, label_pos=0.4, font_size=16,
                                     font_weight='bold', font_color='red')

        plt.show()

    def labels_helper(self, my_list, stok):
        labels = nx.get_edge_attributes(self.G, 'weight')
        edges = list(self.G.edges)

        adjacent = self.adjacent(stok)
        counter = 0
        for _ in adjacent:
            counter += (my_list[edges.index((_, stok))][1])

        self.text = 'Max Flow Is ' + str(counter) + '  '
        i = 0
        for edge in edges:
            labels[edge] = str(my_list[i][0]) + '/' + str(my_list[i][1])
            i += 1
        return labels

    def adjacent(self, s):
        adjacent = []

        for a in sorted(self.G.edges):
            if int(a[0]) == s:
                adjacent.append(int(a[1]))
            elif int(a[1]) == s:
                adjacent.append(int(a[0]))

        return adjacent

    def search_in_width(self, start=0, sec=0.5):
        self.text = ""
        nodes_tuple = sorted(tuple(self.G.nodes))

        visited = [False] * len(nodes_tuple)

        queue = [int(nodes_tuple[start])]

        visited[int(nodes_tuple[start])] = True

        while queue:
            s = queue.pop(0)
            self.animation_helper(s)

            adjacent = self.adjacent(s)

            for j in adjacent:
                self.change_color(j, "yellow")

            self.draw()
            time.sleep(sec)

            for i in adjacent:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    def search_in_depth_help_func(self, start, visited, sec=0.5):
        visited[start] = True

        self.animation_helper(start)

        adjacent = self.adjacent(start)

        for j in adjacent:
            self.change_color(j, "yellow")

        self.draw()
        time.sleep(sec)

        for i in adjacent:
            if not visited[i]:
                self.search_in_depth_help_func(i, visited)

    def search_in_depth(self, start=0, sec=0.5):
        self.text = ""
        visited = [False] * (len(tuple(self.G.nodes)))

        self.change_color(start, "green")
        self.draw()
        time.sleep(sec)

        self.search_in_depth_help_func(start, visited)

    def prim_search(self, start=0, sec=0.9):

        self.text = ""
        self.text += str(start) + '->'
        visited = set()
        visited.add(start)
        while len(visited) < len(self.G.nodes):

            my_list = list()
            for visited_node in visited:
                a = dict(self.G[visited_node])
                for node in tuple(a.keys()):
                    if not visited.__contains__(node):
                        my_list.append((node, visited_node))

            min_int = float('inf')
            next_destination_pair = tuple()

            for pair in my_list:
                weight = self.G[pair[0]][pair[1]]['weight']
                if weight < min_int:
                    min_int = weight
                    next_destination_pair = pair

            if visited.__contains__(next_destination_pair[0]):
                visited.add(next_destination_pair[1])
                self.text += str(next_destination_pair[1]) + "->"
            else:
                visited.add(next_destination_pair[0])
                self.text += str(next_destination_pair[0]) + "->"

            test_list = list(self.G.edges)
            if test_list.__contains__(next_destination_pair):
                self.color_map_edge[test_list.index(next_destination_pair)] = 'blue'
                self.change_color(next_destination_pair[0], "orange")
                self.change_color(next_destination_pair[1], "orange")
            else:
                self.color_map_edge[test_list.index((next_destination_pair[1], next_destination_pair[0]))] = 'blue'
                self.change_color(next_destination_pair[0], "orange")
                self.change_color(next_destination_pair[1], "orange")
            self.draw()
            time.sleep(sec)

    def dijkstra_search(self, start, finish):
        last_node_plus_weight = list()
        done = set()
        visited = list()  # but not done

        visited.append(start)

        for node in self.G.nodes:
            last_node_plus_weight.append((node, float('inf')))

        while len(done) < len(self.G.nodes):
            v = visited.pop(0)  # main node for iteration

            self.text = ""

            adjacent = self.adjacent(v)

            queue = list()
            for adjacent_node in adjacent:
                queue.append(adjacent_node)

            list.sort(queue, key=lambda element: self.G[v][element]['weight'])
            visited += list.copy(queue)
            for adjacent_node in adjacent:
                if not visited.__contains__(adjacent_node) and not done.__contains__(adjacent_node):
                    visited.append(adjacent_node)
            visited = list(dict.fromkeys(visited))

            while queue:
                curr = queue.pop(0)
                if v != start:
                    weight = self.G[curr][v]['weight'] + last_node_plus_weight[v][1]
                else:
                    weight = self.G[curr][v]['weight']

                if weight < last_node_plus_weight[curr][1]:
                    last_node_plus_weight[curr] = (v, weight)

            done.add(v)
        self.animation_and_text_helper_dijkstra_and_astar(start, finish, last_node_plus_weight)

    def animation_and_text_helper_dijkstra_and_astar(self, start, finish, last_node_plus_weight):
        self.text = self.dijkstra_helper(start, finish, last_node_plus_weight)[0]

        nodes = self.dijkstra_helper(start, finish, last_node_plus_weight)[1]

        i = 0
        while i < len(nodes) - 1:
            test_list = list(self.G.edges)
            self.change_color(nodes[i], "orange")
            self.change_color(nodes[i + 1], "orange")
            if test_list.__contains__((nodes[i], nodes[i + 1])):
                self.color_map_edge[test_list.index((nodes[i], nodes[i + 1]))] = 'blue'
            else:
                self.color_map_edge[test_list.index((nodes[i + 1], nodes[i]))] = 'blue'
            i += 1

    @staticmethod
    def dijkstra_helper(strt, fnsh, result_list):
        string_result = ""
        queue = list()
        sum_weights = 0
        index = fnsh
        queue.append(index)
        string_result += '>-' + str(index)
        sum_weights += result_list[index][1]
        while index != strt:
            index = result_list[index][0]
            queue.append(index)
            string_result += '>-' + str(index)
        string_result = string_result[::-1]
        string_result = string_result[:-2]
        string_result += "  Sum = " + str(sum_weights) + "##"
        return string_result, queue

    # help function for getting evrestcic distance
    # h(v) = \sqrt {(v.x - goal.x) ^ 2 + (v.y - goal.y) ^ 2}
    # Если передвижение не ограничено сеткой, то можно использовать евклидово расстояние по прямой:
    # https://neerc.ifmo.ru/wiki/index.php?title=%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_A*
    def evrestic_distance(self, current_node_index, goal_node_distance):
        layout = nx.circular_layout(self.G)
        current_x = layout[current_node_index][0]
        current_y = layout[current_node_index][1]
        goal_x = layout[goal_node_distance][0]
        goal_y = layout[goal_node_distance][1]
        return math.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)

    def a_star(self, start, goal):
        done = list()
        nodes_to_look = list()
        last_node_plus_weight = list()

        for node in self.G.nodes:
            last_node_plus_weight.append((node, float('inf')))

        evrestics = list()
        for _ in self.G.nodes:
            evrestics.append(0)

        nodes_to_look.append(start)
        last_node_plus_weight[start] = (start, 0)
        evrestics[start] = self.evrestic_distance(start, goal)

        while len(nodes_to_look) != 0:
            my_max = float('inf')
            current = 0
            for a in evrestics:
                if 0 < a < my_max and not done.__contains__(evrestics.index(a)):
                    my_max = a
                    current = evrestics.index(a)
            if current == goal:
                self.animation_and_text_helper_dijkstra_and_astar(start, goal, last_node_plus_weight)
                return True
            nodes_to_look.remove(current)
            done.append(current)
            adjacent = self.adjacent(current)
            for v in adjacent:
                if last_node_plus_weight[current][1] != float('inf'):
                    tentative_score = last_node_plus_weight[current][1] + self.G[current][v]['weight']
                else:
                    tentative_score = self.G[current][v]['weight']

                if not done.__contains__(v) and tentative_score < last_node_plus_weight[v][1]:
                    last_node_plus_weight[v] = (current, tentative_score)
                    if last_node_plus_weight[v][1] != float('inf'):
                        evrestics[v] = last_node_plus_weight[v][1] + self.evrestic_distance(v, goal)
                    else:
                        evrestics[v] = self.evrestic_distance(v, goal)
                    if not nodes_to_look.__contains__(v):
                        nodes_to_look.append(v)

        return False

    def ford_falkerson(self, istock, stock):
        done = list()
        nodes_to_look = list()
        last_node_plus_weight = list()
        edge_attributes = list()
        edges = list(self.G.edges)

        for node in self.G.nodes:
            last_node_plus_weight.append((node, float('inf')))
        for edge in edges:
            edge_attributes.append((self.G[edge[0]][edge[1]]['weight'], 0))

        nodes_to_look.append(istock)
        done.append(istock)

        while nodes_to_look:

            current_node = nodes_to_look.pop(0)

            adjacent = self.adjacent(current_node)

            for v in adjacent:
                if edges.__contains__((current_node, v)):
                    cur = edges.index((current_node, v))
                else:  # значит ребро орентировано не с этой вершины не с этой вершины
                    continue

                if done.__contains__(v):
                    continue

                if edge_attributes[cur][1] < edge_attributes[cur][0]:
                    nodes_to_look.append(v)
                    last_node_plus_weight[v] = (current_node, min(last_node_plus_weight[current_node][1],
                                                                  (edge_attributes[cur][0] - edge_attributes[cur][1])))
                    done.append(v)

            if last_node_plus_weight[stock][1] != float('inf'):
                self.potok_change(stock, last_node_plus_weight, edge_attributes)
                done = list()
                last_node_plus_weight[stock] = (stock, float('inf'))
                nodes_to_look = list()
                nodes_to_look.append(istock)
        self.labels = self.labels_helper(edge_attributes, stock)

    def potok_change(self, stock, spisok_metok, edge_metki):
        edges = list(self.G.edges)
        delta = spisok_metok[stock][1]
        prev = spisok_metok[stock][0]
        tmp = stock
        while spisok_metok[prev][1] != float('inf'):
            if edges.__contains__((prev, tmp)):
                index = edges.index((prev, tmp))
            else:
                index = edges.index((tmp, prev))
            self.color_map_edge[index] = "blue"
            edge_metki[index] = (edge_metki[index][0], edge_metki[index][1] + delta)
            tmp = prev
            prev = spisok_metok[tmp][0]
        if edges.__contains__((prev, tmp)):
            index = edges.index((prev, tmp))
        else:
            index = edges.index((tmp, prev))

        self.color_map_edge[index] = "blue"
        edge_metki[index] = (edge_metki[index][0], edge_metki[index][1] + delta)
        return edge_metki

    def bidirectional_dijkstra(self, start, goal):

        prih = list()
        for (u, v) in self.G.edges:
            prih.append(v)
        if not prih.__contains__(goal):
            self.text = "Goal node is source"
            return

        R_G = self.G.reverse()

        dist_o = list()
        prev_o = list()
        proc_original = list()

        proc_reversed = list()
        dist_r = list()
        prev_r = list()

        for _ in range(len(self.G.nodes)):
            prev_r.append(None)
            prev_o.append(None)
            dist_o.append(float('inf'))
            dist_r.append(float('inf'))

        dist_o[start] = 0
        dist_r[goal] = 0

        while True:
            node_index = self.extract_min_dijkstra(dist_o, proc_original)
            self.process_dijkstra(node_index, self.G, dist_o, prev_o, proc_original)
            if proc_reversed.__contains__(node_index):
                result = self.shortest_pass(start, dist_o, prev_o, proc_original, goal, dist_r, prev_r, proc_reversed)
                self.bidirectional_animation_helper(result, start, goal)
                return

            node_r_index = self.extract_min_dijkstra(dist_r, proc_reversed)
            self.process_dijkstra(node_r_index, R_G, dist_r, prev_r, proc_reversed)
            if proc_original.__contains__(node_r_index):
                result = self.shortest_pass(goal, dist_r, prev_r, proc_reversed, start, dist_o, prev_o, proc_original)
                self.bidirectional_animation_helper(result, start, goal)
                return

    def bidirectional_animation_helper(self, result, start, goal, method=True):
        sum_weight = result[0]
        visited_nodes = [start, ]
        kostil = result[1][0]
        visited_nodes += result[1]
        edges = list(self.G.edges)

        if len(visited_nodes) == 2:
            index = edges.index((start, goal))
            self.color_map_edge[index] = "blue"
            self.text = "Sum is " + str(sum_weight) + "##"
        else:
            for a in range(len(visited_nodes) - 1):
                if edges.__contains__((visited_nodes[a], visited_nodes[a + 1])):
                    index = edges.index((visited_nodes[a], visited_nodes[a + 1]))
                elif edges.__contains__((visited_nodes[a+1], visited_nodes[a])):
                    index = edges.index((visited_nodes[a + 1], visited_nodes[a]))
                else:
                    continue
                self.color_map_edge[index] = "blue"

            index = edges.index((kostil, goal))
            self.color_map_edge[index] = "blue"
        if not method:
            if edges.__contains__((kostil, result[1][-1])):
                index = edges.index((kostil, result[1][-1]))
                self.color_map_edge[index] = "black"
            else:
                index = edges.index((result[1][-1], kostil))
                self.color_map_edge[index] = "black"

            self.text = "Sum is " + str(sum_weight) + "##"

    @staticmethod
    def extract_min_dijkstra(dist, proc):
        my_min = float('inf')
        for a in dist:
            if proc.__contains__(dist.index(a)):
                continue
            if a < my_min:
                my_min = a
        return dist.index(my_min)

    def process_dijkstra(self, node, g, dist, prev, proc):
        for to_n in g.adj[node]:
            self.relax_dijkstra(node, to_n, dist, prev)
        proc.append(node)

    def relax_dijkstra(self, from_node, to_node, dist, prev):
        edges = list(self.G.edges)
        if edges.__contains__((from_node, to_node)):
            if dist[to_node] > dist[from_node] + self.G[from_node][to_node]['weight']:
                dist[to_node] = dist[from_node] + self.G[from_node][to_node]['weight']
                prev[to_node] = from_node
        else:
            if dist[to_node] > dist[from_node] + self.G[to_node][from_node]['weight']:
                dist[to_node] = dist[from_node] + self.G[to_node][from_node]['weight']
                prev[to_node] = from_node

    @staticmethod
    def shortest_pass(s, dist, prev, proc, t, r_dist, r_prev, r_proc):
        distance = float('inf')
        ubest = None
        for u in list(set(proc + r_proc)):
            if dist[u] + r_dist[u] < distance:
                ubest = u
                distance = dist[u] + r_dist[u]
        path = list()
        last = ubest
        while last != s:
            path.append(last)
            last = prev[last]

        path.reverse()
        last = ubest
        while last != t:
            last = r_prev[last]
            path.append(last)
        return distance, path

    def bidirectional_a_star(self, start, goal):
        prih = list()
        for (u, v) in self.G.edges:
            prih.append(v)
        if not prih.__contains__(goal):
            self.text = "Goal node is source"
            return

        R_G = self.G.reverse()

        dist_o = list()
        prev_o = list()
        proc_original = list()

        proc_reversed = list()
        dist_r = list()
        prev_r = list()

        for _ in range(len(self.G.nodes)):
            prev_r.append(None)
            prev_o.append(None)
            dist_o.append(float('inf'))
            dist_r.append(float('inf'))

        dist_o[start] = 0
        dist_r[goal] = 0

        while True:
            node_index = self.extract_min_a_stars(dist_o, proc_original, goal)
            self.process_dijkstra(node_index, self.G, dist_o, prev_o, proc_original)
            if proc_reversed.__contains__(node_index):
                result = self.shortest_pass(start, dist_o, prev_o, proc_original, goal, dist_r, prev_r, proc_reversed)
                self.bidirectional_animation_helper(result, start, goal, False)
                return

            node_r_index = self.extract_min_a_stars(dist_r, proc_reversed, goal)
            self.process_dijkstra(node_r_index, R_G, dist_r, prev_r, proc_reversed)
            if proc_original.__contains__(node_r_index):
                result = self.shortest_pass(goal, dist_r, prev_r, proc_reversed, start, dist_o, prev_o, proc_original)
                self.bidirectional_animation_helper(result, start, goal, False)
                return

    def extract_min_a_stars(self, dist, proc, goal):
        my_min = float('inf')
        for a in dist:
            if proc.__contains__(dist.index(a)):
                continue
            distance = self.evrestic_distance(dist.index(a), goal)
            if a + distance < my_min:
                my_min = a
        return dist.index(my_min)
