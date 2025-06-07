import numpy as np
import matplotlib.pyplot as plt
from py_Utils import  CSpace
import heapq
import time
plt.ion()

class PRM(object):
    def __init__(self, env_map,  max_itr=1000,  dist=10):
        self.max_itr = max_itr
        self.map = env_map 
        self.env_rows, self.env_cols = env_map.shape #rows~y, cols~x
        self.max_dist=dist #[pixels]
        self.build_prm_graph()


    def build_prm_graph(self):
        # sampling
        self.graph = {}
        self.vertices = []
        attempts = 0
        while len(self.vertices) < self.max_itr and attempts < self.max_itr * 10:
            sample = self.sample()
            if not self.is_in_collision(sample):
                self.vertices.append(sample)
                self.graph[sample] = []
            attempts += 1
        # wiring neighbors
        for v in self.vertices:
            neighbors = self.find_neighbors_in_range(v)
            for nei, dist in neighbors:
                if self.local_planner(v, nei, dist):
                    self.graph[v].append((nei, dist))

    def sample(self):
        x = np.random.randint(0, self.env_cols)
        y = np.random.randint(0, self.env_rows)
        return (int(x), int(y))  # Always return a tuple of ints

    def is_in_collision(self, config):
        x, y = config
        if x < 0 or x >= self.env_cols or y < 0 or y >= self.env_rows:
            return True
        return self.map[y, x] != 0

    def find_neighbors_in_range(self, vertex1):
        distances = []
        for v in self.vertices:
            if v == vertex1:
                continue
            dist = np.linalg.norm(np.array(vertex1) - np.array(v))
            if dist <= self.max_dist:
                distances.append((v, dist))
        return distances

    def local_planner(self, config1, config2, dist):
        # Simple straight-line check
        num_steps = int(dist) #*100
        x1, y1 = config1
        x2, y2 = config2
        for i in range(1, num_steps):
            t = i / num_steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            if self.is_in_collision((x, y)):
                return False
        return True

class A_Star():
    def __init__(self, prm: PRM):
        self.prm = prm

    def h(self, current, goal):
        return np.linalg.norm(np.array(current) - np.array(goal))


    def find_path(self, start, goal):
        # Add start and goal to the graph if not present
        if start not in self.prm.graph:
            self.prm.graph[start] = []
            for nei, dist in self.prm.find_neighbors_in_range(start):
                if self.prm.local_planner(start, nei, dist):
                    self.prm.graph[start].append((nei, dist))
                    self.prm.graph[nei].append((start, dist))
        if goal not in self.prm.graph:
            self.prm.graph[goal] = []
            for nei, dist in self.prm.find_neighbors_in_range(goal):
                if self.prm.local_planner(goal, nei, dist):
                    self.prm.graph[goal].append((nei, dist))
                    self.prm.graph[nei].append((goal, dist))
        
        open_set = []
        heapq.heappush(open_set, (0 + self.h(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_set:
            _, curr_g, current = heapq.heappop(open_set)
            if current == goal:
                path = self.reconstruct_path(current, came_from, start)
                return path, g_score[goal]
            closed_set.add(current)
            for neighbor, cost in self.prm.graph.get(current, []):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.h(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        return None, float('inf')

    def reconstruct_path(self, current, came_from, start):
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

        

class Plotter():
    def __init__(self, inflated_map): 
        self.env_rows, self.env_cols = inflated_map.shape
        self.map = inflated_map
    
    def draw_graph(self, graph, start, goal,path=None):
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        plt.xlim([0, self.env_cols])
        plt.ylim([0, self.env_rows])
        for vertex in graph.keys():
            plt.scatter(vertex[0], vertex[1], s=20, c='g')
        for vertex in graph.keys():
            for nei, dist in graph[vertex]:
                plt.plot([vertex[0],nei[0]], [vertex[1], nei[1]], color='b', linewidth=1)
        if path is not None:
            for i in range(len(path)-1):
                plt.plot([path[i][0],path[i+1][0]], [path[i][1],path[i+1][1]], color='r', linewidth=3)
        plt.scatter(start[0], start[1], s=100, c='g')
        plt.scatter(goal[0],goal[1], s=100, c='r')
        plt.imshow(self.map, origin="lower")
        plt.pause(100)
    


def inflate(map_, inflation):#, resolution, distance):
    cells_as_obstacle = int(inflation) #int(distance/resolution)
    map_[95:130, 70] = 100
    original_map = map_.copy()
    inflated_map = map_.copy()
    # add berrier
    rows, cols = inflated_map.shape
    for j in range(cols):
        for i in range(rows):
            if original_map[i,j] != 0:
                i_min = max(0, i-cells_as_obstacle)
                i_max = min(rows, i+cells_as_obstacle)
                j_min = max(0, j-cells_as_obstacle)
                j_max = min(cols, j+cells_as_obstacle)
                inflated_map[i_min:i_max, j_min:j_max] = 100
    return inflated_map       



def main():
    start_time = time.time()
    map_original = np.array(np.load('maze_test.npy'), dtype=int)
    resolution=0.05000000074505806
    robot_raduis = 0.15
    converter = CSpace(resolution, origin_x=-4.73, origin_y=-5.66, map_shape=map_original.shape )
    map_original = np.array(np.load('maze_test.npy'), dtype=int)
    inflated_map = inflate(map_original, robot_raduis /resolution)
    prm = PRM(env_map=inflated_map,  max_itr=1000, dist = 30)
    astar = A_Star(prm)
    start_full = tuple(converter.meter2pixel([0.0,0.0]))
    goal_full = tuple(converter.meter2pixel([-2, 0]))
    # Only use (x, y)
    start = (start_full[0], start_full[1])
    goal = (goal_full[0], goal_full[1])
    print(start)
    print(goal)
    path, cost = astar.find_path(start, goal)
    print(f'path cost: {cost}, time: {time.time()-start_time}')
    plotter = Plotter(inflated_map)
    #print('path:',path)
    plotter.draw_graph(prm.graph, start, goal,path)


if __name__ == "__main__":
    main()


