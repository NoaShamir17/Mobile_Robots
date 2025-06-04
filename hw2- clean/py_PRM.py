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
        # TODO
        # wiring neighbors
        # TODO
        

    def find_neighbors_in_range(self, vertex1):
        distances = [] # tuples of [(nei, dist)]
        pass
        # TODO
        return distances # tuples of [(nei, dist)]
    
    def get_cost(self, vertex1, vertex2):
        pass
    
    def sample(self):
        pass # cols~x, rows~y, 
    

    def is_in_collision(self, config):
        pass

    
    def local_planner(self, config1, config2, dist):
        pass

class A_Star():
    def __init__(self, prm: PRM):
        self.prm = prm

    def h(self, current, goal):
        pass


    def find_path(self, start, goal):
        start = (start[0], start[1])
        goal = (goal[0], goal[1])
        self.prm.graph[start] = self.prm.find_neighbors_in_range(start)
        for nei, dist in self.prm.graph[start]:
            self.prm.graph[nei].append((start, dist))
        self.prm.graph[goal] = self.prm.find_neighbors_in_range(goal)
        #TODO


    def reconstruct_path(self, current, came_from, start):
        path = []
        # TODO
        pass

        

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
    start=converter.meter2pixel([0.0,0.0])
    goal = converter.meter2pixel([-2, 0])
    print(start)
    print(goal)
    path, cost = astar.find_path(start, goal)
    print(f'path cost: {cost}, time: {time.time()-start_time}')
    plotter = Plotter(inflated_map)
    plotter.draw_graph(prm.graph, start, goal,path)


if __name__ == "__main__":
    main()


