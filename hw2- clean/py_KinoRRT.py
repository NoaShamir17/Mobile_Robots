import numpy as np
import matplotlib.pyplot as plt
from py_Utils import Tree,  CSpace
plt.ion()

class KINORRT(object):
    def __init__(self, env_map, max_step_size = 0.5, max_itr=5000, p_bias = 0.05, converter: CSpace =None ):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.p_bias = p_bias
        self.tree = Tree()
        self.map = env_map
        self.env_rows, self.env_cols = env_map.shape
        self.env_yaw_range = 2*np.pi
        self.converter = converter
        self.ackerman = Odom(converter)
        
        

    def find_path(self, start, goal):
        itr = 0
        self.tree.AddVertex(start)

        while itr < self.max_itr:
            
            # sample random vertex
            x_random = self.sample(goal)

            # find nearest neighbor
            x_near_idx, x_near = self.tree.GetNearestVertex(x_random)
            
            # sample random control command
            delta_time, steering, velocity = self.ackerman.sample_control_command()
            
            # propagate
            x_new, edge, edge_cost = self.ackerman.propagate(steering, velocity ,delta_time, x_near)
            
            # add vertex and edge
            if self.local_planner(edge):
                # TODO
                if self.tree.isConfExists(x_new):
                    #can implement path improvement here
                    continue
                eid = self.tree.AddVertex(x_new)
                self.tree.vertices[eid].set_waypoints(edge)
                self.tree.AddEdge(x_near_idx, eid, edge_cost)

                #end condition, not necessarily here, could be after max iterations, and path improvement should be added
                if np.linalg.norm(np.array(x_new[:2]) - np.array(goal[:2])) < self.max_step_size and abs(np.rad2deg(x_new[2] - goal[2]) % 360) < 15:
                    path, path_idx, cost = self.get_shortest_path(eid)
                    print(f'found path with cost: {cost}')
                    return path, path_idx, cost

            itr += 1
            if itr%1000 ==0:
                print(f'itr: {itr}')
        return None, None, None
    
    def sample(self, goal):
        # TODO
        return goal if np.random.rand() < self.p_bias else np.random.uniform(0, 1, 2) * np.array([self.env_cols, self.env_rows])  # sample random point in the map

    
    def is_in_collision(self, x_new):
        # TODO
        if self.map[int(x_new[1]), int(x_new[0])] != 0: #maybe switch x and y
            return True
        

    
    def local_planner(self, edge):
        #TODO
        for x in edge:
            if self.is_in_collision(x):
                return False
        return True
    
    def get_shortest_path(self, goal_idx):
        '''
        Returns the path and cost from some vertex to Tree's root
        @param dest - the id of some vertex
        return the shortest path and the cost
        '''
        # TODO
        path = [self.tree.vertices[goal_idx].conf]
        path_idx = [goal_idx]
        cost = self.tree.vertices[goal_idx].cost
        while goal_idx != self.tree.GetRootID():
            goal_idx = self.tree.edges[goal_idx]
            path_idx.append(goal_idx)
            path.append(self.tree.vertices[goal_idx].conf)
        return path, path_idx , cost
    
    

class Plotter():
    def __init__(self, inflated_map): 
        self.env_rows, self.env_cols = inflated_map.shape
        self.map = inflated_map
    
    def draw_tree(self, tree:Tree, start, goal, path=None, path_idx = None):
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.xlim([0, self.env_cols])
        plt.ylim([0, self.env_rows])
        if path is not None:
            for idx in path_idx:
                try:
                    vertex = tree.vertices[idx]
                    for waypoint in vertex.waypoints:
                        plt.scatter(waypoint[0], waypoint[1], s=20, c='m')
                except:
                    pass

        for i in range(len(tree.vertices)):
            conf = tree.vertices[i].conf
            plt.scatter(conf[0], conf[1], s=10, c='b')

        
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
    

class Odom(object):
    def __init__(self, converter:CSpace):
        self.wheelbase = 0.35
        self.max_steering_angle = np.deg2rad(35)
        self.min_velocity, self.max_velocity = 0.5, 1
        self.min_time, self.max_time = 1, 2
        self.converter = converter
    
    def sample_control_command(self):
        # TODO
        delta_time = np.random.uniform(self.min_time, self.max_time)
        steering = np.random.uniform(-self.max_steering_angle, self.max_steering_angle)
        velocity = np.random.uniform(self.min_velocity, self.max_velocity)
        return delta_time, steering, velocity

    def propagate(self,  steering, velocity ,delta_time, initial_x):
        initial_x = self.converter.pixel2meter(initial_x)
        x = initial_x[0]
        y = initial_x[1]
        theta= initial_x[2]
        theta_dot = velocity * np.tan(steering) / self.wheelbase
        dt = 0.03
        edge = [[x,y,theta]]
        cost = 0
        for _ in range(int(delta_time/dt)):
            theta += theta_dot * dt
            x_dot = velocity * np.cos(theta)
            y_dot = velocity * np.sin(theta)
            x += x_dot * dt
            y += y_dot * dt
            cost += ((edge[-1][0] - x)**2 + (edge[-1][1] - y)**2)**0.5
            edge.append([x,y,theta])
        edge = self.converter.pathmeter2pathindex(edge)
        new_state = edge[-1]
        return new_state, edge, cost




def main():
    map_original = np.array(np.load(r'C:\Users\yahli\Documents\Mobile_Robots\hw2- clean\maze_test.npy'), dtype=int)
    resolution=0.05000000074505806
    inflated_map = inflate(map_original, 0.2/resolution)
    converter = CSpace(resolution, origin_x=-4.73, origin_y=-5.66, map_shape=map_original.shape)
    start=converter.meter2pixel([0.0,0.0])
    goal = converter.meter2pixel([6.22, -4.22])
    print(start)
    print(goal)
    kinorrt_planner = KINORRT(env_map=inflated_map, max_step_size=20, max_itr=10000, p_bias=0.05,converter=converter )
    path, path_idx, cost = kinorrt_planner.find_path(start, goal)
    print(f'cost: {cost}')
    plotter = Plotter(inflated_map=inflated_map)
    plotter.draw_tree(kinorrt_planner.tree, start, goal, path, path_idx)
    


if __name__ == "__main__":
    main()


