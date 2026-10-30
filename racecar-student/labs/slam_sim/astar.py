import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from nav_msgs.msg import OccupancyGrid # comment out if unit test

class AStarPlanner:
    """
    A* path planner for ROS 2 nav_msgs/msg/OccupancyGrid messages.

    The planner expects occupancy values in the standard ROS format:

        -1      = unknown
         0      = free
        1..100  = occupancy probability

    By default:
        - cells with occupancy >= 60 are treated as obstacles
        - unknown cells are treated as obstacles
    """

    def __init__(self, occupancy_grid, obstacle_threshold=60, allow_unknown=False,
                 inflation_radius=0.0): #direclty read from ros2 occupancy grid message
        self.map = occupancy_grid

        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution

        self.origin_x = occupancy_grid.info.origin.position.x
        self.origin_y = occupancy_grid.info.origin.position.y

        self.obstacle_threshold = obstacle_threshold
        self.allow_unknown = allow_unknown
        self.inflation_radius = inflation_radius

        # ROS OccupancyGrid stores data in row-major order:
        #
        # index = y * width + x
        #
        self.data = np.asarray(occupancy_grid.data, dtype=np.int16).reshape(self.height, self.width)

        unknown = self.data == -1
        obstacle = self.data >= obstacle_threshold
        blocked = obstacle.copy()
        if inflation_radius > 0 and obstacle.any():
            dist = distance_transform_edt(~obstacle) * self.resolution # distance to nearest
            blocked |= dist < inflation_radius
        self._blocked = blocked
        traversable = ~blocked
        if not allow_unknown:
            traversable &= ~unknown
        else:
            traversable |= unknown & ~blocked
        self._traversable = traversable
        self._trav = None 

    @staticmethod
    def heuristic(a, b):
        """
        Octile-distance heuristic for an 8-connected grid.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to integer grid coordinates.

        This implementation assumes the map origin has no rotation.
        """
        return (math.floor((x - self.origin_x) / self.resolution),
                math.floor((y - self.origin_y) / self.resolution))

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert a grid cell to the world coordinates of its center.
        """
        return (self.origin_x + (grid_x + 0.5) * self.resolution,
                self.origin_y + (grid_y + 0.5) * self.resolution)

    def is_in_bounds(self, x, y):
        """
        Check whether a grid coordinate is inside the map.
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def get_occupancy(self, x, y):
        """
        Return the occupancy value for a grid cell.

        ROS OccupancyGrid data is stored as:

            data[y * width + x]
        """
        return int(self.data[y, x])

    def is_traversable(self, x, y):
        """
        Determine whether a cell can be used by the planner.
        """
        if not self.is_in_bounds(x, y):
            return False
        return bool((self._trav if self._trav is not None else self._traversable)[y][x])

    def get_neighbors(self, current):
        """
        Return valid neighboring cells and their movement costs.

        Diagonal movement is allowed. Diagonal corner-cutting is prevented,
        so the robot cannot move diagonally between two obstacles.
        """
        x, y = current
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = x + dx, y + dy
            if not self.is_traversable(nx, ny):
                continue

            # Prevent diagonal corner-cutting.
            if dx != 0 and dy != 0:
                if not self.is_traversable(x + dx, y) or not self.is_traversable(x, y + dy):
                    continue

            yield (nx, ny), math.hypot(dx, dy)

    def _clear_around(self, cell, radius_m):
        """
        Clear area around cell, prevent path planning from start from failing due to occupancy.
        """
        r = int(math.ceil(radius_m / self.resolution))
        cx, cy = cell
        y0, y1 = max(0, cy - r), min(self.height, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.width, cx + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        near = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        free_enough = self.data[y0:y1, x0:x1] < self.obstacle_threshold # keep real obstacles blocked
        sub = self._traversable[y0:y1, x0:x1]
        sub |= near & free_enough & ((self.data[y0:y1, x0:x1] != -1) | self.allow_unknown)

    def plan(self, start_world, goal_world): #gives path from start to goal in world coords
        start = self.world_to_grid(*start_world[:2]) #shorter get in grid
        goal = self.world_to_grid(*goal_world[:2])#shorter get in grid

        if not self.is_in_bounds(*start): #bounds check
            raise ValueError(
                f"Start position {start_world} is outside the map."
            )
        
        if not self.is_in_bounds(*goal): #bounds check
            raise ValueError(
                f"Goal position {goal_world} is outside the map."
            )

        if self.inflation_radius > 0:
            self._clear_around(start, self.inflation_radius)
        self._trav = self._traversable.tolist() #use list for faster index

        if not self.is_traversable(*start): #possible check
            raise ValueError(
                f"Start cell {start} is occupied or unknown."
            )
        
        if not self.is_traversable(*goal): #possible check
            raise ValueError(
                f"Goal cell {goal} is occupied or unknown."
            )

        open_set = []

        # counter avoids comparing grid-coordinate tuples when two heap entries have the same priority.
        counter = 0
        heapq.heappush(open_set, (self.heuristic(start, goal), counter, start))

        came_from = {} #path reconstruction dict (traceback)
        g_score = {start: 0.0} #cost from start to current, start->self is obviously 0
        closed = set() #visited

        while open_set: #nodes to explore
            
            _, _, current = heapq.heappop(open_set) #get node with lowest fscore (cost to neighbors + heuristic)
            if current in closed: # if visited, skip
                continue

            if current == goal: #end
                return self._reconstruct_path(came_from, current) #reconstruct, then return list

            closed.add(current) #visited update

            for neighbor, step_cost in self.get_neighbors(current): #check neighbors and their costs
                if neighbor in closed: # visited, skip
                    continue

                tentative_g = g_score[current] + step_cost #cost start->current + current->neighbor

                if tentative_g >= g_score.get(neighbor, float("inf")): #no clue why og code expanded everything
                    continue

                came_from[neighbor] = current #add to dict
                g_score[neighbor] = tentative_g #update g_score

                counter += 1 #increment to avoid comparing same-priority nodes
                heapq.heappush(
                    open_set,
                    (
                        tentative_g + self.heuristic(neighbor, goal), 
                        counter, 
                        neighbor
                    ),
                )

        return None

    def _reconstruct_path(self, came_from, current): #just traverse backwards from goal to start
        grid_path = [current]

        while current in came_from: #while can go back
            current = came_from[current] #which did i come from?
            grid_path.append(current) #we add that to the path

        grid_path.reverse() #reverse so start->goal instead of goal->start

        return [
            self.grid_to_world(x, y) #world coords
            for x, y in grid_path
        ]