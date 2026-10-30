from nav_msgs.msg import OccupancyGrid as ROSOccupancyGrid
import numpy as np
import math

#units: meters, radians, log-odds

LOG_ODDS_LIMIT = 5.0 #clamp so cant becoem too fixed

def to_absolute(points, pose): #convert points from lidar frame to world frame
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([pose[0], pose[1]])


def bayesian_update(prev_odds, mea_odds, prior_odds): #bayesian update log-odds
    return prev_odds + mea_odds - prior_odds #addition since log turns multiplication into addition

def log_odds(prob): #probability to log-odds
    return math.log(prob / (1 - prob))

def recover_probability(log_odd): #log-odds back to probability
    return 1 / (1 + math.exp(-log_odd))

def bresenham(x0, y0, x1, y1, cell_size=1): #bresenham algo for all cells between two points, used to find empty cells between lidar origin and hit point
    x0 = int(math.floor(x0 / cell_size)) #world coords into grid cell coords
    y0 = int(math.floor(y0 / cell_size))
    x1 = int(math.floor(x1 / cell_size))
    y1 = int(math.floor(y1 / cell_size))

    cells = [] #output list of cells crossed by the line

    dx = abs(x1 - x0) #horiz distance between start and end cells
    dy = abs(y1 - y0) #vert distance between start and end cells

    sx = 1 if x0 < x1 else -1 #direction to move in x (+1 or -1)
    sy = 1 if y0 < y1 else -1 #direction to move in y (+1 or -1)

    err = dx - dy #error term to decide when to step in each direction

    while True:
        cells.append((x0, y0)) #add current cell to the line path

        if x0 == x1 and y0 == y1: #stop once the end cell is reached
            break

        e2 = 2 * err #double error to avoid repeated calculations

        if e2 > -dy: #if error is large enough, move in x direction
            err -= dy #update error
            x0 += sx #move to next x cell

        if e2 < dx: #if error is large enough, move in y direction
            err += dx #update error
            y0 += sy #move to next y cell

    return cells #return all cells crossed by the line

class OccupancyGrid: #class for occupancy grid repr
    def __init__(self, width, height, resolution, prior_odds, poh, pom, origin_x=None, origin_y=None):
        self.width = width #set parameters
        self.height = height
        self.resolution = resolution
        self.origin_x = -width * resolution / 2.0 if origin_x is None else float(origin_x)
        self.origin_y = -height * resolution / 2.0 if origin_y is None else float(origin_y)

        self.prior_odds = log_odds(prior_odds)
        self.prob_occ_hit = log_odds(poh)
        self.prob_occ_miss = log_odds(pom)
        self.grid = np.full((width, height), self.prior_odds, dtype=np.float64) #initialize grid with prior odds

    #ros conversion cuz ros has diff standards (yay)
    def to_ros_occupancy_grid(self, frame_id="map", origin_x=None, origin_y=None, stamp=None,
                              unknown_threshold=0.1): #minimum probability difference from prior to be considered known
        msg = ROSOccupancyGrid() #create occupancy grid message
        msg.header.frame_id = frame_id #header
        if stamp is not None:
            msg.header.stamp = stamp #timestamp

        msg.info.resolution = float(self.resolution) #metadata
        msg.info.width = int(self.width)
        msg.info.height = int(self.height)

        msg.info.origin.position.x = float(self.origin_x if origin_x is None else origin_x) #origin
        msg.info.origin.position.y = float(self.origin_y if origin_y is None else origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0 #world frame, no rotation
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0 #orientation represented as quaternion

        probabilities = 1.0 / (1.0 + np.exp(-self.grid)) #convert log-odds to probabilities using numpy vectorization
        occupancy = np.rint(probabilities * 100.0).astype(np.int8) #convert probabilities from 0-1 to 0-100 for ROS

        prior_probability = recover_probability(self.prior_odds) #prior probability
        unknown = np.abs(probabilities - prior_probability) < unknown_threshold #cells where probability has not changed significantly
        occupancy[unknown] = -1 #set unknown cells to -1

        msg.data = occupancy.T.flatten().tolist() #flatten in ROS row-major order: index = y * width + x
        return msg

    def _in_bounds(self, c): #chelper
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def update_grid(self, point_cloud, pose, beam_stride=1): #update grid using lidar pointss
        pts = np.asarray(point_cloud, dtype=float).reshape(-1, 2)[::max(1, int(beam_stride))]
        if len(pts) == 0:
            return

        abs_points = to_absolute(pts, pose) #get points in world frame
        x0 = pose[0] - self.origin_x #convert lidar origin to map coords
        y0 = pose[1] - self.origin_y

        empty = set()
        hit = set()
        for px, py in abs_points: #get cells between lidar origin and each hit point
            cells = bresenham(x0, y0, px - self.origin_x, py - self.origin_y, self.resolution)
            for c in cells[:-1]: #all cells before the last one are empty
                if self._in_bounds(c):
                    empty.add(c)
            if self._in_bounds(cells[-1]): #last cell is the lidar hit
                hit.add(cells[-1])

        #update each cell at most once per scan; if any beam ends in a cell, count as a hit
        empty -= hit

        for x, y in empty: #update empty cells using the probability of occupancy given a miss
            self.grid[x, y] = bayesian_update(self.grid[x, y], self.prob_occ_miss, self.prior_odds)
        for x, y in hit: #update hit cells using the probability of occupancy given a hit
            self.grid[x, y] = bayesian_update(self.grid[x, y], self.prob_occ_hit, self.prior_odds)

        np.clip(self.grid, -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT, out=self.grid) #clamp log-odds to prevent cells from becoming too certain

    def world_to_grid(self, x, y): #convert world coords to grid coords
        return (int(math.floor((x - self.origin_x) / self.resolution)),
                int(math.floor((y - self.origin_y) / self.resolution)))

    def grid_to_world(self, x, y): #convert grid coords to the world coords at the cell center
        return (self.origin_x + (x + 0.5) * self.resolution,
                self.origin_y + (y + 0.5) * self.resolution)

    def get_odds(self, x, y): #get probability of occupancy at world coord
        c = self.world_to_grid(x, y)
        if not self._in_bounds(c): #return 0.5 if outside the map
            return 0.5
        return recover_probability(self.grid[c[0], c[1]])