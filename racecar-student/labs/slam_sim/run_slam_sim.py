import sys
import os

sys.path.insert(1, '../../library')
import racecar_core

from slam_setup import build_nodes #install and import
from sim_bridge import RacecarBridgeNode
from map_plot import save_map_plot

GOAL_XY = (0.7, -6) #goal from start frame, in m. plus x forwards, plus y to the left
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_plots") #thingy

rc = racecar_core.create_racecar()

bridge = None
se_node = None
controller = None
plot_count = 0
trail = []

def start():
    global bridge, se_node, controller, plot_count, trail
    rc.drive.stop()
    se_node, controller = build_nodes(GOAL_XY) #reset fake ros bus + clock
    bridge = RacecarBridgeNode(rc)
    bridge.set_max_speed(0.5)
    plot_count = 0
    trail = []
    rc.set_update_slow_time(1.0)

def update():
    bridge.step()
    pose = se_node.get_state_estimate()
    if not trail or (pose.x - trail[-1][0]) ** 2 + (pose.y - trail[-1][1]) ** 2 > 0.05 ** 2: #dist > 5cm
        trail.append((pose.x, pose.y)) #store

def update_slow():
    global plot_count
    pose = se_node.get_state_estimate()
    print(f"est pose x={pose.x:6.2f} y={pose.y:6.2f} theta={pose.theta:6.2f} rad | "
          f"cmd v={bridge.cmd_speed:4.2f} m/s steer={bridge.cmd_steer:5.2f} rad")
 
    file = os.path.join(PLOT_DIR, f"map_{plot_count:04d}.png")
    save_map_plot(se_node.get_map_estimate(), pose, GOAL_XY, trail, file)
    plot_count += 1

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()