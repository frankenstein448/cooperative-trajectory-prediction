import numpy as np
 
# Fitted via coarse-to-fine grid search against dense CARLA Town05 waypoints
# (median residual 0.267m, scene 77, 500 agent points). See grid_search_transform.py.
THETA_DEG = 70.5
TX = -39.00
TY = 4.75
 
_THETA_RAD = np.radians(THETA_DEG)
_C, _S = np.cos(_THETA_RAD), np.sin(_THETA_RAD)
_R = np.array([[_C, -_S], [_S, _C]])
_T = np.array([TX, TY])
 
 
def v2xsim_to_carla(x, y):
    """
    Convert a single V2X-Sim ego_pose (x, y) world position into CARLA
    Town05 world-frame coordinates.
    """
    p = np.array([x, y])
    return tuple(p @ _R.T + _T)
 
 
def v2xsim_to_carla_batch(points):
    """
    Convert an (N, 2) array of V2X-Sim (x, y) positions into CARLA
    Town05 world-frame coordinates. Returns an (N, 2) array.
    """
    points = np.asarray(points, dtype=np.float64)
    return points @ _R.T + _T
 
 
if __name__ == "__main__":
    # Sanity check against the verified agent 1 / scene 77 / frame 23 point
    x, y = v2xsim_to_carla(87.72, 105.93)
    print(f"Transformed: ({x:.2f}, {y:.2f})")
