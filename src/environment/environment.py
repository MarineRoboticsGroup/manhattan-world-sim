from copy import deepcopy

import numpy as np
from typing import Tuple, List, Union

from src.agent.agent import Agent, RobotAgent, BeaconAgent
from src.geometry.TwoDimension import SE2Pose, Point2


def _find_nearest(
    array: Union[np.ndarray, List[float]], value: float
) -> Tuple[int, float, float]:
    """Finds the nearest value in the array to the given value. Returns the
    index, difference, and value of the nearest value in the array.

    Args:
        array (Union[np.ndarray, List[float]]): the array to check the nearest
            value of
        value (float): the value to check for the nearest value in the array

    Returns:
        Tuple[int, float, float]: index of the nearest value, difference between
            values, and value of the nearest value
    """
    assert isinstance(array, np.ndarray) or isinstance(array, list)
    assert len(array) > 0
    assert isinstance(value, float)

    array = np.asarray(array)
    distances = np.abs(array - value)
    idx = distances.argmin()
    delta = value - array[idx]
    return idx, delta, array[idx]

class ManhattanWorld:
    """
    This class creates a simulated environment of Manhattan world with landmarks.
    """

    def __init__(
        self,
        grid_vertices_shape: tuple = (9, 9),
        cell_scale: float = 1.0,
        robot_area: List[Tuple] = None,
        check_collision: bool = True,
        tol: float = 1e-5,
    ):
        """Constructor for Manhattan waterworld environment. Note that the
        landmarks are only allowed in areas that is infeasible to the robot. As
        of now the robot feasible area is only rectangular

        Args:
            grid_vertices_shape (tuple, optional): a tuple defining the shape of grid vertices; note that the vertices follow ij indexing. Defaults to (9, 9).
            cell_scale (int, optional): width and length of a cell. Defaults to 1.
            robot_area (List[Tuple], optional): bottom left and top right vertices of a rectangular area; all the rest area will be infeasible. Defaults to None.
            check_collision (bool, optional): [description]. Defaults to True.
            tol (float, optional): [description]. Defaults to 1e-5.
        """
        assert isinstance(grid_vertices_shape, tuple)
        assert len(grid_vertices_shape) == 2
        self._x_pts, self._y_pts = grid_vertices_shape

        assert isinstance(cell_scale, float)
        self._scale = cell_scale

        assert isinstance(check_collision, bool)
        self._check_collision = check_collision

        assert isinstance(tol, float)
        self._tol = tol

        assert isinstance(robot_area, list) or robot_area is None
        if robot_area is not None:
            assert self.check_vertex_list_valid(robot_area)

        # create grid
        self._grid = np.zeros(grid_vertices_shape, dtype=np.float32)

        # define the grid over which the robot can move
        self._x_coords = np.arange(self._x_pts) * self._scale
        self._y_coords = np.arange(self._y_pts) * self._scale
        self._xv, self._yv = np.meshgrid(self._x_coords, self._y_coords, indexing="ij")

        # agents are added by vertices but stored with groundtruth poses or points
        self._robot_poses = {}
        self._landmark_points = {}

        if robot_area is not None:
            # ensure a rectangular feasible area for robot
            bl, tr = robot_area
            self._robot_feasibility = np.zeros((self._x_pts, self._y_pts), dtype=bool)
            self._robot_feasibility[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
        else:
            # if no area specified, all area is now feasible
            self._robot_feasibility = np.ones((self._x_pts, self._y_pts), dtype=bool)

    def set_robot_area_feasibility(self, area: List[Tuple[int, int]]):
        """Sets the feasibility status for the robots as a rectangular area. Anything
        outside of this area will be the inverse of the status.

        Args:
            area (List[Tuple[int, int]]): the feasibility area for robots, denoted by the
                bottom left and top right vertices.
        """
        assert self.check_vertex_list_valid(area)
        assert len(area) == 2

        mask = np.zeros((self._x_pts, self._y_pts), dtype=bool)
        bl, tr = area
        mask[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
        self._robot_feasibility[mask] = True
        self._robot_feasibility[np.invert(mask)] = False

    def get_neighboring_vertices(self, vert: Tuple[int, int]) -> List[tuple]:
        """gets all neighboring vertices to the vertex at index (i, j). Only
        returns valid indices (not out of bounds)

        Args:
            vert (tuple): a vertex index (i, j)

        Returns:
            List[tuple]: list of all neighboring vertices
        """
        assert self.check_vertex_valid(vert)
        i, j = vert
        candidate_vertices = [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]
        vertices_in_bound = [
            v for v in candidate_vertices if self.vertex_is_in_bounds(v)
        ]
        return vertices_in_bound

    def get_neighboring_robot_vertices(
        self, vert: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """get all neighboring vertices to the vertex at index (i, j) that are
        feasible for the robot. Only returns valid indices (not out of bounds)

        Args:
            i (int): [description]
            j (int): [description]
            feasibility ([type], optional): [description]. Defaults to None.

        Returns:
            List[Tuple[int, int]]: the list of neighboring vertices that are
                feasible for the robot
        """
        assert self.check_vertex_valid(vert)

        neighbor_verts = self.get_neighboring_vertices(vert)
        assert self.check_vertex_list_valid(neighbor_verts)

        feasible_neighbor_verts = self.get_robot_feasible_vertices(
            neighbor_verts,
        )
        assert self.check_vertex_list_valid(feasible_neighbor_verts)

        return feasible_neighbor_verts

    def get_robot_feasible_vertices(
        self,
        vertices: List[tuple],
    ) -> List[Tuple[int, int]]:
        """Picks out vertices from a list that are in the robot feasible area.

        Args:
            vertices (List[tuple]): list of vertices

        Returns:
            List[tuple]: list of robot feasible vertices from the input list
        """
        assert self.check_vertex_list_valid(vertices)

        feasible_verts = [v for v in vertices if self.vertex_is_robot_feasible(v)]
        return feasible_verts

    # TODO look over this function
    def nearest_robot_vertex_coordinates(self, x: float, y: float) -> List[tuple]:
        """Return the nearest coordinates of a vertex to the given coordinates

        Args:
            x (float): [description]
            y (float): [description]

        Raises:
            ValueError: [description]

        Returns:
            List[tuple]: [description]
        """
        i, dx, x_close = _find_nearest(self._x_coords, x)
        j, dy, y_close = _find_nearest(self._y_coords, y)
        if abs(dx) < self._tol and abs(dy) < self._tol:
            goal_vertices = self.get_neighboring_vertices(i, j)
        elif abs(dy) < self._tol:
            if dx > 0:
                goal_vertices = [(i + 1, j), (i, j)]
            else:
                goal_vertices = [(i, j), (i - 1, j)]
        elif abs(dx) < self._tol:
            if dy > 0:
                goal_vertices = [(i, j + 1), (i, j)]
            else:
                goal_vertices = [(i, j), (i, j - 1)]
        else:
            raise ValueError(
                "The robot with location (" + str(x),
                ", " + str(y) + ") falls off the grid.",
            )
        goal_vertices = self.get_robot_feasible_vertices(goal_vertices)
        return [self.vertex2coordinate(*vertex) for vertex in goal_vertices]

    def coordinate2vertex(self, x: float, y: float) -> Tuple[int, int]:
        """Takes a coordinate and returns the corresponding vertex. Requires the
        coordinate correspond to a valid vertex.

        Args:
            x (float): x-coordinate
            y (float): y-coordinate

        Raises:
            ValueError: the coordinate does not correspond to a valid vertex

        Returns:
            Tuple[int, int]: the corresponding vertex indices
        """
        i, dx, x_close = _find_nearest(self._x_coords, x)
        j, dy, y_close = _find_nearest(self._y_coords, y)
        if abs(dx) < self._tol and abs(dy) < self._tol:
            return (i, j)
        else:
            raise ValueError(
                "The input (" + str(x) + ", " + str(y) + ") is off grid vertices."
            )

    def coordinates2vertices(
        self, coords: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Takes in a list of coordinates and returns a list of the respective
        corresponding vertices

        Args:
            coords (List[Tuple[int, int]]): list of coordinates

        Returns:
            List[Tuple[int, int]]: list of vertices
        """
        assert isinstance(coords, list)
        assert len(coords) >= 1
        assert isinstance(coords[0], tuple)
        assert all(len(c) == 2 for c in coords)

        nearest_vertices = [self.coordinate2vertex(*c) for c in coords]
        assert self.check_vertex_list_valid(nearest_vertices)
        return nearest_vertices

    # TODO what is going on here
    def agent_xy(self, agent2gt):
        """Returns the x, y coordinates of every agent in the environment.

        Args:
            agent2gt ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.array([[pt.x, pt.y] for key, pt in agent2gt.items()])

    def no_collision(self, x, y, agent2gt):
        """Checks if agent is already at that x, y coordinate

        Args:
            x ([type]): [description]
            y ([type]): [description]
            agent2gt ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self._check_collision:
            gt_xy = self.agent_xy(agent2gt)
            if gt_xy.shape[0] > 0:
                xy = np.array([x, y])
                min_dist = min(np.linalg.norm(gt_xy - xy, axis=1))
                if min_dist > self._tol:
                    return True
                else:
                    print(
                        "Collision: minimum distance to existing agents is "
                        + str(min_dist)
                    )
                    return False
            else:
                return True
        else:
            return True

    def add_landmark(self, beacon: BeaconAgent, vert: Tuple[int, int]):
        """Adds landmark to the environment.

        Args:
            beacon (BeaconAgent): [description]
            vert (Tuple[int, int]): [description]

        Returns:
            [type]: [description]
        """
        # TODO replace beacon with BeaconAgent functionality
        assert isinstance(beacon, BeaconAgent)
        assert self.check_vertex_valid(vert)

        i, j = vert
        assert isinstance(i, int) and isinstance(j, int)

        if self._landmark_feasibility[i, j] and beacon not in self._landmark_points:
            x, y = self.vertex2coordinate(i, j)
            if not self._landmark_points or self.no_collision(
                x, y, self._landmark_points
            ):
                self._landmark_points[beacon] = Point2(x, y)
                return True
            else:
                print("Add abort: landmark collision found.")
                return False
        elif beacon in self._beacon2point:
            print("Add abort: duplicated landmark.")
            return False
        else:
            print(
                "Add abort: vertex ("
                + str(i)
                + ", "
                + str(j)
                + ") is infeasible for adding landmarks."
            )
            return False

    def add_robot(self, robot: RobotAgent, vert: Tuple[int, int], orientation=0.0):
        """Adds robot to the environment.

        Args:
            robot (RobotAgent): [description]
            vert (Tuple[int, int]): [description]
            orientation (float, optional): [description]. Defaults to 0.0.

        Returns:
            [type]: [description]
        """
        # TODO replace robot functionality with RobotAgent functionality
        assert isinstance(robot, RobotAgent)
        assert self.check_vertex_valid(vert)
        assert isinstance(orientation, float)

        i, j = vert

        # TODO what are the conditions for adding a robot?
        if self._robot_feasibility[i, j] and robot not in self._robot_poses:
            x, y = self.vertex2coordinate(i, j)
            if not self._robot_poses or self.no_collision(x, y, self._robot_poses):
                self._robot_poses[robot] = SE2Pose(x, y, orientation)
                return True
            else:
                print("Add abort: robot collision found.")
                return False
        elif robot in self._robot_poses:
            print("Add abort: duplicated robot.")
            return False
        else:
            print(
                "Add abort: vertex ("
                + str(i)
                + ", "
                + str(j)
                + ") is infeasible for adding robots."
            )
            return False

    def remove_robot(self, robot: RobotAgent):
        """Removes robot from the environment.

        Args:
            agent ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert isinstance(robot, RobotAgent)
        assert robot in self._robot_poses
        del self._robot_poses[robot]

    def remove_landmark(self, beacon: BeaconAgent):
        """removes beacon from the environment

        Args:
            beacon (BeaconAgent): [description]
        """
        assert isinstance(beacon, BeaconAgent)
        assert beacon in self._landmark_points
        del self._landmark_points[beacon]

    def vertex2coordinate(self, vert: Tuple[int, int]) -> Tuple[float, float]:
        """Takes a vertex and returns the corresponding coordinates

        Args:
            vert (Tuple[int, int]): (i, j) vertex

        Returns:
            Tuple[float, float]: (x, y) coordinates
        """
        assert self.check_vertex_valid(vert)

        i, j = vert
        return (self._xv[i, j], self._yv[i, j])

    def vertices2coordinates(
        self, vertices: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """Takes a list of vertices and returns a list of the corresponding coordinates

        Args:
            vertices (List[Tuple[int, int]]): list of (i, j) vertices

        Returns:
            List[Tuple[float, float]]: list of (x, y) coordinates
        """
        assert self.check_vertex_list_valid(vertices)
        return [self.vertex2coordinate(v) for v in vertices]

    # TODO review this function
    def is_xy_on_robot_grid(self, x, y):
        """Checks if a coordinate is in the robot feasible region

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        # if x and y is on the grid and within the robot area
        # its nearest points in the area should be more than two.
        nearest_xy = self.nearest_robot_vertex_coordinates(x, y)
        if len(nearest_xy) >= 2:
            return True
        else:
            return False

    # TODO integrate this with current agent code
    def update_robot_pose(self, robot: RobotAgent, pose: SE2Pose):
        """Just updates the robot pose in the grid. Enforces that the robot
        cannot leave the grid

        Args:
            robot (RobotAgent): [description]
            pose (SE2Pose): [description]

        Returns:
            [type]: [description]
        """
        assert robot in self._robot_poses
        x, y = pose.x, pose.y
        on_grid_in_bound = self.is_xy_on_robot_grid(x, y)
        no_collision = self.no_collision(x, y, self._robot_poses)
        if on_grid_in_bound and no_collision:
            self._robot_poses[robot] = pose
            return True
        if not no_collision:
            print("Update abort: found collision.")
        if not on_grid_in_bound:
            print("Update abort: pose off grid.")
        return False

    def is_robot_vertex(self, i, j):
        return self._robot_feasibility[i, j]

    def vertex_is_landmark_feasible(self, vert: Tuple[int, int]) -> bool:
        """Returns whether the vertex is feasible for landmarks.

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if vertex is feasible for landmarks, False otherwise
        """
        assert self.check_vertex_valid(vert)

        i, j = vert
        return not self._robot_feasibility[i, j]

    def vertex_is_robot_feasible(self, vert: Tuple[int, int]) -> bool:
        """Returns whether the vertex is feasible for robot

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if the vertex is feasible for robot, False otherwise
        """
        assert self.check_vertex_valid(vert)

        i, j = vert
        return self._robot_feasibility[i, j]

    @property
    def vertices(self) -> np.ndarray:
        mesh = np.array(
            np.meshgrid(np.arange(self._x_pts), np.arange(self._y_pts), indexing="ij")
        )
        combinations = mesh.T.reshape(-1, 2)
        return combinations

    @property
    def robot_feasible_vertices(self) -> np.ndarray:
        res = []
        for pt in self.vertices:
            if self.is_robot_vertex(*pt):
                res.append(pt)
        return np.array(res)

    @property
    def landmark_feasible_vertices(self) -> np.ndarray:
        res = []
        for pt in self.vertices:
            if self.vertex_is_landmark_feasible(*pt):
                res.append(pt)
        return np.array(res)

    @property
    def meshgrid(self) -> tuple:
        return self._xv, self._yv

    @property
    def robot_feasibility(self) -> np.ndarray:
        return self._robot_feasibility

    @property
    def shape(self) -> tuple:
        return (self._x_pts, self._y_pts)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def robots(self) -> List:
        return [agent for agent in self._robot_poses]

    @property
    def landmarks(self) -> List:
        return [agent for agent in self._landmark_points]

    def __str__(self):
        line = ""
        line += "Shape: " + self.shape.__repr__() + "\n"
        line += "Cell scale: " + self.scale.__repr__() + "\n"
        line += "Robot feasible vertices: " + self._robot_feasibility.__repr__() + "\n"
        line += (
            "Landmark feasible vertices: "
            + self._landmark_feasibility.__repr__()
            + "\n"
        )
        line += "Robots: " + self._robot_poses.__repr__() + "\n"
        line += "Landmarks: " + self._landmark_points.__repr__() + "\n"
        return line

    def vertex_is_in_bounds(self, vert: Tuple[int, int]) -> bool:
        assert isinstance(vert, tuple)
        assert len(vert) == 2
        assert all(isinstance(x, int) for x in vert)

        x_in_bounds = 0 <= vert[0] < self._x_pts
        y_in_bounds = 0 <= vert[1] < self._y_pts
        return x_in_bounds and y_in_bounds

    def check_vertex_valid(self, vert: Tuple[int, int]):
        """Checks that the indices of the vertex are within the bounds of the grid

        Args:
            vert (tuple): (i, j) indices of the vertex

        Returns:
            bool: True if the vertex is valid, False otherwise
        """
        assert isinstance(vert, tuple)
        assert len(vert) == 2
        assert all(isinstance(i, int) for i in vert)
        assert 0 <= vert[0] < self._x_pts
        assert 0 <= vert[1] < self._y_pts
        return True

    def check_vertex_list_valid(self, vertices: List[tuple]):
        """Checks that the indices of the vertex list are within the bounds of the grid

        Args:
            vertices (List[tuple]): list of vertices
        """
        assert isinstance(vertices, list)
        assert all(isinstance(v, tuple) for v in vertices)
        assert all(self.check_vertex_valid(v) for v in vertices)
        return True

    def robot_edge_path(
        self, feasibility=None, start_point: tuple = None
    ) -> List[tuple]:
        # the default direction is counter-clockwise
        next_wps = []
        # get a list of waypoints along the edge of feasible area
        if feasibility is None:
            feasibility = deepcopy(self.robot_feasibility)

        edge_pts = set()
        feasible_pts = np.array(np.where(feasibility)).T
        # compute edge points first and then consider their order
        for pt in feasible_pts:
            nb_pts = self.get_neighboring_robot_vertices(*pt, feasibility=feasibility)
            if len(nb_pts) < 4:
                edge_pts.add((pt[0], pt[1]))

        if start_point is None:
            # take the top left vertex as the start point
            for i in range(feasibility.shape[0]):
                if start_point is not None:
                    break
                for j in range(feasibility.shape[1]):
                    if feasibility[i, j]:
                        start_point = (i, j)
                        break
        next_wps.append(start_point)

        counterclock_nb = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        while True:
            cur_point = next_wps[-1]
            i, j = cur_point
            feasibility[i, j] = False
            feas_rbt_pts = self.get_neighboring_robot_vertices(i, j, feasibility)
            if len(feas_rbt_pts) > 0:
                pts_degree = np.array(
                    [
                        len(
                            self.get_neighboring_robot_vertices(
                                *pt, feasibility=feasibility
                            )
                        )
                        for pt in feas_rbt_pts
                    ]
                )
                min_degree_idx = np.where(pts_degree == np.amin(pts_degree))[0]
                next_pt_idx = 0
                least_order = np.inf
                for idx in min_degree_idx:
                    diff_vec = (feas_rbt_pts[idx][0] - i, feas_rbt_pts[idx][1] - j)
                    cur_order = counterclock_nb.index(diff_vec)
                    if cur_order < least_order:
                        least_order = cur_order
                        next_pt_idx = idx
                next_wps.append(feas_rbt_pts[next_pt_idx])
                if len(next_wps) == len(edge_pts):
                    if set(next_wps) == edge_pts:
                        if start_point in set(
                            self.get_neighboring_vertices(*next_wps[-1])
                        ):
                            next_wps.append(start_point)
                            break
                        else:
                            raise ValueError("Edge points cannot form a loop.")
                    else:
                        raise ValueError("Non-edge vertices are added.")
            else:
                break
        return next_wps

    def robot_lawn_mower(self, feasibility=None) -> List[tuple]:
        # the default direction is counter-clockwise
        next_wps = []
        # get a list of waypoints along the edge of feasible area
        if feasibility is None:
            feasibility = deepcopy(self.robot_feasibility)

        inverse_i = False
        for j in range(feasibility.shape[1]):
            if feasibility[:, j].any():
                indices = np.where(feasibility[:, j])[0]
                if not inverse_i:
                    for i in indices:
                        next_wps.append((i, j))
                else:
                    for i in indices[::-1]:
                        next_wps.append((i, j))
                inverse_i = not inverse_i
        return next_wps

    def plaza1_path(self) -> List[tuple]:
        edge_path = self.robot_edge_path()
        lawn_mower = self.robot_lawn_mower()
        return edge_path[:-1] + lawn_mower
