from copy import deepcopy

import random
import itertools
import numpy as np
from typing import Tuple, List, Union
import matplotlib.pyplot as plt

from manhattan.geometry.TwoDimension import SE2Pose, Point2
from manhattan.agent.agent import Robot


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


# TODO rewrite to capture the row_corner_number and col_corner_number cases
class ManhattanWorld:
    """
    This class creates a simulated environment of Manhattan world with beacons.
    """

    def __init__(
        self,
        grid_vertices_shape: tuple = (9, 9),
        row_corner_number: int = 1,
        column_corner_number: int = 1,
        cell_scale: float = 1.0,
        robot_area: List[Tuple] = None,
        check_collision: bool = True,
        tol: float = 1e-5,
    ):
        """Constructor for Manhattan waterworld environment. Note that the
        beacons are only allowed in areas that is infeasible to the robot. As
        of now the robot feasible area is only rectangular

        Args:
            grid_vertices_shape (tuple, optional): a tuple defining the shape of
                grid vertices; note that the vertices follow ij indexing.
                Defaults to (9, 9).
            cell_scale (int, optional): width and length of a cell. Defaults to 1.
            robot_area (List[Tuple], optional): [(left, bottom), (right, top)]
                bottom left and top right vertices of a rectangular area; all
                the rest area will be infeasible. Defaults to None.
            check_collision (bool, optional): [description]. Defaults to True.
            tol (float, optional): [description]. Defaults to 1e-5.
        """
        assert isinstance(grid_vertices_shape, tuple)
        assert len(grid_vertices_shape) == 2
        self._num_x_pts, self._num_y_pts = grid_vertices_shape

        # have to add one to get the number of rows and columns
        self._num_x_pts += 1
        self._num_y_pts += 1

        assert isinstance(row_corner_number, int)
        assert isinstance(column_corner_number, int)
        self._row_corner_number = row_corner_number
        self._column_corner_number = column_corner_number

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
        self._x_coords = np.arange(self._num_x_pts) * self._scale
        self._y_coords = np.arange(self._num_y_pts) * self._scale
        self._xv, self._yv = np.meshgrid(self._x_coords, self._y_coords, indexing="ij")

        # agents are added by vertices but stored with groundtruth poses or points
        self._robot_poses = {}
        self._beacon_points = {}

        if robot_area is not None:
            # ensure a rectangular feasible area for robot
            bl, tr = robot_area

            # set bounds on feasible area as variables
            self._min_x_idx_feasible = bl[0]
            self._max_x_idx_feasible = tr[0]
            self._min_y_idx_feasible = bl[1]
            self._max_y_idx_feasible = tr[1]

            self._min_x_coord_feasible = bl[0] * self._scale
            self._max_x_coord_feasible = tr[0] * self._scale
            self._min_y_coord_feasible = bl[1] * self._scale
            self._max_y_coord_feasible = tr[1] * self._scale

            # also save a mask for the feasible area
            self._robot_feasibility = np.zeros(
                (self._num_x_pts, self._num_y_pts), dtype=bool
            )
            self._robot_feasibility[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
        else:
            # if no area specified, all area is now feasible

            # set bounds on feasible area as variables
            self._min_x_idx_feasible = 0
            self._max_x_idx_feasible = self._num_x_pts - 1
            self._min_y_idx_feasible = 0
            self._max_y_idx_feasible = self._num_y_pts - 1

            self._min_x_coord_feasible = np.min(self._x_coords)
            self._max_x_coord_feasible = np.max(self._x_coords)
            self._min_y_coord_feasible = np.min(self._y_coords)
            self._max_y_coord_feasible = np.max(self._y_coords)

            # also save a mask for the feasible area
            self._robot_feasibility = np.ones(
                (self._num_x_pts, self._num_y_pts), dtype=bool
            )

        # make sure nothing weird happened in recording these feasible values
        assert self._x_coords[self._min_x_idx_feasible] == self._min_x_coord_feasible
        assert self._x_coords[self._max_x_idx_feasible] == self._max_x_coord_feasible
        assert self._y_coords[self._min_y_idx_feasible] == self._min_y_coord_feasible
        assert self._y_coords[self._max_y_idx_feasible] == self._max_y_coord_feasible

    def __str__(self):
        line = "ManhattanWorld Environment\n"
        line += "Shape: " + self.shape.__repr__() + "\n"
        line += f"Row Corner Number: {self.row_corner_number}\n"
        line += f"Column Corner Number: {self.column_corner_number}\n"
        line += f"Cell Scale: {self.cell_scale}\n"
        line += f"Robot Feasible Area: {self.robot_area}\n"
        line += (
            "Beacon feasible vertices: " + self._beacon_feasibility.__repr__() + "\n"
        )
        line += "Robots: " + self._robot_poses.__repr__() + "\n"
        line += "Beacons: " + self._beacon_points.__repr__() + "\n"
        return line

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, 0.0, self._x_coords[-1], self._y_coords[-1])

    def set_robot_area_feasibility(self, area: List[Tuple[int, int]]):
        """Sets the feasibility status for the robots as a rectangular area. Anything
        outside of this area will be the inverse of the status.

        Args:
            area (List[Tuple[int, int]]): the feasibility area for robots, denoted by the
                bottom left and top right vertices.
        """
        assert self.check_vertex_list_valid(area)
        assert len(area) == 2

        mask = np.zeros((self._num_x_pts, self._num_y_pts), dtype=bool)
        bl, tr = area

        # set bounds on feasible area as variables
        self._min_x_idx_feasible = bl[0]
        self._max_x_idx_feasible = tr[0]
        self._min_y_idx_feasible = bl[1]
        self._max_y_idx_feasible = tr[1]

        self._min_x_coord_feasible = bl[0] * self._scale
        self._max_x_coord_feasible = tr[0] * self._scale
        self._min_y_coord_feasible = bl[1] * self._scale
        self._max_y_coord_feasible = tr[1] * self._scale

        # also save a mask for the feasible area
        mask[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
        self._robot_feasibility[mask] = True
        self._robot_feasibility[np.invert(mask)] = False

        # make sure nothing weird happened in recording these feasible values
        assert (
            abs(self._min_x_idx_feasible * self._scale - self._min_x_coord_feasible)
            < self._tol
        )
        assert (
            abs(self._max_x_idx_feasible * self._scale - self._max_x_coord_feasible)
            < self._tol
        )
        assert (
            abs(self._min_y_idx_feasible * self._scale - self._min_y_coord_feasible)
            < self._tol
        )
        assert (
            abs(self._max_y_idx_feasible * self._scale - self._max_y_coord_feasible)
            < self._tol
        )

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

        # connectivity is based on whether we are at a corner or not
        if i % self._column_corner_number == 0:
            candidate_vertices.append((i - 1, j))
            candidate_vertices.append((i + 1, j))
        if j % self._row_corner_number == 0:
            candidate_vertices.append((i, j - 1))
            candidate_vertices.append((i, j + 1))

        # prune all vertices that are out of bounds
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
            vert (tuple): a vertex index (i, j)

        Returns:
            List[Tuple[int, int]]: the list of neighboring vertices that are
                feasible for the robot
        """
        assert self.check_vertex_valid(vert)

        neighbor_verts = self.get_neighboring_vertices(vert)
        assert self.check_vertex_list_valid(neighbor_verts)

        feasible_neighbor_verts = [
            v for v in neighbor_verts if self.vertex_is_robot_feasible(v)
        ]

        return feasible_neighbor_verts

    def get_neighboring_robot_vertices_not_behind_robot(
        self, robot: Robot,
    ) -> List[Tuple[int, int]]:
        """get all neighboring vertices to the vertex the robot is at which are
        not behind the given robot

        Args:
            robot (Robot): the robot

        Returns:
            List[Tuple[int, int]]: the list of neighboring vertices that are
                not behind the robot
        """
        assert isinstance(robot, Robot)

        # get robot position
        robot_loc = robot.position
        robot_pose = robot.pose

        # get robot vertex
        robot_vert = self.point2vertex(robot_loc)
        assert self.check_vertex_valid(robot_vert)

        # get neighboring vertices in the robot feasible space
        neighboring_feasible_vertices = self.get_neighboring_robot_vertices(robot_vert)
        assert self.check_vertex_list_valid(neighboring_feasible_vertices)

        # convert vertices to points
        neighboring_feasible_pts = [
            self.vertex2point(v) for v in neighboring_feasible_vertices
        ]

        not_behind_pts = []
        for pt in neighboring_feasible_pts:
            distance, bearing = robot_pose.range_and_bearing_to_point(pt)
            if np.abs(bearing) < (np.pi / 2) + self._tol:
                not_behind_pts.append((pt, bearing))

        return not_behind_pts

    def get_random_robot_pose(self, local_frame: str) -> SE2Pose:
        """Returns a random, feasible robot pose located on a corner in the
        grid.

        Note: this will not sample any points on the edge of the grid

        Returns:
            SE2Pose: a random, feasible robot pose
        """
        assert isinstance(local_frame, str)

        feasible_x_vals = (self._min_x_coord_feasible < self._x_coords) & (
            self._x_coords < self._max_x_coord_feasible
        )
        feasible_y_vals = (self._min_y_coord_feasible < self._y_coords) & (
            self._y_coords < self._max_y_coord_feasible
        )

        cornered_x_vals = np.zeros(feasible_x_vals.shape).astype(bool)
        cornered_y_vals = np.zeros(feasible_y_vals.shape).astype(bool)
        for i in range(len(cornered_x_vals)):
            if i % self._column_corner_number == 0:
                cornered_x_vals[i] = True
        for j in range(len(cornered_y_vals)):
            if j % self._row_corner_number == 0:
                cornered_y_vals[j] = True

        sampleable_x_vals = cornered_x_vals & feasible_x_vals
        sampleable_y_vals = cornered_y_vals & feasible_y_vals

        x_sample = np.random.choice(self._x_coords[sampleable_x_vals])
        y_sample = np.random.choice(self._y_coords[sampleable_y_vals])

        # pick a rotation from 0 to 3/2 pi
        rotation_sample = np.random.choice(np.linspace(0, (3 / 2) * np.pi, num=4))

        return SE2Pose(
            x_sample,
            y_sample,
            rotation_sample,
            local_frame=local_frame,
            base_frame="world",
        )

    def get_random_beacon_point(self, frame: str) -> Point2:
        """Returns a random beacon point on the grid.

        Args:
            frame (str): the frame of the beacon

        Returns:
            Point2: a random valid beacon point, None if no position is feasible
        """
        assert isinstance(frame, str)

        # TODO this is also somewhat naive but it works... could revisit this later

        # get random beacon position by generating all possible coordinates and
        # then just pruning those that are not feasible for the beacon
        x_idxs = np.arange(self._num_x_pts)
        y_idxs = np.arange(self._num_y_pts)

        # combination of all possible verts on the grid
        possible_verts = itertools.product(x_idxs, y_idxs)  # cartesian product

        # prune out the infeasible vertices
        feasible_verts = [
            vert for vert in possible_verts if self.vertex_is_beacon_feasible(vert)
        ]

        if len(feasible_verts) == 0:
            return None

        # randomly sample one of the vertices
        vert_sample = random.choice(feasible_verts)
        # vert_sample = feasible_verts[vert_sample_idx]

        i, j = vert_sample
        position = Point2(self._xv[i, j], self._yv[i, j], frame=frame)
        return position

    ###### Coordinate and vertex conversion methods ######

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

    def vertex2point(self, vert: Tuple[int, int]) -> Point2:
        """Takes a vertex and returns the corresponding point in the world frame

        Args:
            vert (Tuple[int, int]): (i, j) vertex

        Returns:
            Point2: point in the world frame
        """
        assert self.check_vertex_valid(vert)

        x, y = self.vertex2coordinate(vert)
        return Point2(x, y, frame="world")

    def point2vertex(self, point: Point2) -> Tuple[int, int]:
        """Takes a point in the world frame and returns the corresponding
        vertex

        Args:
            point (Point2): point in the world frame

        Returns:
            Tuple[int, int]: (i, j) vertex
        """
        assert isinstance(point, Point2)
        assert point.frame == "world"

        x, y = point.x, point.y
        return self.coordinate2vertex(x, y)

    ####### Check vertex validity #########

    def pose_is_robot_feasible(self, pose: SE2Pose) -> bool:
        """Takes in a pose and returns whether the robot is feasible at that
        pose. Checks that rotation is a multiple of pi/2, that the
        position is on a robot feasible point in the grid

        Args:
            pose (SE2Pose): the pose to check

        Returns:
            bool: True if the robot is feasible at that pose, False otherwise
        """
        assert isinstance(pose, SE2Pose), f"pose: {pose}, type: {type(pose)}"

        rotation_is_good = abs(pose.theta % (np.pi / 2.0)) < self._tol
        if not rotation_is_good:
            print(f"Rotation is {pose.theta} and not a multiple of pi/2")
            return False

        vert = self.coordinate2vertex(pose.x, pose.y)
        if not self.vertex_is_robot_feasible(vert):
            print(f"Coordinate {pose.x}, {pose.y} from vertex {vert} is not feasible")
            return False

        return True

    def position_is_beacon_feasible(self, position: Point2) -> bool:
        """Takes in a position and returns whether the position is feasible
        for a beacon.

        Args:
            position (Point2): the position to check

        Returns:
            bool: True if the position is feasible for a beacon, False otherwise
        """
        assert isinstance(position, Point2)

        vert = self.coordinate2vertex(position.x, position.y)
        return self.vertex_is_beacon_feasible(vert)

    def vertex_is_beacon_feasible(self, vert: Tuple[int, int]) -> bool:
        """Returns whether the vertex is feasible for beacons.

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if vertex is feasible for beacons, False otherwise
        """
        assert self.check_vertex_valid(vert)

        # if not a robot travelable location then it is good for a beacon
        return not self.vertex_is_robot_feasible(vert)

    def vertex_is_robot_feasible(self, vert: Tuple[int, int]) -> bool:
        """Returns whether the vertex is feasible for robot. This checks whether
        the index of the vertex would be on one of the allowed lines the robot
        can travel on and then returns whether this is within the defined
        'feasible region'

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if the vertex is feasible for robot, False otherwise
        """
        assert self.check_vertex_valid(vert)

        i, j = vert

        # vertex can only be feasible if on one of the lines defined by the
        # row/column spacing
        if i % self._column_corner_number == 0 or j % self._row_corner_number == 0:
            return self._robot_feasibility[i, j]
        else:
            return False

    def vertex_is_in_bounds(self, vert: Tuple[int, int]) -> bool:
        assert isinstance(vert, tuple)
        assert len(vert) == 2
        assert all(isinstance(x, np.integer) for x in vert)

        x_in_bounds = 0 <= vert[0] < self._num_x_pts
        y_in_bounds = 0 <= vert[1] < self._num_y_pts
        return x_in_bounds and y_in_bounds

    def check_vertex_valid(self, vert: Tuple[int, int]):
        """Checks that the indices of the vertex are within the bounds of the grid

        Args:
            vert (tuple): (i, j) indices of the vertex

        Returns:
            bool: True if the vertex is valid, False otherwise
        """
        assert isinstance(vert, tuple)
        assert len(vert) == 2, f"vert: {vert}, len: {len(vert)}"
        assert all(
            isinstance(i, np.integer) for i in vert
        ), f"vert: {vert}, type1: {type(vert[0])}, type2: {type(vert[1])}"
        assert 0 <= vert[0] < self._num_x_pts
        assert 0 <= vert[1] < self._num_y_pts
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

    ####### visualization #############

    def plot_environment(self):
        assert self._robot_feasibility.shape == (self._num_x_pts, self._num_y_pts)

        # get rows and cols that the robot is allowed to travel on
        x_pts = np.arange(self._num_x_pts)
        valid_x = x_pts[x_pts % self._column_corner_number == 0]
        valid_x = self._scale * valid_x

        y_pts = np.arange(self._num_y_pts)
        valid_y = y_pts[y_pts % self._row_corner_number == 0]
        valid_y = self._scale * valid_y

        # the bounds of the valid x and y values
        max_x = np.max(valid_x)
        min_x = np.min(valid_x)
        max_y = np.max(valid_y)
        min_y = np.min(valid_y)

        # plot the travelable rows and columns
        plt.vlines(valid_x, min_y, max_y)
        plt.hlines(valid_y, min_x, max_x)

        for i in range(self._num_x_pts):
            for j in range(self._num_y_pts):

                # the robot should not be traveling on these locations
                if (
                    i % self._column_corner_number != 0
                    and j % self._row_corner_number != 0
                ):
                    continue

                if self._robot_feasibility[i, j]:
                    plt.plot(self._xv[i, j], self._yv[i, j], "ro", markersize=3)
                else:
                    plt.plot(self._xv[i, j], self._yv[i, j], "go", markersize=3)

