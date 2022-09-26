import numpy as np
import math
from typing import List, Tuple, Union, Optional, overload

_RAD_TO_DEG_FACTOR = 180.0 / np.pi
_DEG_TO_RAD_FACTOR = np.pi / 180.0
_TWO_PI = 2 * np.pi


def none_to_zero(x: Optional[float]) -> float:
    return 0.0 if x is None else x


def theta_to_pipi(theta: float):
    return (theta + np.pi) % _TWO_PI - np.pi


class Point2(object):
    dim = 2

    def __init__(self, x: float, y: float, frame: str) -> None:
        """[summary]
        Create a 2D point
        : please use += -+ ... to do modify self properties

        Args:
            x (float): the x coordinate
            y (float): the y coordinate
            frame (str): the frame of this point
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(frame, str)
        self._x = x
        self._y = y
        self._frame = frame

    @classmethod
    def by_array(
        cls, other: Union[List[float], Tuple[float, float], np.ndarray], frame: str
    ) -> "Point2":
        return cls(other[0], other[1], frame)

    @staticmethod
    def dist(x1: np.ndarray, x2: np.ndarray):
        """Euclidean distance between two points"""
        assert isinstance(x1, np.ndarray)
        assert isinstance(x2, np.ndarray)
        return np.linalg.norm(x1 - x2)

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        assert isinstance(x, float)
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        assert isinstance(y, float)
        self._y = y

    @property
    def frame(self) -> str:
        return self._frame

    @property
    def norm(self) -> float:
        return np.linalg.norm([self.x, self.y])

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def inverse(self) -> "Point2":
        return Point2(-self.x, -self.y, self.frame)

    def set_x_y(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    def copy(self) -> "Point2":
        """Returns a deep copy of this point

        Returns:
            Point2: A deep copy of this point
        """
        return Point2(self.x, self.y, self.frame)

    def transform_to(self, other: "Point2") -> "Point2":
        """Returns the point other with respect to this point

        Returns:
            Point2: The other point in the frame of this point
        """
        print("Not sure this is a good function to have... raising an error for now")
        raise NotImplementedError
        assert isinstance(other, Point2)
        assert self.frame == other.frame  # frame check

        return other - self

    def distance(self, other: "Point2") -> float:
        """Returns the distance between this point and the other point

        Args:
            other (Point2): the other point

        Returns:
            float: the distance
        """
        assert isinstance(other, Point2)
        assert self.frame == other.frame

        diff_x = self.x - other.x
        diff_y = self.y - other.y
        return np.linalg.norm([diff_x, diff_y])

    def __add__(self, other: "Point2") -> "Point2":
        """Returns a point that is the sum of this point and the other point

        Args:
            other (Point2): the other point

        Returns:
            Point2: The sum of this point and the other point
        """
        assert isinstance(other, Point2)
        assert self.frame == other.frame

        return Point2(self.x + other.x, self.y + other.y, self.frame)

    def __sub__(self, other: "Point2") -> "Point2":
        """Returns a point that is the difference between this point and the
        other point

        Args:
            other (Point2): the other point

        Returns:
            Point2: The difference between this point and the other point
        """
        assert isinstance(other, Point2)
        assert self.frame == other.frame

        return Point2(self.x - other.x, self.y - other.y, self.frame)

    def __mul__(self, other: float) -> "Point2":
        """Returns a scalar multiple of this point

        Args:
            other (float): the scalar multiple

        Returns:
            Point2: The scalar multiple of this point
        """
        assert np.isscalar(other)
        return Point2(self.x * other, self.y * other, self.frame)

    def __rmul__(self, other: float) -> "Point2":
        """Defines the left scalar multiple of this point so can use left or
        right multiplication of scalars with this class

        Args:
            other (float): the scalar multiple

        Returns:
            Point2: The scalar multiple of this point
        """
        return Point2(self.x * other, self.y * other, self.frame)

    def __truediv__(self, other: Union[int, float]) -> "Point2":
        assert np.isscalar(other)
        if other == 0.0:
            raise ValueError("Cannot divide by zeros.")
        else:
            return Point2(self.x / other, self.y / other, self.frame)

    def __iadd__(self, other: "Point2") -> "Point2":
        assert isinstance(other, Point2)
        assert self.frame == other.frame
        self._x += other.x
        self._y += other.y
        return self

    def __isub__(self, other: "Point2") -> "Point2":
        assert isinstance(other, Point2)
        assert self.frame == other.frame
        self._x -= other.x
        self._y -= other.y
        return self

    def __imul__(self, other: float) -> "Point2":
        assert np.isscalar(other)
        self._x *= other
        self._y *= other
        return self

    def __itruediv__(self, other: Union[int, float]) -> "Point2":
        assert np.isscalar(other)
        if other == 0.0:
            raise ValueError("Cannot divide by zeros.")
        else:
            self._x /= other
            self._y /= other
            return self

    def __neg__(self) -> "Point2":
        return self.inverse()

    def __str__(self) -> str:
        string = "Point2{"
        string += f"x: {self.x}, "
        string += f"y: {self.y}, "
        string += f"frame: {self.frame}"
        string += "}"
        return string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point2):
            return False
        if isinstance(other, Point2):
            return (
                abs(self.x - other.x) < 1e-4
                and abs(self.y - other.y) < 1e-4
                and self.frame == other.frame
            )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.frame))


class Rot2(object):
    dim = 1

    def __init__(self, theta: float, local_frame: str, base_frame: str):
        """Create a 2D rotation in radians with a specified base and local
        frame

        Args:
            theta (float): the rotation in radians
            local_frame (str): the local frame of the rotation
            base_frame (str): the base frame of the rotation
        """
        assert isinstance(theta, float)
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)

        # enforcing _theta in [-pi, pi] as a state
        self._theta = theta_to_pipi(none_to_zero(theta))
        self._local_frame = local_frame
        self._base_frame = base_frame
        # free theta
        # self._theta = none_to_zero(theta)

    @classmethod
    def by_degrees(cls, degrees: float, local_frame: str, base_frame: str) -> "Rot2":
        """Generate a rotation from degrees with a specified local and base frame

        Args:
            degrees (float): the rotation in degrees
            local_frame (str): the local frame of the rotation
            base_frame (str): the base frame of the rotation

        Returns:
            Rot2: the 2D rotation
        """
        assert isinstance(degrees, float)
        return cls(none_to_zero(degrees) * _DEG_TO_RAD_FACTOR, local_frame, base_frame)

    @classmethod
    def by_xy(cls, x: float, y: float, local_frame: str, base_frame: str) -> "Rot2":
        """Generate a rotation from the arctan of the x and y values with a
        specified local and base frame

        Args:
            x (float): the x component of the rotation
            y (float): the y component of the rotation
            local_frame (str): the local frame of the rotation
            base_frame (str): the base frame of the rotation

        Returns:
            Rot2: the 2D rotation
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        return cls(
            math.atan2(none_to_zero(y), none_to_zero(x)), local_frame, base_frame
        )

    @classmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str) -> "Rot2":
        """Generates a rotation from a 2x2 rotation matrix
        the matrix should be of the form:
        np.array([ [self.cos, -self.sin],
                    [self.sin,  self.cos]])

        Args:
            matrix (np.ndarray): the 2x2 rotation matrix
            local_frame (str): the local frame of the rotation
            base_frame (str): the base frame of the rotation

        Returns:
            Rot2: the 2D rotation
        """
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float64
        return cls(np.arctan2(matrix[1, 0], matrix[0, 0]), local_frame, base_frame)

    @classmethod
    def exp_map(cls, vector: np.ndarray, local_frame: str, base_frame: str) -> "Rot2":
        """Generate a rotation from a 1d rotation vector (scalar) with a
        specified local and base frame

        expect vector is a 1*1 array for 2D

        Args:
            vector (np.ndarray): the 2D rotation vector
            local_frame (str): the local frame of the rotation
            base_frame (str): the base frame of the rotation

        Returns:
            Rot2: the 2D rotation
        """
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (1, 1)
        assert len(vector) == 1
        return cls(vector[0], local_frame, base_frame)

    @staticmethod
    def dist(x1: "Rot2", x2: "Rot2"):
        """chordal distance between x1 and x2 which are np.ndarray forms of Rot2"""
        return np.linalg.norm((x1.inverse() * x2).log_map())

    def log_map(self):
        """
        Logarithmic map
        """
        return np.array([self.theta])

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def degrees(self) -> float:
        return self.theta * _RAD_TO_DEG_FACTOR

    @property
    def cos(self) -> float:
        return math.cos(self.theta)

    @property
    def sin(self) -> float:
        return math.sin(self.theta)

    @property
    def matrix(self):
        return np.array([[self.cos, -self.sin], [self.sin, self.cos]])

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    def set_theta(self, theta: float = None) -> "Rot2":
        if theta is not None:
            assert np.isscalar(theta)
            self._theta = theta
        return self

    def bearing_to_local_frame_point(self, local_pt: Point2) -> float:
        """
        returns the bearing of the local frame point in radians

        Args:
            local_frame_pt (Point2): the point in the local frame

        Returns:
            float: the bearing of the local frame point in radians
        """
        assert isinstance(local_pt, Point2)
        assert self.local_frame == local_pt.frame
        return math.atan2(local_pt.y, local_pt.x)

    def bearing_to_base_frame_point(self, base_frame_pt: Point2) -> float:
        """Gets the bearing to a given point expressed in the same base frame
        as this rotation

        Args:
            base_frame_pt (Point2): the given point

        Returns:
            float: the bearing in radians
        """
        assert isinstance(base_frame_pt, Point2)
        assert self.base_frame == base_frame_pt.frame

        local_pt = self.unrotate_point(base_frame_pt)
        return math.atan2(local_pt.y, local_pt.x)

    def inverse(self) -> "Rot2":
        """Returns a rotation with the opposite rotation

        Returns:
            Rot2: the inverse of this rotation
        """
        # make sure to flip the frames
        return Rot2(-self.theta, self.base_frame, self.local_frame)

    def copy(self) -> "Rot2":
        """Returns deep copy of this rotation

        Returns:
            Rot2: a deep copy of this rotation
        """
        return Rot2(self.theta, self.local_frame, self.base_frame)

    def rotate_point(self, local_pt: Point2) -> Point2:
        """Rotates a point in the local frame to the base frame

        Args:
            local_pt (Point2): the given point in the local frame

        Returns:
            Point2: the rotated point
        """
        assert isinstance(local_pt, Point2)
        assert self.local_frame == local_pt.frame

        return self * local_pt

    def unrotate_point(self, base_frame_pt: Point2) -> Point2:
        """Rotates a point in the base frame to the local frame

        Args:
            base_frame_pt (Point2): a point in the base frame

        Returns:
            Point2: the point in the local frame
        """
        assert isinstance(base_frame_pt, Point2)
        assert self.base_frame == base_frame_pt.frame

        return self.inverse() * base_frame_pt

    @overload
    def __mul__(self, other: "Rot2") -> "Rot2":
        pass

    @overload
    def __mul__(self, other: Point2) -> Point2:
        pass

    def __mul__(self, other: Union["Rot2", Point2]) -> Union["Rot2", Point2]:
        """Rotates a point or rotation by this rotation

        Args:
            other (Rot2, Point2): the rotation or point to rotate

        Raises:
            ValueError: the other is not a rotation or point

        Returns:
            (Rot2, Point2): the rotated object
        """
        if isinstance(other, Rot2):
            assert self.local_frame == other.base_frame
            return Rot2(self.theta + other.theta, self.local_frame, other.base_frame)
        elif isinstance(other, Point2):
            assert self.local_frame == other.frame
            x = self.cos * other.x - self.sin * other.y
            y = self.sin * other.x + self.cos * other.y
            return Point2(x, y, frame=self.base_frame)

        raise ValueError("Not a Point2 or Rot2 type to multiply.")

    @overload
    def __imul__(self, other: "Rot2") -> "Rot2":
        pass

    @overload
    def __imul__(self, other: Point2) -> Point2:
        pass

    def __imul__(  # type: ignore
        self, other: Union["Rot2", Point2]
    ) -> Union["Rot2", Point2]:
        if isinstance(other, Rot2):
            assert self.local_frame == other.base_frame
            self._theta += other.theta
            self._local_frame = other.local_frame
            return self
        elif isinstance(other, Point2):
            assert self.local_frame == other.frame
            x = self.cos * other.x - self.sin * other.y
            y = self.sin * other.x + self.cos * other.y
            return Point2(x, y, frame=self.base_frame)

        raise ValueError("Not a Rot2 type to multiply.")

    def __str__(self) -> str:
        string = "Rot2{"
        string += f"theta: {self.theta: .3f},"
        string += f"base_frame: {self.base_frame},"
        string += f"local_frame: {self.local_frame}"
        string += "}"
        return string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rot2):
            return False

        if isinstance(other, Rot2):
            same_frames = (
                self.local_frame == other.local_frame
                and self.base_frame == other.base_frame
            )
            angle_similar = abs(self.theta - other.theta) < 1e-8
            return same_frames and angle_similar

    def __hash__(self) -> int:
        return hash(  # type: ignore
            str(self.theta), (self.base_frame), (self.local_frame)
        )  # type: ignore


class SE2Pose(object):
    dim = 3

    def __init__(
        self, x: float, y: float, theta: float, local_frame: str, base_frame: str
    ) -> None:
        """A pose in SE(2) defined by translation and rotation

        Args:
            x (float): the x-coordinate of the pose in the base frame
            y (float): the y-coordinate of the pose in the base frame
            theta (float): the angle of the pose in radians in the base frame
            local_frame (str): the local frame of the pose
            base_frame (str): the base frame of the pose
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(theta, float)
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

        self._point = Point2(x=x, y=y, frame=base_frame)
        self._rot = Rot2(theta=theta, local_frame=local_frame, base_frame=base_frame)
        self._local_frame = local_frame
        self._base_frame = base_frame

    @classmethod
    def by_pt_rt(
        cls, pt: Point2, rt: Rot2, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(pt, Point2)
        assert isinstance(rt, Rot2)
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

        return cls(pt.x, pt.y, rt.theta, local_frame, base_frame)

    @classmethod
    def by_matrix(
        cls, matrix: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        """Constructs a pose from the homogenous transformation matrix
        representation

        the matrix should be:
        np.array([ [self.cos, -self.sin, x],
                     [self.sin,  self.cos, y],
                     [       0,         0, 1]])

        Args:
            matrix: the homogenous transformation matrix
            local_frame: the local frame of the matrix
            base_frame: the base frame of the matrix

        Returns:
            SE2Pose: the pose corresponding to the input matrix
        """
        assert isinstance(matrix, np.ndarray)
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

        pt = Point2.by_array(matrix[0:2, 2], local_frame)
        rt = Rot2.by_matrix(matrix[0:2, 0:2], local_frame, base_frame)
        return SE2Pose.by_pt_rt(pt, rt, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        """Constructs a pose from a exponential map vector

        using notes found here: https://ethaneade.com/lie.pdf

        :expect vector is a 1*3 array for 2D

        Args:
            vector: the exponential map vector
            local_frame: the local frame of the exponential map vector
            base_frame: the base frame of the exponential map vector

        Returns:
            SE2Pose: the pose corresponding to the input vector

        """
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        w = vector[2]
        if abs(w) < 1e-10:
            return SE2Pose(vector[0], vector[1], w, local_frame, base_frame)
        else:
            cos_theta = np.cos(w)
            sin_theta = np.sin(w)

            # get rotation
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            assert R.shape == (2, 2)
            rt = Rot2.by_matrix(R, local_frame, base_frame)

            # get translation
            V = np.array([[sin_theta, cos_theta - 1], [1 - cos_theta, sin_theta]]) / w
            u = vector[0:2]
            t = V @ u
            assert len(u) == 2
            pt = Point2.by_array(t, base_frame)

            return SE2Pose.by_pt_rt(pt, rt, local_frame, base_frame)

    @classmethod
    def by_array(
        cls,
        other: Union[List[float], Tuple[float, float, float], np.ndarray],
        local_frame: str,
        base_frame: str,
    ) -> "SE2Pose":
        """Generate a SE2Pose from a list, tuple, or numpy array

        Args:
            other: a list, tuple, or numpy array of the form [x, y, theta]
            local_frame: the local frame of the pose
            base_frame: the base frame of the pose

        Returns:
            SE2Pose: the pose corresponding to the input vector
        """
        assert isinstance(other, (list, tuple, np.ndarray))
        assert len(other) == 3
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

        return cls(other[0], other[1], other[2], local_frame, base_frame)

    @staticmethod
    def dist(x1: np.ndarray, x2: np.ndarray):
        """chordal distance between x1 and x2 which are np.ndarray forms of SE2
        poses"""
        raise NotImplementedError
        # I'm not sure this is the chordal distance? - Alan
        return np.linalg.norm(
            (SE2Pose.by_array(x1).inverse() * SE2Pose.by_array(x2)).log_map()
        )

    @property
    def theta(self):
        return self._rot.theta

    @property
    def x(self):
        return self._point.x

    @property
    def y(self):
        return self._point.y

    @property
    def rotation(self):
        return self._rot

    @property
    def translation(self):
        return self._point

    @property
    def matrix(self):
        r_c = self._rot.cos
        r_s = self._rot.sin
        x = self._point.x
        y = self._point.y
        return np.array([[r_c, -r_s, x], [r_s, r_c, y], [0, 0, 1]])

    @property
    def array(self):
        return np.array([self.x, self.y, self.theta])

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    def log_map(self):
        r = self._rot
        t = self._point
        w = r.theta
        if abs(w) < 1e-10:
            return np.array([t.x, t.y, w])
        else:
            c_1 = r.cos - 1.0
            s = r.sin
            det = c_1 * c_1 + s * s
            rot_pi_2 = Rot2(np.pi / 2.0, self.local_frame, self.base_frame)
            p = rot_pi_2 * (r.unrotate_point(t) - t)
            v = (w / det) * p
            return np.array([v.x, v.y, w])

    def grad_x_logmap(self):
        if abs(self.theta) < 1e-10:
            return np.identity(3)
        else:
            logmap_x, logmap_y, logmap_th = self.log_map()
            th_2 = logmap_th / 2.0
            diag1 = th_2 * np.sin(logmap_th) / (1.0 - np.cos(logmap_th))
            return np.array(
                [
                    [
                        diag1,
                        th_2,
                        (
                            logmap_x / logmap_th
                            + th_2 * (self.x / (np.cos(logmap_th) - 1))
                        ),
                    ],
                    [
                        -th_2,
                        diag1,
                        (
                            logmap_y / logmap_th
                            + th_2 * (self.y / (np.cos(logmap_th) - 1))
                        ),
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )

    def range_and_bearing_to_point(self, pt: Point2) -> Tuple[float, float]:
        """Returns the range and bearing from this pose to the point

        Args:
            pt (Point2): the point to measure to

        Returns:
            Tuple[float, float]: (range, bearing)
        """
        assert isinstance(pt, Point2)
        diff = pt - self._point
        dist = diff.norm
        bearing = self._rot.bearing_to_base_frame_point(diff)
        # bearing = self._rot.bearing_to_local_frame_point(diff)
        # bearing = self._rot.bearing(diff)
        return dist, bearing

    def inverse(self) -> "SE2Pose":
        inv_t = -(self._rot.unrotate_point(self._point))

        # make sure that the base and local frames are flipped
        return SE2Pose.by_pt_rt(
            pt=inv_t,
            rt=self._rot.inverse(),
            local_frame=self._base_frame,
            base_frame=self._local_frame,
        )

    def copy(self) -> "SE2Pose":
        """Returns a deep copy of this pose

        Returns:
            SE2Pose: deep copy of this pose
        """
        return SE2Pose(
            x=self.x,
            y=self.y,
            theta=self.theta,
            local_frame=self._local_frame,
            base_frame=self._base_frame,
        )

    def transform_to(self, other):
        """Returns the coordinate frame of the other pose in the coordinate
        frame of this pose.

        e.g.    T_0_to_1 -> T_0_to_2
                = inv(T_0_to_1) * T_0_to_2
                = T_1_to_0 * T_0_to_2
                = T_1_to_2

        Args:
            other (SE2Pose): the other pose which we want the coordinate frame
                with respect to

        Returns:
            SE2Pose: the coordinate frame of this pose with respect to the given
                pose
        """
        assert isinstance(other, SE2Pose)

        # must have common base frame
        assert self.base_frame == other.base_frame

        # doesn't make sense if have same local frame
        assert not self.local_frame == other.local_frame

        return self.inverse() * other

    def transform_local_point_to_base(self, local_point: Point2) -> Point2:
        """Returns a point expressed in local frame of self in the base frame of
        self

        Args:
            local_point (Point2): the point expressed in the local frame of self

        Returns:
            Point2: a point expressed in the base frame of self
        """
        assert isinstance(local_point, Point2)
        return self * local_point

    def transform_base_point_to_local(self, base_point: Point2) -> Point2:
        """Returns a point expressed in base frame of self in the local frame of
        self

        Args:
            base_point (Point2): the point expressed in the base frame of self

        Returns:
            Point2: a point expressed in the local frame of self
        """
        assert isinstance(base_point, Point2)
        return self.inverse() * base_point

    def distance_to_pose(self, other: "SE2Pose") -> float:
        """Returns the distance between this pose and another pose

        Args:
            other (SE2Pose): the other pose

        Returns:
            float: the distance between this pose and the other pose
        """
        assert isinstance(other, SE2Pose)

        cur_position = self.translation
        other_position = other.translation
        assert cur_position.frame == other_position.frame

        dist = cur_position.distance(other_position)

        assert isinstance(dist, float)
        return dist

    def __mul__(self, other):
        assert isinstance(other, (SE2Pose, Point2))

        if isinstance(other, SE2Pose):
            assert self.local_frame == other.base_frame
            r = self._rot * other.rotation
            t = self._point + self._rot * other.translation
            return SE2Pose.by_pt_rt(
                pt=t, rt=r, local_frame=other.local_frame, base_frame=self.base_frame
            )
        if isinstance(other, Point2):
            assert self.local_frame == other.frame
            return self._rot * other + self._point

    def __imul__(self, other):
        if isinstance(other, SE2Pose):
            pos = self * other
            self._point = self._point.set_x_y(x=pos.x, y=pos.y)
            self._rot = self._rot.set_theta(pos.theta)
            self._local_frame = other.local_frame
            return self
        raise ValueError("Not a Pose2 type to multiply.")

    def __str__(self) -> str:
        string = "Pose2{"
        string += f"x: {self.x}, "
        string += f"y: {self.y}, "
        string += f"theta: {self.theta}, "
        string += f"local_frame: {self.local_frame}, "
        string += f"base_frame: {self.base_frame}"
        string += "}"
        return string

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SE2Pose):
            return (
                abs(self._rot.theta - other.theta) < 1e-8
                and abs(self._point.x - other.x) < 1e-8
                and abs(self._point.y - other.y) < 1e-8
                and self._local_frame == other.local_frame
                and self._base_frame == other.base_frame
            )
        return False

    def __hash__(self):
        return hash(
            (
                self._point.x,
                self._point.y,
                self._rot.theta,
                self._local_frame,
                self._base_frame,
            )
        )
