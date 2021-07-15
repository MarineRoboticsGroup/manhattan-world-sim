import unittest
import numpy.testing as test

from Environment import *

class ManhattanWaterworldTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scale = 2
        cls.bl = (2,2)
        cls.tr = (4, 3)
        cls.size = (7, 6)
        cls.env1 = ManhattanWaterworld(cls.size, cls.scale, [cls.bl,cls.tr])
        cls.env2 = ManhattanWaterworld(cls.size, cls.scale, None, [cls.bl,cls.tr],check_collision=False)
        cls.envs = [cls.env1, cls.env2]
        cls.x_min = 0
        cls.x_max = (cls.size[0] - 1) * cls.scale
        cls.y_min = 0
        cls.y_max = (cls.size[1] - 1) * cls.scale
        cls.bl_xy = (cls.x_min, cls.y_min)
        cls.tr_xy = (cls.x_max, cls.y_max)

        cls.env1_rbt_vertices = {(2,2),(2,3),(3,2),(3,3),(4,2),(4,3)}
        cls.env2_lmk_vertices = {(2,2),(2,3),(3,2),(3,3),(4,2),(4,3)}
        cls.vertices = np.array([np.meshgrid(np.arange(cls.size[0]),np.arange(cls.size[1]))]).T.reshape(-1,2)

    def test_init(self):
        envs = [self.env1, self.env2]
        for env in envs:
            test.assert_equal(self.x_max, env._x_coords[-1])
            test.assert_equal(self.x_min, env._x_coords[0])
            test.assert_equal(self.y_max, env._y_coords[-1])
            test.assert_equal(self.y_min, env._y_coords[0])
            self.assertCountEqual(set(map(tuple, self.vertices)), set(map(tuple, env.vertices)))
            test.assert_equal(self.bl_xy, env.vertex2coordinate(*env.vertices[0]))
            test.assert_equal(self.tr_xy, env.vertex2coordinate(*env.vertices[-1]))
        self.assertCountEqual(self.env1_rbt_vertices, set(map(tuple, self.env1.robot_feasible_vertices)))
        self.assertCountEqual(self.env2_lmk_vertices, set(map(tuple, self.env2.landmark_feasible_vertices)))
        self.assertCountEqual(set(map(tuple, self.env1.vertices))-set(map(tuple, self.env1.robot_feasible_vertices)),
                              set(map(tuple, self.env2.vertices))-set(map(tuple, self.env2.landmark_feasible_vertices)))

    def test_rbt_lmk(self):
        for i, env in enumerate(self.envs):
            if i == 0:
                self.assertFalse(env.add_robot('rbt', 0, 0))
                self.assertTrue(env.add_robot('rbt', self.bl[0], self.bl[1]))
                self.assertFalse(env.add_robot('rbt1', self.bl[0], self.bl[1]))
                self.assertFalse(env.add_robot('rbt', self.tr[0], self.tr[1]))
                self.assertTrue(env.add_robot('rbt1', self.tr[0], self.tr[1]))
                self.assertTrue(env.remove_robot('rbt'))
                self.assertTrue(env.reset_robot('rbt1', self.tr[0], self.tr[1]))
                self.assertTrue(list(env.robots)[0] == 'rbt1')
                self.assertTrue(env.update_robot_pose('rbt1', SE2Pose(6,6, -np.pi/2)))
                self.assertFalse(env.update_robot_pose('rbt1', SE2Pose(6,6.6, -np.pi/2)))
                self.assertFalse(env.update_robot_pose('rbt1', SE2Pose(0,0, -np.pi/2)))
            else:
                self.assertFalse(env.add_landmark('rbt', 0, 0))
                self.assertTrue(env.add_landmark('rbt', self.bl[0], self.bl[1]))
                self.assertTrue(env.add_landmark('rbt1', self.bl[0], self.bl[1]))
                self.assertFalse(env.add_landmark('rbt', self.tr[0], self.tr[1]))
                self.assertTrue(env.remove_landmark('rbt'))
                self.assertTrue(env.reset_landmark('rbt1', self.tr[0], self.tr[1]))
                self.assertTrue(list(env.landmarks)[0] == 'rbt1')
                self.assertFalse(env.add_robot('rbt', self.tr[0], self.tr[1]))
                self.assertTrue(env.add_robot('rbt', 0, 0))
                self.assertTrue(env.add_robot('rbt1', env._x_pts-1, env._y_pts-1))
                self.assertTrue(env.reset_robot('rbt1', 0, 0))
                self.assertTrue(env.update_robot_pose('rbt1', SE2Pose(0,0,-np.pi)))

if __name__ == '__main__':
    unittest.main()