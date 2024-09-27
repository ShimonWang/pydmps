"""
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pydmps.dmp import DMPs

import numpy as np


class DMPs_rhythmic(DMPs):
    """An implementation of discrete DMPs
    离散 DMP 的实现"""

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        # 调用超级类构造函数
        super(DMPs_rhythmic, self).__init__(pattern="rhythmic", **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        # 设置高斯基函数的方差
        # 通过反复试验来确定间距
        self.h = np.ones(self.n_bfs) * self.n_bfs  # 1.75

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time
        设置高斯基函数的中心
        在整个运行时间内均匀分布"""

        c = np.linspace(0, 2 * np.pi, self.n_bfs + 1)  # np.linspace(start, stop, num=50, **kwargs)
        c = c[0:-1]  # 索引[ , )左开右闭
        self.c = c

    def gen_front_term(self, x, dmp_num):
        """Generates the front term on the forcing term.
        For rhythmic DMPs it's non-diminishing, so this
        function is just a placeholder to return 1.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        生成力项的前项。
        对于有节奏的 DMP，它是非递减的，因此这个
        函数只是一个返回 1 的占位符。

        x float：典型系统的当前值
        dmp_num int：当前 DMP 的索引
        """

        if isinstance(x, np.ndarray):
            return np.ones(x.shape)
        return 1

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        生成路径模仿的目标。
        对于有节奏的 DMP，目标是理想轨迹的平均值。
        的平均值。

        y_des np.数组：所需的轨迹
        """

        goal = np.zeros(self.n_dmps)
        for n in range(self.n_dmps):
            num_idx = ~np.isnan(y_des[n])  # ignore nan's when calculating goal # 计算目标时忽略 nan’s
            goal[n] = 0.5 * (y_des[n, num_idx].min() + y_des[n, num_idx].max())

        return goal

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system state or path.

        x float, array: the canonical system state or path
        为给定的典型系统状态或路径生成基函数活动。
        典型系统状态或路径的基函数。

        x 浮点数，数组：典型系统状态或路径
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]  # x转换为2维数组，且列数为1
        return np.exp(self.h * (np.cos(x - self.c) - 1))  # psi=exp(h*cos(x-c)-1)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        在基函数上生成一组权重，使其与目标强制项轨迹相匹配。
        以匹配目标强制项轨迹。

        f_target np.array：所需的力项轨迹
        """

        # calculate x and psi
        # 计算 x 和 psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        # 使用加权线性回归有效计算 BF 权重
        for d in range(self.n_dmps):
            for b in range(self.n_bfs):
                self.w[d, b] = np.dot(psi_track[:, b], f_target[:, d]) / (
                        np.sum(psi_track[:, b]) + 1e-10
                )


# ==============================
# Test code  # 测试代码
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    # 测试正常运行
    dmp = DMPs_rhythmic(n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.rollout()

    # 绘图 1
    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track)) * dmp.goal, "r--", lw=2)
    plt.plot(y_track, lw=2)
    plt.title("DMP system - no forcing term")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["goal", "system state"], loc="lower right")
    plt.tight_layout()

    # test imitation of path run
    # 测试路径运行的模仿
    plt.figure(2, figsize=(6, 4))
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    # 直线到达目标
    path1 = np.sin(np.arange(0, 2 * np.pi, 0.01) * 5)
    # a strange path to target
    # 一条通往目标的奇怪路径
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0):] = 0.5

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_rhythmic(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))
        y_track, dy_track, ddy_track = dmp.rollout()

        # 绘图 2
        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)  # dmp1的轨迹
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)  # dmp2的轨迹

    # 图窗2 子图1
    plt.subplot(211)
    a = plt.plot(path1, "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend([a[0]], ["desired path"], loc="lower right")

    # 图窗2 子图2
    plt.subplot(212)
    b = plt.plot(path2, "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["%i BFs" % i for i in n_bfs], loc="lower right")

    plt.tight_layout()
    plt.show()
