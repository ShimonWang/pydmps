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


class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs
    离散 DMP 的实现"""

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        # 调用超级类构造函数
        super(DMPs_discrete, self).__init__(pattern="discrete", **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        # 设置高斯基函数的方差
        # 通过反复试验来确定间距
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time
        设置高斯基函数的中心
        在整个运行时间内均匀分布"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]
            
            x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # 选择我们希望中心位于的时间点
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des)：
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        # 在整个时间段内的预期激活次数
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))  # self.n_bfs*1
        for n in range(len(des_c)):
            # finding x for desired times t
            # 为所需时间 t 寻找 x
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        生成力项的递减前沿项。

        x float：典型系统的当前值
        dmp_num int：当前 dmp 的索引
        """
        return x * (self.goal[dmp_num] - self.y0[dmp_num])  # x(g-y0)

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        生成路径模仿的目标。
        对于有节奏的 DMP，目标是理想轨迹的平均值。
        的平均值。

        y_des np.array：所需的轨迹
        """

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        为给定的典型系统滚动生成基函数活动。

        x float, array:典型系统状态或路径
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]  # x转换为2维数组，且列数为1
        return np.exp(-self.h * (x - self.c) ** 2)  # psi_i = exp(-h_i(x_i-c_i)^2) self.n_bfs*1维度

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        在基函数上生成一组权重，使其与目标强制项轨迹相匹配。
        以匹配目标强制项轨迹。

        f_target np.array：所需的力项轨迹
        w np.array: self.ndmps * self.n_bfs
        """

        # calculate x and psi
        # 计算 x 和 psi
        x_track = self.cs.rollout()  # x_track self.timesteps*1
        psi_track = self.gen_psi(x_track)  # self.timesteps*1

        # efficiently calculate BF weights using weighted linear regression
        # 使用加权线性回归有效计算 BF 权重
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            # 空间缩放项
            k = self.goal[d] - self.y0[d]
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])  # numerator 分子
                denom = np.sum(x_track ** 2 * psi_track[:, b])  # denominator 分母
                self.w[d, b] = numer / denom
                if abs(k) > 1e-5:
                    self.w[d, b] /= k

        self.w = np.nan_to_num(self.w)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为) 或者用户使用nan、posinf和neginf关键字来定义数字


# ==============================
# Test code  # 测试代码
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    # 测试正常运行
    dmp = DMPs_discrete(dt=0.05, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.rollout()

    # 绘图 1
    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track)) * dmp.goal, "r--", lw=2)  # `lw` linewidths的简写 绘制目标点
    plt.plot(y_track, lw=2)  # 绘制跟踪轨迹
    plt.title("DMP system - no forcing term")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["goal", "system state"], loc="lower right")
    # loc="lower right"：这是一个命名参数，用于指定图例的位置。loc 可以是字符串（如 "upper right"、"lower left" 等）
    plt.tight_layout()  # 调整子图之间和子图周围的填充

    # test imitation of path run
    # 测试路径运行的模仿
    plt.figure(2, figsize=(6, 4))
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    # 直线到达目标
    path1 = np.sin(np.arange(0, 1, 0.01) * 5)
    # a strange path to target
    # 一条通往目标的奇怪路径
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0):] = 0.5

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))  # y_des 2*100
        # change the scale of the movement
        # 改变运动的幅度
        dmp.goal[0] = 3
        dmp.goal[1] = 2

        y_track, dy_track, ddy_track = dmp.rollout()

        # 绘图 2
        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)  # dmp1的轨迹
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)  # dmp2的轨迹

    plt.subplot(211)
    a = plt.plot(path1 / path1[-1] * dmp.goal[0], "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend([a[0]], ["desired path"], loc="lower right")  # 选择subplot(211)的a的红线句柄
    # plt.legend(["desired path"], loc="lower right")
    # print([a[0]])
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["%i BFs" % i for i in n_bfs], loc="lower right")

    plt.tight_layout()
    plt.show()
