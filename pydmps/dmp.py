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

特拉维斯·德沃尔夫版权所有

本程序是自由软件:您可以重新发布和/或修改它
在GNU通用公共许可证的条款下，由
自由软件基金会，或者是许可证的第三版，或者
(根据你的选择)任何以后的版本。

发布这个程序是希望它能有所帮助，
但无任何保证;甚至没有隐含的保证
适销性或适合于某一特定目的。看到
详细信息请参见GNU通用公共许可证。

您应该已经收到了GNU通用公共许可证的副本
和这个项目一起。如果没有，请参见 <http://www.gnu.org/licenses/>。
"""
import numpy as np

from pydmps.cs import CanonicalSystem


class DMPs(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper.
    动态电机基元的实现,
    正如 Stefan Schaal 博士（2002 年）的论文中所描述的那样。"""

    def __init__(
        self, n_dmps, n_bfs, dt=0.01, y0=0, goal=1, w=None, ay=None, by=None, **kwargs
    ):  # 关键字参数列表**kwargs
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        n_dmps int：动态电机基元数
        n_bfs int： 每个 DMP 的基函数数量
        dt float：模拟的时间步长
        y0 list：DMP 的初始状态
        goal list：DMP 的目标状态
        w list：可调参数，控制基函数的振幅
        ay int：吸引子项 y 动态的增益 alphay
        by int：吸引子项 y 动力学增益 betay
        """

        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        if isinstance(y0, (int, float)):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))     # 每个 DMP 的基函数数量
        self.w = w

        self.ay = np.ones(n_dmps) * 25.0 if ay is None else ay  # Schaal 2012  # b = a if a > 0 else -a
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012

        # set up the CS
        # 设置 CS 正则系统
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.reset_state()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0
        检查初始位置和目标是否相同
        如果相同，则稍作偏移，使力项不为 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4

    def gen_front_term(self, x, dmp_num):
        raise NotImplementedError()  # NotImplementedError: 子类没有实现父类要求一定要实现的接口。

    def gen_goal(self, y_des):
        raise NotImplementedError()

    def gen_psi(self):
        raise NotImplementedError()

    def gen_weights(self, f_target):
        raise NotImplementedError()

    def imitate_path(self, y_des, plot=False):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        接收所需的轨迹，并生成一套最能实现该轨迹的
        系统参数。

        y_des 列表/数组：每个 DMP 的期望轨迹
                          应为 [n_dmps, run_time] 形状
        """

        # set initial state and goal
        # 设置初始状态和目标
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()  # 每个DMP的期望轨迹 disire
        self.goal = self.gen_goal(y_des)

        # self.check_offset()

        # generate function to interpolate the desired trajectory
        # 生成插值所需轨迹的函数
        import scipy.interpolate  # scipy.interpolate	插值

        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])  # shape[0]：表示矩阵的行数 shape[1]：表示矩阵的列数
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])  # 一维插值一维数据的插值运算可以通过方法 interp1d() 完成 `gen`=generate
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des with central differences
        # 利用中心差计算 y_des 的速度
        dy_des = np.gradient(y_des, axis=1) / self.dt  # np.gradient(f):用于计算数组f中元素的梯度，当f为多维时，返回每个维度的梯度。

        # calculate acceleration of y_des with central differences
        # 利用中心差计算 y_des 的加速度
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        f_target = np.zeros((y_des.shape[1], self.n_dmps))  # self.timesteps行 self.n_dmps列
        # find the force required to move along this trajectory
        # 求出沿此轨迹移动所需的力
        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]
            )

        # efficiently generate weights to realize f_target
        # 有效生成权重以实现 f_target
        self.gen_weights(f_target)

        if plot is True:
            # plot the basis function activations
            # 绘制基函数激活图
            import matplotlib.pyplot as plt  # ?

            plt.figure()
            plt.subplot(211)
            psi_track = self.gen_psi(self.cs.rollout())
            plt.plot(psi_track)
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            # 绘制所需的力函数与近似值的对比图
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[:, ii], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(
                    np.sum(psi_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset_state()
        return y_des

    def rollout(self, timesteps=None, **kwargs):
        """Generate a system trial, no feedback is incorporated.
        生成系统试验，不包含反馈。"""

        self.reset_state()  # 每次模拟前重置系统状态

        if timesteps is None:
            if "tau" in kwargs:
                timesteps = int(self.timesteps / kwargs["tau"])  # 如果指定了 tau 参数，缩放时间步数
            else:
                timesteps = self.timesteps  # 否则使用默认的时间步数

        # set up tracking vectors
        # 设置跟踪向量
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):

            # run and record timestep
            # 运行并记录时间步
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        运行 DMP 系统的单个时间步。

        tau float：缩放时间步长
                   提高 tau 值可加快系统执行速度
        error float：可选的系统反馈
        """

        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        # 运行正则系统
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        # 生成基函数激活
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):

            # generate the forcing term
            # 生成力项
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d]))  # 对于二维数组，计算的是两个数组的矩阵乘积
            sum_psi = np.sum(psi)
            if np.abs(sum_psi) > 1e-6:
                f /= sum_psi

            # DMP acceleration
            # DMP加速度
            self.ddy[d] = (
                self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f
            )
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy
