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

import numpy as np  # 导入 numpy，用于数值计算


# 典型动力系统类，用于实现离散或有节奏的衰减行为
class CanonicalSystem:
    """Implementation of the canonical dynamical system
    as described in Dr. Stefan Schaal's (2002) paper
    典型动力系统的实施
    如 Stefan Schaal 博士（2002 年）的论文所述"""

    # 构造函数，初始化系统的时间步长、增益项和模式类型
    def __init__(self, dt, ax=1.0, pattern="discrete"):
        """Default values from Schaal (2012)

        dt float: the timestep
        ax float: a gain term on the dynamical system
        pattern string: either 'discrete' or 'rhythmic'
        默认值取自 Schaal（2012 年）

        dt float：时间步长
        ax float：动态系统的增益项
        pattern（模式）字符串：“离散 ”或 "有节奏
        """
        self.ax = ax  # 系统的增益系数

        # 根据选择的模式，设置相应的 step 函数以及运行时长
        self.pattern = pattern
        if pattern == "discrete":
            self.step = self.step_discrete  # 如果模式为离散，使用离散的 step 函数
            self.run_time = 1.0  # 运行时长为1秒
        elif pattern == "rhythmic":
            self.step = self.step_rhythmic  # 如果模式为有节奏，使用有节奏的 step 函数
            self.run_time = 2 * np.pi  # 运行时长为 2π，即一个周期
        else:
            raise Exception(
                "Invalid pattern type specified: \
                Please specify rhythmic or discrete."
            )

        self.dt = dt  # 每次迭代的时间步长
        self.timesteps = int(self.run_time / self.dt)  # 总的迭代次数

        self.reset_state()  # 初始化系统状态

    # 模拟系统在指定时间内的行为
    def rollout(self, **kwargs):
        """Generate x for open loop movements.
        为开环运动生成 x
        """
        if "tau" in kwargs:
            timesteps = int(self.timesteps / kwargs["tau"])  # 如果指定了 tau 参数，缩放时间步数
        else:
            timesteps = self.timesteps  # 否则使用默认的时间步数
        self.x_track = np.zeros(timesteps)  # 创建一个零数组，用于记录每一步的 x 值

        self.reset_state()  # 每次模拟前重置系统状态
        for t in range(timesteps):
            self.x_track[t] = self.x  # 记录当前状态
            self.step(**kwargs)  # 更新状态

        return self.x_track  # 返回整个轨迹

    def reset_state(self):
        """Reset the system state重置系统状态"""
        self.x = 1.0

    # 离散模式下的状态更新函数，x 随时间指数衰减
    def step_discrete(self, tau=1.0, error_coupling=1.0):
        """Generate a single step of x for discrete
        (potentially closed) loop movements.
        Decaying from 1 to 0 according to dx = -ax*x.

        tau float: gain on execution time
                   increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1
        为离散（可能是闭环）运动生成单步 x
        (可能的闭环运动）
        根据 dx = -ax*x 从 1 到 0 递减。

        tau float：执行时间增益
                   增加 tau，使系统执行得更快
        error_coupling float：如果误差大于 1，则减慢速度
        """
        # 根据公式 dx = -ax * x 更新 x，tau 控制执行速度，error_coupling 用于耦合误差
        self.x += (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x

    def step_rhythmic(self, tau=1.0, error_coupling=1.0):
        """Generate a single step of x for rhythmic
        closed loop movements. Decaying from 1 to 0
        according to dx = -ax*x.

        tau float: gain on execution time
                   increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1
        """
        self.x += (1 * error_coupling * tau) * self.dt
        return self.x


# ==============================
# Test code # 测试代码部分
# ==============================
if __name__ == "__main__":

    # 创建离散模式的典型系统，时间步长为0.001秒
    cs = CanonicalSystem(dt=0.001, pattern="discrete")
    # test normal rollout
    # 进行常规系统的状态模拟
    x_track1 = cs.rollout()

    # 重置状态
    cs.reset_state()
    # test error coupling
    # 测试带有误差耦合的系统
    timesteps = int(1.0 / 0.001)  # 计算总时间步数
    x_track2 = np.zeros(timesteps)  # 初始化记录数组
    err = np.zeros(timesteps)  # 创建误差数组
    err[200:400] = 2  # 在 200 到 400 时间步内添加误差
    err_coup = 1.0 / (1 + err)  # 根据误差计算耦合项
    for i in range(timesteps):
        x_track2[i] = cs.step(error_coupling=err_coup[i])  # 每步根据误差耦合更新系统状态

    # 导入 matplotlib 库用于绘制图像
    import matplotlib.pyplot as plt

    # 创建一个图表，绘制无误差耦合和带误差耦合的系统行为
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(x_track1, lw=2)  # 绘制正常衰减轨迹
    ax1.plot(x_track2, lw=2)  # 绘制误差耦合后的轨迹
    plt.grid()  # 显示网格
    plt.legend(["normal rollout", "error coupling"])  # 添加图例
    ax2 = ax1.twinx()  # 创建双轴图，便于同时显示误差
    ax2.plot(err, "r-", lw=2)  # 绘制误差曲线
    plt.legend(["error"], loc="lower right")  # 显示误差曲线图例
    plt.ylim(0, 3.5)  # 设置 y 轴的范围
    plt.xlabel("time (s)")  # x 轴标签
    plt.ylabel("x")  # y 轴标签
    plt.title("Canonical system - discrete")  # 图表标题

    # 设置右侧 y 轴标签的颜色
    for t1 in ax2.get_yticklabels():
        t1.set_color("r")

    # 布局紧凑显示
    plt.tight_layout()
    plt.show()

    # 测试有节奏模式下的系统行为
    cs = CanonicalSystem(dt=0.001, pattern="rhythmic")
    # test normal rollout
    # 进行常规系统的状态模拟
    x_track1 = cs.rollout()

    import matplotlib.pyplot as plt

    # 再次绘制有节奏系统的状态轨迹
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(x_track1, lw=2)
    plt.grid()
    plt.legend(["normal rollout"], loc="lower right")
    plt.xlabel("time (s)")
    plt.ylabel("x")
    plt.title("Canonical system - rhythmic")
    plt.show()
