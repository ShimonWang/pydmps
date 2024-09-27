"""
Copyright (C) 2016 Travis DeWolf

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
版权 (C) 2016 Travis DeWolf

本程序为自由软件：您可以根据 GNU General Public License（GNU 通用公共许可证）发布的条款重新发布和/或修改本程序。
或对其进行修改。
自由软件基金会发布的 GNU 通用公共许可证的条款进行重新发布和/或修改。
(由您选择）任何后续版本。

发布本程序是希望它能派上用场、
但不附带任何保证；甚至不附带以下默示保证
适销性或特定用途的适用性。 参见
更多详情，请参阅 GNU 通用公共许可证。

您应该随本程序一起收到一份 GNU 通用公共许可证副本。
的副本。 如果没有，请参阅 <http://www.gnu.org/licenses/>。
"""

# 导入必要的库
import numpy as np  # 导入用于数组操作和数学计算的NumPy库
import matplotlib.pyplot as plt  # 导入Matplotlib用于绘制图形

import pydmps.dmp_discrete  # 导入pydmps库中的DMP离散系统

# 设置DMP的参数
beta = 20.0 / np.pi  # 控制障碍物避让的角度范围，通常较小的值会使避让更强
gamma = 100  # 控制避让强度的参数，值越大避让效果越强
R_halfpi = np.array(
    [
        [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ]
)  # 90度旋转矩阵，用于计算避让方向 # 旋转矩阵是2D空间的基本操作，用于旋转向量

# 设置障碍物的数量和位置
num_obstacles = 5  # 设置障碍物数量为5
obstacles = np.random.random((num_obstacles, 2)) * 2 - 1  # 在[-1, 1]范围内随机生成障碍物的位置，范围是2D空间


# 定义避免障碍物的函数
def avoid_obstacles(y, dy, goal):  # 传入当前状态位置`y`，速度`dy`，以及目标点`goal`
    p = np.zeros(2)  # 初始化偏移量`p`，用于计算避让方向

    for obstacle in obstacles:  # 遍历每个障碍物
        # based on (Hoffmann, 2009)
        # 基于（霍夫曼，2009 年）

        # if we're moving
        # 如果系统在运动中（即速度不为零）
        if np.linalg.norm(dy) > 1e-5:  # `np.linalg.norm` 计算向量的范数，判断是否有运动

            # get the angle we're heading in
            # 获取系统当前前进方向的角度
            phi_dy = -np.arctan2(dy[1], dy[0])  # `arctan2` 计算从x轴正方向开始的角度，返回弧度值
            # 根据该角度生成旋转矩阵，使后续坐标旋转到当前运动方向
            R_dy = np.array([
                [np.cos(phi_dy), -np.sin(phi_dy)],
                 [np.sin(phi_dy), np.cos(phi_dy)]
            ])

            # calculate vector to object relative to body
            # 计算障碍物相对于当前状态位置的向量
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            # 按照我们当前前进的方向旋转障碍物的向量
            obj_vec = np.dot(R_dy, obj_vec)  # `dot` 进行矩阵乘法，将障碍物的坐标变换到系统的前进方向坐标系中

            # calculate the angle of obj relative to the direction we're going
            # 计算障碍物相对于当前前进方向的角度
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            # 根据角度 phi 计算避让的角度变化量
            dphi = gamma * phi * np.exp(-beta * abs(phi))  # 避让角度变化量 dphi

            # 计算障碍物和当前运动方向的相对旋转矩阵，计算旋转后的偏移值
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))  # `outer`是外积，计算两个向量的乘积，形成矩阵
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)  # 计算避让强度并确保结果中没有NaN

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            # 如果到障碍物的距离大于到目标点的距离，忽略障碍物
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval  # 累加避让的偏移值
    return p  # 返回最终的避让结果


# test normal run
# 测试正常运行
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=10, w=np.zeros((2, 10)))
# 初始化DMP，`n_dmps`表示维度，`n_bfs`表示基函数数量，`w`为权重

y_track = np.zeros((dmp.timesteps, dmp.n_dmps))  # 初始化记录轨迹的位置
dy_track = np.zeros((dmp.timesteps, dmp.n_dmps))  # 初始化记录轨迹的速度
ddy_track = np.zeros((dmp.timesteps, dmp.n_dmps))  # 初始化记录轨迹的加速度

# 在0到2π之间生成19个目标点
goals = [[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 20)[:-1]]

# 遍历每个目标点
for goal in goals:
    dmp.goal = goal  # 设置DMP的目标点
    dmp.reset_state() # 遍历每个目标点

    for t in range(dmp.timesteps):  # 在每个时间步长中迭代
        y_track[t], dy_track[t], ddy_track[t] = dmp.step(
            external_force=avoid_obstacles(dmp.y, dmp.dy, goal)  # 每步计算避让后的偏移量并更新DMP状态
        )

    # 绘制目标、障碍物和路径
    plt.figure(1, figsize=(6, 6))  # num1 图像的数量 figsize	figure.figsize 图像的长和宽（英寸）
    (plot_goal,) = plt.plot(dmp.goal[0], dmp.goal[1], "gx", mew=3)  # 绘制目标点（绿色叉）
    for obstacle in obstacles:  # 绘制每个障碍物（红色叉）
        (plot_obs,) = plt.plot(obstacle[0], obstacle[1], "rx", mew=3)
    (plot_path,) = plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)  # 绘制路径（蓝色线）
    plt.title("DMP system - obstacle avoidance")  # 设置图标题

# 设置坐标轴比例和显示范围
plt.axis("equal")  # 保持x和y轴的比例相等
plt.xlim([-1.1, 1.1])  # 设置x轴显示范围
plt.ylim([-1.1, 1.1])  # 设置y轴显示范围
plt.show()  # 显示图像
