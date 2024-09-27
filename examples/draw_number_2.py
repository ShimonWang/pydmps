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

import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

y_des = np.load("2.npz")["arr_0"].T
# print(y_des[:, 0][:, None])
# print(y_des - y_des[:, 0])
print(y_des.shape, y_des[:, 0].shape, y_des[:, 0][:, None].shape)
y_des -= y_des[:, 0][:, None]

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=False)
y_track, dy_track, ddy_track = dmp.rollout()
plt.figure(1, figsize=(6, 6))

plt.plot(y_des[0, :], y_des[1, :], "b", lw=2)
plt.plot(y_track[:, 0], y_track[:, 1], "r--", lw=2)
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
