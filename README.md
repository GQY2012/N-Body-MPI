# N-Body-MPI
lab3 of Parallel Computing

  N体问题是指找出已知初始位置、速度和质量的多个物体在经典力学情况下的后续运动。在本次实验中，你需要模拟N个物体在二维空间中的运动情况。通过计算每两个物体之间的相互作用力，可以确定下一个时间周期内的物体位置。
在本次实验中，N个小球在均匀分布在一个正方形的二维空间中，小球在运动时没有范围限制。每个小球间会且只会受到其他小球的引力作用。为了方便起见，在计算作用力时，两个小球间的距离不会低于其半径之和，在其他的地方小球位置的移动不会受到其他小球的影响（即不会发生碰撞，挡住等情况）。你需要计算模拟一定时间后小球的分布情况，并通过MPI并行化计算过程。<br>
实验要求<br>
1.有关参数要求如下：<br>
  a)引力常数数值取6.67*10^-11<br>
  b)小球重量都为 10000kg<br>
  c)小球半径都为1cm<br>
  d)小球间的初始间隔为1cm，例：N=36时，则初始的正方形区域为5cm*5cm<br>
  e)小球初速为0.<br>
  f)对于时间间隔，公式如下<br>
  delta_t=1/timestep<br>
其中，timestep表示在1s内程序迭代的次数，小球每隔delta_t时间更新作用力，速度，位置信息。结果中程序总的迭代次数=timestep*模拟过程经历的时间，你可以根据你的硬件环境自己设置这些数值，理论上来说，时间间隔越小，模拟的真实度越高。<br>

2.你的程序中，应当实现下面三个函数:<br>
  a)compute_force()：计算每个小球受到的作用力<br>
  b)compute_velocities(): 计算每个小球的速度<br>
  c)compute_positions(): 计算每个小球的位置<br>
典型的程序中，这三个函数应该是依次调用的关系。<br>
如果你的方法中不实现这三个函数，应当在报告中明确说明，并解释你的方法为什么不需要上述函数的实现。<br>

3.报告中需要有N=64和N=256的情况下通过调整并行度计算的程序执行时间和加速比。<br>
