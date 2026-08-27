# RL 版 Hello World：N 级倒立摆

## Basic Intro

- 使用 PPO 算法， actor 和 critic 是两个完全无关的 Net 。
- 网络直接接收当前小车位置、摆杆角度和角速度，不再堆叠历史帧。
- 调参不说了。
- TQC checkpoint 会保存网络、优化器、经验回放池、环境进度和随机数状态，可直接续训。

## __Version 1 Pole__

- 可以在从任意角度回归到两个 0 阶量全 0 ，符合最好的想象。

## __Version 2 Poles__

- 目前做不到从任意角度回归，仅能做到从小的 0 阶扰动回归。
