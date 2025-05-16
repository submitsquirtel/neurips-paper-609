# RobotArena

This repository hosts the codebase supporting our NeurIPS paper submission. It is organized into two primary components:


## 1. Robotic Policy Evaluation Framework

In `benchmark_eval` folder, we provide the code for systematic  evaluation of robotic policies such as **Octo, CogAct, SpatialVLA, and RoboVLM**.

## 2. Scene Placement in Simulation (Real2Sim)

In `scene_generation` folder, we provide the code for reconstructing real-world scenes such as scenes in **bridge dataset** into simulation environments. The code is designed to be modular, allowing for easy integration of new datasets and environments. The scene generation process is divided into three main components: