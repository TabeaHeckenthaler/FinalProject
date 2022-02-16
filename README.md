# FinalProject of course "Advanced Python Programming Skills"
## Weizmann 2021/2022, lecturer: Gabor Szabo
by Tabea Heckenthaler

The [main function](https://github.com/TabeaHeckenthaler/AntsShapes/blob/master/final_project_main_Gabor.py) 
is split into 2 sections: Walk in network of states, Simulation of trajectories within a piano-movers type maze.
Importantly, the second function runs 
[tests](https://github.com/TabeaHeckenthaler/AntsShapes/blob/master/PS_Search_Algorithms/Path_Planning_Rotation_students_test.py), 
that currently are not successful, as I was asked to guide 4 Master students, and they are currently still working on filling in the 
[functions](https://github.com/TabeaHeckenthaler/AntsShapes/blob/master/PS_Search_Algorithms/Path_Planning_Rotation_students.py). 
Also, the code will not be able to run on a foreign computer because it accesses large matrices stored as pickles on my 
local computer.
In the following, I describe more of the details of my project. 


## Piano-mover's Problem
I study the solution of piano-mover's type mazes. 
These mazes consist of static boundaries, through which an extended object has to be maneuvered in 2 dimensions.
The [specific maze](https://github.com/TabeaHeckenthaler/FinalProject/blob/main/Graphs/Maze_states.png) 
we constructed consists of a T shaped object in a rectangular maze with 2 exit slits. 
The object has 3 degrees of freedom (2 spacial coordinates, x and y, and orientation of its own body axis, theta).
These 3 degrees of freedom span a 3 dimensional [space](https://github.com/TabeaHeckenthaler/FinalProject/blob/main/Graphs/Large_human_SPT_states.png), 
where every pixel in space represents a configuration of the object in the maze. 
These configurations are either 'possible' (collision free, dark) or 'impossible' (colliding, translucent) maze configurations. 
Every solution of the maze, can be represented by a continuous line in the 'possible' configuration space. 

## 1. Walk in network of states
### Separation of configuration space into states
Given a set of trajectories, our aim is to analyze their flow in CS. 
For this we separate CS into different [areas](https://github.com/TabeaHeckenthaler/FinalProject/blob/main/Graphs/Large_human_SPT_transitions.png) 
by eroding the CS, and finding all connected components. Connected components are termed a 'states'.

### Network visualisation and analysis 
Based on the CS, we create a network of states: Network nodes represent states in CS, 
network edges between nodes are transitions from one state to another state in CS. 
The visualizisation of the transition matrices can be found [here](https://github.com/TabeaHeckenthaler/FinalProject/tree/main/Graphs/Results).
The visualisation of the network are in form of [.html](https://github.com/TabeaHeckenthaler/FinalProject/tree/main/Graphs/Results) 
files created using a method provided by pathpy.

## 2. Simulation of trajectories within a piano-movers type maze
I prepared a framework of a computational solver, which is supposed to resemble humans walking through the maze. 
The goal of the solver is to find the shortest path length from initial configuration to a predefined finishing 
configuration.
I will guide 4 "experimental project" students. The frame work contains a few missing functions, which are supposed to 
be filled in by the students (marked by # TODO Rotation students).
There are tests which test the functions, which the students wrote.

### Description of human like solver
The human like solver has its own 'planning space' CS' which has lower resolution than the real configuration space, 
hence CS != CS'(t=0). 
The solver plans his path according to his low resolution representation of the maze. This low resolution representation
consists of 'bins', which each contain multiple original pixels. The solver has information on the coverage of 
possible configurations within every bin. This coverage is taken as 'walking speed within the bin': High coverage 
corresponds to a high walking speed (preferential for reaching the goal fast), and low coverage corresponds to low walking speed.
The solver calculates the walking time to the end bin from every bin in his planning space. 

At every time step, the solver chooses out of all bins, which are neighbors to his current bin, the bin closest to the end bin 
(the 'greedy bin'). 
From within this 'greedy bin', he chooses a pixel randomly, which he attempts to walk to. 
If he succeeds, he updates his current pixel to the aforementioned pixel.
If he does not succeed, he decreases the speed within this chosen bin (Bayesian update, part of the job to the rotation 
students). He recalculates the walking time to the end bin from every bin in his planning 
space, and continues to the next time step. 
We will be able to tune the solver parameters such as the aforementioned resolution, and dimension of CS (2D or 3D).
