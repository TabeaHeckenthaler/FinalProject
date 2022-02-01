# FinalProject
by Tabea Heckenthaler

Final project of the course "Advanced Python Programming Skills" in Weizmann 2021/2022, by Gabor Szabo. 

## Simulation of trajectories within a piano-movers type maze
Given is a piano-movers type maze, which can be represented in configuration space (CS). CS is a 3 dimensional boolean array where every element is either a 'possible' (collision free) or 'impossible' (colliding) maze configuration.

We program a human like solver, which has its own configurations space CS' which is initialized with some precision (importantly, CS != CS'(t=0)), and continually updated during solving time. The goal is to find the shortest path length from initial position to a predifined finish line. We will be able to play with solver parameters such as aforementioned precision, solver memory and a parameter influencing the updating of the CS after a collision. 

## Seperation of CS into states
Given a set of trajectories, our aim is to analyze their flow in CS. For this we seperate CS into different areas by eroding the CS, and finding all connected components. Connected components are termed a 'states'. 


## Network visualisation and analysis 
Based on the CS, we create a network of states: Nodes represent states in CS, edges are transitions from one state to another state in CS. 
The visualisation of the network will be in form of .html files using a method provided by pathpy.
Given the fact, that the process of state transitions is not necessarily Markovian, we analyze the paths of solvers on the higher-order-network using pathpy. 
Pathpy is an open source python package which allows analysis of time series data on networks.  

Our analysis will contain assessments 
1. whether the paths on the network can be well represented by a markovian network 
2. diffusion speed up or slow down on the network for different solver parameters and 
3. further network analytics provided by pathpy.
