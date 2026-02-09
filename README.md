# Codes: Optimizing Map Feature Collection Routing Using Graph Database Technology

## Overview

Note: This is the coding implementation for the study "Optimizing Map Feature Collection Routing Using Graph Database Technology" by Kaede Hasegawa and Antonis Bikakis.

Accurate digital maps are essential for meeting transportation and logistics demands efficiently, necessitating frequent updates to road features. This raises the challenge of determining an optimal path that visits all roads requiring data collection, known as the Rural Postman Problem (RPP). Traditionally, this problem has been addressed using heuristics and metaheuristics with relational databases. However, graph databases, which can better represent and store relationships in network structure, may offer advantages in solving RPP. Therefore, this study investigates the novel application of graph databases and their query language for developing optimisation algorithms to solve the RPP. Three existing algorithms—(a) Nearest Neighbour, (b) Monte Carlo, and (c) Genetic Algorithm—were implemented using Cypher queries (e.g., A* shortest path algorithm) and compared with (d) a Cypher-only algorithm designed to compute the optimal path.

This repository aims to provide the full coding implementation of the RPP Algorithms. The dataset used in the study is provided by HERE Technologies, which is protected by intellectual property right, so the dataset cannot be provided. This can run if you have your own road network graph database on Neo4j; this repository is more for explanation purposes rather than testing. 

## Files Description
1. **algorithm.py**: This file includes all RPP algorithm functions. It also has functions to store and plot performance data.

2. **main.py**: This file is where all functions in `algorithm.py` is used to enable algorithms to run from terminal. 

3. **requirements.txt**: This file contains all library used in main.py, so you can pip install.
  
4. **GraphDatabaseGuide.md**: This file explains how to make the road network graph database using your own road data
Note: Data used in the paper cannot be published as the data belongs to HERE Technologies and is protected by Intellectual Property Rights.

## Setup Instructions
1. **Python Environment**: Ensure Python is installed. Required libraries include pandas, numpy, matplotlib, and neo4j. Install them using:
   ```
   pip install -r requirements.txt
   ```

2. **Data Preparation**: Open Neo4j with your created Road Network Graph Database running. You can create your own Road Network Graph Database by referring to `GraphDatabaseGuide.md`.
(The sample dataset is from OpenStreetMap, which is an open data licensed under the Open Data Commons Open Database License (ODbL) by the OpenStreetMapFoundation (OSMF). Regarding licensing, please refer to the link: https://www.openstreetmap.org/copyright)

## Usage Guide 
   - Open the terminal in the project directory.
   - Run `main.py` with the necessary arguments (see example below): uri username password graph_name 
      ```
      python main.py bolt://localhost:7690 neo4j password1 graph1
      ```
      
## Expected Outputs
The above usage guide allows you to run all algorithms with 3 specified edges (takes the least amount of time). You can change other variables of algorithms in line 39 and 40 in `algorithm.py`. If you want to run the algorithm with another graph database, you need to open the graph database on neo4j and conduct usage guide again.
