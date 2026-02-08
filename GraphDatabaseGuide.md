# Guide: How to create Road Network Graph Database

## Overview

This README file aims to explain how to create Road Network Graph Database using your own road data.

## Pre-requisites: Preparing CSV files
Make sure the CSV file name and column names are exactly as below:
- **road.csv**: This CSV file has road/edge information, with columns listed below as mandatory. If the road is bi-directional, it should have 2 rows with swapped REF_IN_ID and NREF_IN_ID.
    - id: ID of the road
    - junction_in: Start node of the road
    - junction_out: End node of the road
    - direction: Direction of the road
    - distance: Distance of the road in meters
- **junction.csv**: This CSV file has junction/node information, with columns listed below as mandatory. One row per node. 
  - id: ID of the junction
  - longitude: longitude of the junction
  - latitude: latitude of the junction

## Loading CSV files into Neo4j
1. Create new project and new database in Neo4j
2. Click the "Open folder" -> "Import" dropdown menu (often represented by three dots or a folder icon next to the "Start" button)
3. Place the `junction.csv` and `road.csv` in this folder
4. Activate the database and open the terminal
5. In the terminal, run the following code to load junctions
   ```
   LOAD CSV WITH HEADERS from 'file:///junction.csv' AS junction
   CREATE (:Junction {id:junction.id, longitude:toFloat(junction.longitude), latitude:toFloat(junction.latitude)})
   ```
6. In the terminal, run the following code to load roads
   ```
   LOAD CSV WITH HEADERS from 'file:///road.csv' AS road
   MATCH (j1:Junction {id:road.junction_in}), (j2:Junction {id:road.junction_out})
   MERGE (j1)-[r:DRIVE_TO]->(j2)
   SET r.id = road.id, r.distance = toFloat(road.distance), r.direction = road.direction
   ```

# Processing loaded data in Neo4j
In the terminal, run the following code in sequence to remove small disconnected graph:
1. ```
   CALL gds.graph.project(
   'myGraph', // The name of the graph in GDS
   'Junction', // The label of the nodes we want to include in the graph
   'DRIVE_TO' // The type of relationships we want to include in the graph
   );
   ```
2. ```
   CALL gds.scc.stream('myGraph')
   YIELD nodeId, componentId
   RETURN componentId, count(*) as size
   ORDER BY size DESC
   LIMIT 1;
   ```
3. ```
   WITH 0 AS largestComponentId
   CALL gds.scc.stream('myGraph')
   YIELD nodeId, componentId
   WITH nodeId
   WHERE componentId = largestComponentId
   MATCH (n) WHERE id(n) = nodeId
   SET n.mainComponent = true;
   ```
4. ```
   MATCH (n)
   WHERE n.mainComponent IS NULL
   DETACH DELETE n;
   ```
5. ```
   MATCH (n)
   REMOVE n.mainComponent;
   ```

## How would the created Graph Database will look like
<img width="1413" height="401" alt="example graph database" src="https://github.com/user-attachments/assets/fc735880-1fbc-4882-9226-4340cb37269b" />
