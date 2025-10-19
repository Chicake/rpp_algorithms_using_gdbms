# Load libraries
from neo4j import GraphDatabase
import random
import time
import pandas as pd
from copy import deepcopy
import collections
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

# The class storing all functions to run all algorithms

"""
Input:
driver - Neo4j driver 
num_edges (optional) - number of specified edges in list
max_it=1000 (optional) - number of iterations for Monte Carlo Method 
population_size=20 (optional) - population size for genetic algorithm
generations=100 (optional) - number of generations for genetic algorithm 
crossover_rate=0.6 (optional) - crossover rate (0-1) for genetic algorithm
mutation1_rate=0.1 (optional) - reciprocal exchange rate (0-1) for genetic algorithm 
mutation2_rate=0.1 (optional) - inversion rate (0-1) for genetic algorithm
timeout=120 (optional) - maximum time each algorithm can run for in seconds 
trials=3 (optional) - number of trials to conduct per condition for averaging out the performance
alpha1=1 (optional) - alpha value for weights used for probabilistic selection in Monte Carlo Method 
alpha2=1 (optional) - alpha value for weights used for probabilistic selection in genetic algorithm

Output:
trial_list - list of trial number 
num_edge_list - list of number of specified edges 
algorithm_list - list of name of the algorithms conducted (Cypher, NN, MC, or GA)
distance_list - list of total distance of the computed path
time_list - list of computational time taken to run the algorithm
"""

class dissertation:
    
    def __init__(self, driver, num_edges=[3], max_it=1000, population_size=20, generations=100, crossover_rate=0.6, 
                 mutation1_rate=0.1, mutation2_rate=0.1, timeout=120, trials=3, alpha1=1, alpha2=1):
        
        # Define variables
        self.driver = driver
        self.num_edges = num_edges
        self.max_it = max_it
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation1_rate = mutation1_rate
        self.mutation2_rate = mutation2_rate
        self.timeout = timeout
        self.trials = trials
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        # Create virtual graph in GDS for computing shortest path
        with driver.session() as session:
            query = """
            MATCH (source:Junction)-[r:DRIVE_TO]->(target:Junction)
            RETURN gds.graph.project(
            'shortestPathGraph',
            source,
            target,
            {
            sourceNodeProperties: source { .latitude, .longitude },
            targetNodeProperties: target { .latitude, .longitude },
            relationshipProperties: r { .distance }
            }
            )
            """
            session.run(query)
     
    
    # Randomly select specified edges and a starting node
    def specify_edges(self, num_edge):
    
        edge_list = []
        reference_dict = {}

        with self.driver.session() as session:

            # Sample edges

            # Retrieve all edges 
            edge_query = """
            MATCH (s)-[r:DRIVE_TO]->(t) 
            RETURN DISTINCT r.id AS edge_id, r.direction AS direction, r.distance AS distance, s.id AS start_id, t.id AS end_id
            """
            edge_result = session.run(edge_query)
            df = pd.DataFrame([record.data() for record in edge_result])  # Convert into dataframe
            
            # Sample edges
            sampled_df = df.sample(n=num_edge) 
            reversed_df = sampled_df[sampled_df['direction'] == 'B'].copy()  # Save two-way edges twice with reversed end and start node id
            reversed_df['start_id'], reversed_df['end_id'] = reversed_df['end_id'], reversed_df['start_id']
            final_df = pd.concat([sampled_df, reversed_df], ignore_index=True)

            # Sample starting node

            end_nodes = sampled_df['start_id'].tolist() + sampled_df['end_id'].tolist()
            end_nodes = list(set(end_nodes))  # List of end nodes of specified edges
            # Find all nodes that are not end nodes of the specified edges
            node_query = f"""
            MATCH (n) WHERE NOT n.id IN {end_nodes} 
            RETURN n.id AS node_id 
            LIMIT 1
            """
            node_result = session.run(node_query)
            start_node = node_result.single()["node_id"]  # Select top node in the returned output

            return final_df, start_node
      
    
    # Clear cache of Neo4j
    def clear_cache(self):
    
        with self.driver.session() as session:

            query = """
            CALL db.clearQueryCaches()
            """
            session.run(query)
     
    
    # A* algorithm (in GDS library) to search for shortest path between two nodes
    def a_star_search(self, start, goal):

        with self.driver.session() as session:
            query = """
            MATCH (source:Junction {id: $start}), (target:Junction {id: $goal})
            CALL gds.shortestPath.astar.stream('shortestPathGraph', {
                sourceNode: source,
                targetNode: target,
                latitudeProperty: 'latitude',
                longitudeProperty: 'longitude',
                relationshipWeightProperty: 'distance'
            })
            YIELD totalCost, nodeIds, targetNode, path
            RETURN
                totalCost,
                gds.util.asNode(targetNode).id AS targetNodeId,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).id] AS nodeList
            """
            result = session.run(query, start=start, goal=goal)
            record = result.single()
            total_cost = record["totalCost"]  # Total cost of the path
            target_node = record["targetNodeId"]  # End node
            node_list = record["nodeList"]  # List of traversed nodes
            return total_cost, target_node, node_list  
        
    
    # A* algorithm (in GDS library) to create an edge in the graph that represents shortest path between two nodes
    def a_star_create(self, start, goal):
        
        with self.driver.session() as session:
            query = """
            MATCH (source:Junction {id: $start}), (target:Junction {id: $goal})
            CALL gds.shortestPath.astar.write('shortestPathGraph', {
                sourceNode: source,
                targetNode: target,
                latitudeProperty: 'latitude',
                longitudeProperty: 'longitude',
                relationshipWeightProperty: 'distance',
                writeRelationshipType: 'SHORTEST_PATH',
                writeNodeIds: true
            })
            YIELD relationshipsWritten
            RETURN relationshipsWritten
            """
            session.run(query, start=start, goal=goal)
            
    
    # Function to conduct cypher algorihtm
    def cypher_algorithm(self, specified_edges, start_node_id):
        
        # Retrieve all specified edges and its start and end nodes
        start_nodes_list = specified_edges['start_id'].tolist()
        start_nodes_list = list(set(start_nodes_list))
        end_nodes_list = specified_edges['end_id'].tolist()
        end_nodes_list = list(set(end_nodes_list))
        required_edges = specified_edges['edge_id'].tolist()
        required_edges = list(set(required_edges))

        self.clear_cache()  # Clear cache
        start_time = time.time()  # Start timer

        # Graph Transformation - create shortest paths between all end nodes of specified edges
        for start_node in start_nodes_list:
            self.a_star_create(start_node_id, start_node)
            for end_node in end_nodes_list:
                self.a_star_create(end_node, start_node)

        # Add SHORTEST_PATH edge to specified edges too for easier query of matching path
        query_add_edges = """
        MATCH (s)-[r:DRIVE_TO]->(e)
        WHERE s.id = $start AND r.id = $road AND e.id = $end
        MERGE (s)-[p:SHORTEST_PATH]->(e)
        SET p.id = r.id, p.totalCost = r.distance, p.nodeIds = r.id
        """
        for index, row in specified_edges.iterrows():
            with self.driver.session() as session:
                session.run(query_add_edges, start=row['start_id'], road=row['edge_id'], end=row['end_id'])

        # Cypher query for the main cypher algorithm
        num_edges = len(required_edges)*2
        query = f"""
        MATCH path = (a) -[r:SHORTEST_PATH*{num_edges}]-> ()
        WHERE a.id = '{start_node_id}' AND
        all(relName IN {required_edges} WHERE any(rel IN relationships(path) WHERE rel.id = relName)) 
        WITH path, REDUCE(totalWeight = 0, r IN relationships(path) | totalWeight + toFloat(r.totalCost)) AS pathWeight
        ORDER BY pathWeight ASC
        LIMIT 1
        RETURN [node IN nodes(path) | node.nodeIds] AS nodeList, pathWeight
        """
        with self.driver.session() as session:
            result = session.run(query)
            path = result.single()
        
        end_time = time.time()  # End timer

        total_time = end_time - start_time  # Get total time

        if path == None:  # Safety precaution if no path is found
            return None, total_time, None

        else:  # Return total cost, total time taken and list of traversed nodes
            total_cost = path['pathWeight']
            node_list = path['nodeList']
            return total_cost, total_time, node_list
        
    
    # Function to conduct nearest neighbour algorithm
    def nearest_neighbour(self, specified_edges, start_node_id):

        # Initialise conditions and create storage for result
        edges_list = specified_edges
        start_node = start_node_id
        best_path = []
        cost_list = []

        self.clear_cache()  # Clear cache
        start_time = time.time()  # Start timer

        # Keep finding nearest neighbour until all specified edges are visited
        while len(edges_list.index) > 0:

            # Initialise conditions and create storage for result
            cost_reference_dict = {}
            nodes_list = edges_list['start_id'].tolist()
            nodes_list = list(set(nodes_list))
            same_alert = False

            # Find shortest path cost between start node and all unvisited end nodes
            for node in nodes_list:
                total_cost, target_node, node_list = self.a_star_search(start_node, node)
                # Condition "Same": If an end node of a specified edge is also an end node of another specified edge, remain at the starting node
                if total_cost == 0:  
                    selected = [target_node, node_list, total_cost]
                    same_alert = True
                    break
                # If not, save the cost 
                cost_reference_dict[total_cost] = [target_node, node_list]

            # If Condition "Same", choose a connected specified edge 
            if same_alert is True:
                selected_edge = edges_list[edges_list['start_id'] == selected[0]]

                if len(selected_edge.index) > 1:  # In case there are multiple specified edges satisfying the condition
                    selected_edge = selected_edge.sample(n=1)

                best_path.extend([selected[0]])  # Add the traversed node

            # If it is not Condition "Same" 
            else: 
                min_cost = min(cost_reference_dict.keys())  # Choose a shortest path with minimum cost 
                cost_list.append(min_cost)
                chosen_node_id = cost_reference_dict[min_cost][0]
                selected_edge = edges_list[edges_list['start_id'] == chosen_node_id]  # Choose specified edge starting from the end node of the chosen shortest path

                if len(selected_edge.index) > 1:  # In case there are multiple specified edges satisfying the condition
                    selected_edge = selected_edge.sample(n=1)

                best_path.extend(cost_reference_dict[min_cost][1])  # Add the traversed node

            cost_list.append(selected_edge['distance'].values[0])  # Save cost of the chosen specified edge
            start_node = selected_edge['end_id'].values[0]  # Alter starting node
            selected_edge_id = selected_edge['edge_id'].values[0]
            edges_list = edges_list[edges_list['edge_id'] != selected_edge_id]  # Remove selected specified edge

        end_time = time.time()  # End timer

        total_time = end_time - start_time  # Get total time

        return sum(cost_list), total_time, best_path  # Return total cost of the path, total time taken, and list of traversed nodes

    
    # Function to conduct monte calro method
    def monte_carlo(self, specified_edges, start_node_id):

        # Initialise conditions and create storage for result
        best_path = None
        best_cost = 10000000000000
        start_nodes_list = specified_edges['start_id'].tolist()
        start_nodes_list = list(set(start_nodes_list))
        end_nodes_list = specified_edges['end_id'].tolist()
        end_nodes_list = list(set(end_nodes_list))
        possible_edges = specified_edges.copy(deep=False)
        possible_edges['node_list'] = ""

        self.clear_cache()  # Clear cache of Neo4j
        start_time = time.time()  # Start timer

        # Graph transformation - create shortest paths between all end nodes of specified edges
        for start_node in start_nodes_list:
            total_cost, _, node_list = self.a_star_search(start_node_id, start_node)  # Find shortest path between the starting node and all end nodes
            new_edge = {
                    'edge_id': 'SHORTEST_PATH',
                    'direction': "NA",
                    'distance': total_cost,
                    'start_id': start_node_id,
                    'end_id': start_node,
                    'node_list': node_list
                }
            possible_edges = pd.concat([possible_edges, pd.DataFrame([new_edge])], ignore_index=True)  # Store it as pd DF

            for end_node in end_nodes_list:
                total_cost, _, node_list = self.a_star_search(end_node, start_node)  # Find shortest path between end nodes
                new_edge = {
                    'edge_id': 'SHORTEST_PATH',
                    'direction': "NA",
                    'distance': total_cost,
                    'start_id': end_node,
                    'end_id': start_node,
                    'node_list': node_list
                }
                possible_edges = pd.concat([possible_edges, pd.DataFrame([new_edge])], ignore_index=True)  # Store it as pd DF     

        # Monte Carlo Method starts here
        
        # Repeat for specific number of iterations
        for i in range(self.max_it):

            # Initialise conditions and create storage for result
            cost_list = []
            path = []
            edges_list = specified_edges
            start_node = start_node_id
            shortest_paths = possible_edges.copy()

            # Continue creating path until all specified edges are visited
            while len(edges_list.index) > 0:

                # Initialise conditions
                nodes_list = edges_list['start_id'].tolist()
                nodes_list = list(set(nodes_list))
                same_alert = False
                shortest_paths = shortest_paths[shortest_paths['end_id'].isin(edges_list['start_id'])]
                edge_choices = shortest_paths[shortest_paths['start_id'] == start_node]

                # If it is Condition "Same" (specified edge is directly connected to the starting node)
                if not edge_choices[edge_choices['distance'] == 0].empty:
                    shortest_path = edge_choices[edge_choices['distance'] == 0]  # Choose the shortest path with distance 0 (direcly connected)

                    if len(shortest_path.index) > 1:  # In case there are multiple shortest paths satisfying the condition
                         shortest_path = shortest_path.sample(n=1)
                            
                    path.extend([start_node])  # Add traversed node

                # If it is not Condition "Same"
                else:
                    # Calculate probability (weights) of each shortest path and choose one randomly based on the probability
                    edge_choices = edge_choices.copy()
                    edge_choices.loc[:, 'probability'] = edge_choices['distance'] ** -self.alpha1
                    shortest_path = edge_choices.sample(n=1, weights=edge_choices['probability'])

                    cost_list.append(shortest_path['distance'].values[0])  # Add cost of the chosen shortest path
                    path.extend(shortest_path['node_list'].values[0])  # Add traversed nodes of the chosen shortest path

                selected_edge = edges_list[edges_list['start_id'] == shortest_path['end_id'].values[0]]  # Select specified edge starting from the end node of chosen shortest path

                if len(selected_edge.index) > 1:  # In case there are multiple specified edges satisfying the condition
                    selected_edge = selected_edge.sample(n=1)

                cost_list.append(selected_edge['distance'].values[0])  # Add cost of chosen specified edge

                start_node = selected_edge['end_id'].values[0]  # Alter starting node
                selected_edge_id = selected_edge['edge_id'].values[0]
                edges_list = edges_list[edges_list['edge_id'] != selected_edge_id]  # Remove selected specified edge

            total_distance = sum(cost_list)  # Calculate total distance of the path

            # Save the total distance and traversed nodes if it is shortest out of all iterations
            if total_distance < best_cost:
                best_cost = total_distance
                best_path = path

        end_time = time.time()  # End timer

        total_time = end_time - start_time  # Get total time

        return best_cost, total_time, best_path  # Return total cost of the path, total time taken, and traversed nodes
    
    
    # For genetic algorithm: Chooses one of the two possible shortest path 
    def choose_shortest_path(self, possible_edges, start_node_id, this_edge):

        # Find the specified edge (this_edge)
        filtered = possible_edges[possible_edges['edge_id'] == this_edge]

        # If the specified edge is one-way, find path that starts from the end node of shortest path and end with the start node of specified edge
        if len(filtered) == 1:
            end_node_id = filtered['start_id'].values[0]
            edge_choice = possible_edges.loc[(possible_edges['start_id'] == start_node_id) & 
                                             (possible_edges['end_id'] == end_node_id)]

        # If the specified edge is two-way, find 2 possible shortest paths (one per end node)
        elif len(filtered) > 1:
            first_row = filtered.iloc[0]
            nodes_pair = [first_row['start_id'], first_row['end_id']]
            edges_choices = possible_edges.loc[(possible_edges['start_id'] == start_node_id) & 
                                               (possible_edges['end_id'].isin(nodes_pair))]

            # Randomly select one shortest path and select the direction of the specified edge
            edge_choice = edges_choices.sample(n=1, weights='distance')
            end_node = edge_choice['end_id'].values[0]
            filtered = filtered[filtered['start_id'] == end_node]

        distance = float(edge_choice['distance'].values[0]) + float(filtered['distance'].values[0])  # Total distance of chosen shortest path and specified edge
        node_list = edge_choice['node_list'].values[0]  # List of traversed nodes
        next_node = filtered['end_id'].values[0]  # End node of chosen specified edge

        return distance, node_list, next_node

    
    # For genetic algorithm: Evaluates the fitness score of the given chromosome
    def score_chromosome(self, chromosome, possible_edges, start_node_id):

        # Initialise conditions and create storage
        cost = 0
        shortest_path_nodes = []
        start_node = start_node_id

        # For each specified edge id in the chromosome
        for index in range(len(chromosome)):

            #choose shortest path to move onto next specified edge id
            this_edge = chromosome[index]
            distance, node_list, next_node = self.choose_shortest_path(possible_edges, start_node, this_edge)

            # Save cost (and traversed nodes) and set starting node
            cost += distance
            start_node = next_node
            shortest_path_nodes.extend(node_list)    
        shortest_path_nodes.append(start_node)

        return 1/(cost ** self.alpha2), cost, shortest_path_nodes  # Return probability, cost of the path, and list of traversed nodes

    
    # For genetic algorithm: Partially mapped crossover function
    def pmx(self, parentA, parentB):

        # Choose 2 positions to cutoff
        assert len(parentA) == len(parentB)
        positions = random.sample(range(len(parentA)), 2)
        cutoff_1 = min(positions[0], positions[1])
        cutoff_2 = max(positions[0], positions[1])

        # Inner function to create children
        def offspring(p1, p2):

            # Initialise condition
            p1 = np.array(p1)
            p2 = np.array(p2)
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

            # Copy the rest from parent 2
            for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
                candidate = p2[i]
                # Keep replacing duplicated genes until there is no duplicate
                while candidate in p1[cutoff_1:cutoff_2]: 
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[i] = candidate

            return list(offspring)  # Return child chromosome

        childA = offspring(parentA, parentB)
        childB = offspring(parentA, parentB)

        return childA, childB

    
    # (For genetic algorithm) mutation 1: reciprocal exchange - modifies chromosome in place by exchanging two locations in the chromosome
    def reciprocal_exchange(self, chromosome):
        # get two distinct positions
        positions = random.sample(range(len(chromosome)), 2)
        # swap the elements at those positions
        chromosome[positions[0]], chromosome[positions[1]] = chromosome[positions[1]], chromosome[positions[0]]
        return chromosome

    
    # (For genetic algorithm) mutation 2: inversion - modifies chromosome in place by inverting a subsequence of the chromosome
    def inversion(self, chromosome):
        # get two distinct positions
        positions = random.sample(range(len(chromosome)), 2)
        inversion_point1 = min(positions[0], positions[1])
        inversion_point2 = max(positions[0], positions[1])
        # reverse genes between those positions
        chromosome[inversion_point1:inversion_point2] = chromosome[inversion_point1:inversion_point2][::-1]
     
    
    # Function to conduct genetic algorithm
    def genetic_algorithm(self, specified_edges, start_node_id):

        # Initialise condition and create storage
        start_nodes_list = specified_edges['start_id'].tolist()
        start_nodes_list = list(set(start_nodes_list))
        end_nodes_list = specified_edges['end_id'].tolist()
        end_nodes_list = list(set(end_nodes_list))
        required_edges = specified_edges['edge_id'].tolist()
        required_edges = list(set(required_edges))
        possible_edges = specified_edges.copy()
        possible_edges['node_list'] = possible_edges.apply(lambda row: [row['start_id'], row['end_id']], axis=1)

        best_chromosome = None
        best_chromosome_score = -10000000
        best_path = None
        best_distance = None

        self.clear_cache()  # Clear cache
        start_time = time.time()  # Start timer

        # Graph transformation - create shortest paths between all end nodes of specified edges
        for start_node in start_nodes_list:

            total_cost, _, node_list = self.a_star_search(start_node_id, start_node)   # Find shortest path between starting node and end nodes
            new_edge = {
                    'edge_id': 'SHORTEST_PATH',
                    'direction': "NA",
                    'distance': total_cost,
                    'start_id': start_node_id,
                    'end_id': start_node,
                    'node_list': node_list
                }
            possible_edges = pd.concat([possible_edges, pd.DataFrame([new_edge])], ignore_index=True)  # Store it as pd DF

            for end_node in end_nodes_list:

                total_cost, _, node_list = self.a_star_search(end_node, start_node)  # Find shortest path between end nodes
                new_edge = {
                    'edge_id': 'SHORTEST_PATH',
                    'direction': "NA",
                    'distance': total_cost,
                    'start_id': end_node,
                    'end_id': start_node,
                    'node_list': node_list
                }
                possible_edges = pd.concat([possible_edges, pd.DataFrame([new_edge])], ignore_index=True)  # Store it as pd DF

        # Conduct genetic algorithm from below

        # Create initial population by randomly ordering specified edges
        solution_pool = []
        for i in range(0, self.population_size):
            starting_solution = random.sample(required_edges, len(required_edges))
            solution_pool.append(starting_solution)

        # Evolutionary process starts here and repeats for specified number of generations
        for i in range(self.generations):
            
            # PMX
            
            # Randomly choose 2 parents
            crossover_parent_count = int(self.crossover_rate * self.population_size)
            if crossover_parent_count % 2 == 1:
                crossover_parent_count -= 1 # ensure even number of parents
            crossover_parents = random.sample(solution_pool, crossover_parent_count)
            idx = 0
            # Iterate over pairs of parents and get children
            while idx < len(crossover_parents):
                parent_a = crossover_parents[idx]
                parent_b = crossover_parents[idx+1]
                child_a, child_b = self.pmx(parent_a, parent_b)
                solution_pool.extend([child_a, child_b])
                idx += 2

            # mutation
            for chromosome in solution_pool:
                if random.random() < self.mutation1_rate:  # mutation 1
                    self.reciprocal_exchange(chromosome) 
                if random.random() < self.mutation2_rate:  # mutation 2
                    self.inversion(chromosome)

            # selection of new generation
            
            # Calculate fitness score
            fitness_sum = 0
            score_list = []
            for chromosome in solution_pool:
                fitness_score, total_distance, shortest_path_nodes = self.score_chromosome(chromosome, possible_edges, start_node_id)
                if i + 1 == self.generations:  # If it is last generation, store best scoring chromosome and its information
                    if fitness_score > best_chromosome_score:
                        best_chromosome = chromosome
                        best_chromosome_score = fitness_score
                        best_path = shortest_path_nodes
                        best_distance = total_distance
                else:  # Else, save its fitness score
                    score_list.append(fitness_score)
                    fitness_sum += fitness_score

            # If it is not the last generation, randomly select from current solution pool based on fitness score
            if i + 1 != self.generations:
                new_generation = []
                
                for idx in range(0, self.population_size):
                    selected = random.choices(solution_pool, weights=score_list, k=1)[0]
                    new_generation.append(selected)
                    index = solution_pool.index(selected)
                    solution_pool.pop(index)
                    score_list.pop(index)

                solution_pool = new_generation  # Solution pool for next generation created

        end_time = time.time()  # End timer
        total_time = end_time - start_time  # Get total time

        return best_distance, total_time, best_path  # Return total distance of the path, total time, and traversed nodes
    
    
    # Apply timeout to algorithms
    def run_with_timeout(self, fn, args, timeout):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fn, *args)
            try:
                # Wait for the result with a timeout
                total_cost, total_time, node_list = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # If a timeout occurs, cancel the future and return None
                future.cancel()
                total_cost, total_time, node_list = None, None, None
            return total_cost, total_time, node_list
        
    
    # Function which run all algorithms in one go and return the following in this order:
    # trial number, number of specified edges, algorithm name, total distance, and total time
    def run_all(self):
        
        # Create storage for results
        trial_list = []
        num_edge_list = []
        algorithm_list = []
        distance_list = []
        time_list = []
        
        # For each number of specified edges lised
        for num_edge in self.num_edges:
            
            print(f"--------Required edges: {num_edge}--------")
            
            # For each trial
            for i in range(self.trials):
                
                print(f"Running Trial {i+1}")
                trial_list.extend([i, i, i, i])  # Add trial number
                specified_edges, start_node = self.specify_edges(num_edge)  # Select edges and starting node
                num_edge_list.extend([num_edge, num_edge, num_edge, num_edge])  # Add number of specified edges
                
                # Ran each of the algorithms with message when it finish running
                cypher_total_cost, cypher_total_time, cypher_node_list = self.run_with_timeout(self.cypher_algorithm, [specified_edges, start_node], self.timeout)
                with self.driver.session() as session:  # Delete SHORTEST_PATH edges created in cypher algorithm
                    query = """
                    MATCH ()-[r:SHORTEST_PATH]->()
                    DELETE r
                    """
                    session.run(query)
                print("Ran Cypher")
                nn_total_cost, nn_total_time, nn_node_list = self.run_with_timeout(self.nearest_neighbour, [specified_edges, start_node], self.timeout)
                print("Ran NN")
                mc_total_cost, mc_total_time, mc_node_list = self.run_with_timeout(self.monte_carlo, [specified_edges, start_node], self.timeout)
                print("Ran MC")
                ga_total_cost, ga_total_time, ga_node_list = self.run_with_timeout(self.genetic_algorithm, [specified_edges,start_node], self.timeout)
                print("Ran GA")
                
                # Convert the total distance of the path to meters
                if cypher_total_cost != None:
                    cypher_total_cost = cypher_total_cost*1852
                if nn_total_cost != None:
                    nn_total_cost = nn_total_cost*1852
                if mc_total_cost != None:
                    mc_total_cost = mc_total_cost*1852
                if ga_total_cost != None:
                    ga_total_cost = ga_total_cost*1852
                    
                # Add result of the algorithms
                algorithm_list.extend(["CA", "NN", "MC", "GA"])
                distance_list.extend([cypher_total_cost, nn_total_cost, mc_total_cost, ga_total_cost])
                time_list.extend([cypher_total_time, nn_total_time, mc_total_time, ga_total_time])
                
        return trial_list, num_edge_list, algorithm_list, distance_list, time_list       

# Function to plot performance graph
def performance_plot(df, y_var, file_name, database_sizes):

    # Set up conditions
    algorithms = ["NN", "MC", "GA", "Cypher"]
    markertype = ['o', 'v', '+', 'x']
    markersize = [5, 5, 10, 10]
    linetype = ['solid', 'dashed', 'dotted', 'dashdot']
    label = ["NN", "MC", "GA", "CA"]
    # Create combination of algorithm and number of specified edges for different lines in a line plot
    lines_data = {}
    for algo in algorithms:
        lines_data[algo] = {}
        for size in database_sizes:
            subset = df[(df['Algorithm'] == algo) & (df['Graph Size'] == size)]
            lines_data[algo][size] = subset.set_index('No. of req E')[y_var]

    # Set up the figure container for corresponding number of graph databases
    if len(database_sizes) == 1:
        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    elif len(database_sizes) == 2:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    elif len(database_sizes) == 3:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    
    # Plot line graph for first graph database
    for i in range(4):  # For each algorithm
        ax0.plot(lines_data[algorithms[i]][database_sizes[0]].index, lines_data[algorithms[i]][database_sizes[0]].values, 
                 label=label[i], marker=markertype[i], markersize=markersize[i], linestyle=linetype[i])
    ax0.set_title(f'G({database_sizes[0]})')

    # Plot line graph for second graph database (if 2 or more graph databases)
    if len(database_sizes) > 1:
        for i in range(4):
            ax1.plot(lines_data[algorithms[i]][database_sizes[1]].index, lines_data[algorithms[i]][database_sizes[1]].values, 
                     label=label[i], marker=markertype[i], markersize=markersize[i], linestyle=linetype[i])
        ax1.set_title(f'G({database_sizes[1]})')

    # Plot line graph for third graph database (if 3 graph databases)
    if len(database_sizes) > 2:
        for i in range(4):
            ax2.plot(lines_data[algorithms[i]][database_sizes[2]].index, lines_data[algorithms[i]][database_sizes[2]].values, 
                     label=label[i], marker=markertype[i], markersize=markersize[i], linestyle=linetype[i])
        ax2.set_title(f'G({database_sizes[2]})')

    # Label the graph
    fig.supxlabel('Number of specified edges')
    fig.supylabel(y_var)
    
    # Create legend
    lines, labels = ax0.get_legend_handles_labels()
    ordered_lines = [lines[3], lines[0], lines[1], lines[2]]  # Adjust index according to your actual order
    ordered_labels = ["CA", "NN", "MC", "GA"]
    fig.legend(ordered_lines, ordered_labels, loc='upper right')

    # Format the graph
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 

    # Save and display the graph
    plt.savefig(file_name, format='pdf')
    plt.show()