# This code will only run for 1 graph database, as running with multiple graph database requires reconnection with
# new neo4j graph database. 

# If neo4j is not installed, run below:
# pip install neo4j

# Load libraries
import algorithm
from neo4j import GraphDatabase
import random
import time
import pandas as pd
from copy import deepcopy
import collections
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import argparse

def main():
    # analyse the arguments
    parser = argparse.ArgumentParser(description='neo4j info')
    parser.add_argument('uri', type=str)
    parser.add_argument('user', type=str)
    parser.add_argument('password', type=str)
    parser.add_argument('graph_name', type=str)
    args = parser.parse_args()
    uri = args.uri
    user = args.user
    password = args.password
    graph_name = args.graph_name

    driver = GraphDatabase.driver(uri, auth=(user, password))  # Neo4j driver

    # Dictionary to save result (later converted to Pandas dataframe)
    distance_dict = {'Graph Size': [], 'No. of req E': [], 'Trial': [], 'Algorithm': [], 'Distance (m)': []}
    time_dict = {'Graph Size': [], 'No. of req E': [], 'Trial': [], 'Algorithm': [], 'Time (s)': []}
    database_sizes = []

    # Run the class and assign variables
    algorithms = algorithm.dissertation(driver=driver)

    # Run all algorithms and save results in corresponding variables
    trial_list, num_edge_list, algorithm_list, distance_list, time_list = algorithms.run_all()

    # Save output of the run_all function to dictionary
    for dictionary in [distance_dict, time_dict]:
        graph_column = [graph_name] * len(trial_list)
        dictionary['Graph Size'].extend(graph_column)
        dictionary['No. of req E'].extend(num_edge_list)
        dictionary['Trial'].extend(trial_list)
        dictionary['Algorithm'].extend(algorithm_list)
    distance_dict['Distance (m)'].extend(distance_list)
    time_dict['Time (s)'].extend(time_list)
    database_sizes.append(graph_name)

    # Close the driver
    driver.close()

    # Original data
    distance_table = pd.DataFrame(distance_dict)
    distance_table.to_csv("distance_original.csv")
    time_table = pd.DataFrame(time_dict)
    time_table.to_csv("time_original.csv")

    # Total distance data (trials are averaged)
    distance_avg = distance_table.groupby(['Graph Size', 'No. of req E', 'Algorithm']).agg({'Distance (m)': 'mean'}).reset_index()
    distance_avg.rename(columns={'Distance (m)': 'Mean Distance (m)'}, inplace=True)
    distance_wide = distance_avg.pivot_table(index=['Graph Size', 'No. of req E'], columns='Algorithm', values='Mean Distance (m)', dropna=False)
    distance_wide.to_csv("distance.csv")

    # Total time data (trials are averaged) 
    time_avg = time_table.groupby(['Graph Size', 'No. of req E', 'Algorithm']).agg({'Time (s)': 'mean'}).reset_index()
    time_avg.rename(columns={'Time (s)': 'Mean Time (s)'}, inplace=True)
    time_wide = time_avg.pivot_table(index=['Graph Size', 'No. of req E'], columns='Algorithm', values='Mean Time (s)', dropna=False)
    time_wide.to_csv("time.csv")

    # Run the function for plotting graph
    algorithm.performance_plot(distance_avg, 'Mean Distance (m)', "distance_performance.pdf", database_sizes)  # For distance
    algorithm.performance_plot(time_avg, 'Mean Time (s)', 'time_performance.pdf', database_sizes)  # For time

main()