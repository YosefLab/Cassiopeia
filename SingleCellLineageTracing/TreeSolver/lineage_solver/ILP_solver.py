from gurobipy import *
import networkx
import time as python_time
from os import sys
import datetime



def solve_steiner_instance(model, graph, edge_variables, detailed_output=True,
						   MIPGap = .15, num_threads = 1, time_limit = -1):
    """
    Given a Steienr Tree problem instance, returns a minimum weight subgraph that satisfies the demands.

    :param graph: a directed graph with attribute 'weight' and 'label' on all edges
    :param model: a Gurobi model to be optimized
    :param edge_variables: a dictionary of gurobi edge variables
    :param detailed_output: flag which when True will print the edges in the optimal subgraph
    :param time_limit: time limit for the run in seconds
    :return: an optimal subgraph containing the path(s) if the solution is Optimal, else None
    """
    start_time = python_time.time()


    # SOLVE AND RECOVER SOLUTION

    # Percentage off of optimal (Ie 0.15 = 15% from OPT)
    model.params.MIPGap = MIPGap

    # Tuning parameters found to help speedup runtime (Do not touch)
    model.params.Threads = num_threads
    model.params.Presolve = 2
    model.params.MIPFocus = 1
    model.params.Cuts = 1
    #model.params.Method = 3
    #AW: Alex chose the non-deterministic concurrent but I chose the deterministic concurrent for reproducibility
    model.params.Method = 4

    if time_limit >= 0:
        model.params.TimeLimit = time_limit



    if detailed_output:
        print('-----------------------------------------------------------------------')
    model.optimize()

    # Recover minimal subgraph
    subgraphs = retreive_and_print_subgraph(model, graph, edge_variables, detailed_output)


    end_time = python_time.time()
    days, hours, minutes, seconds = execution_time(start_time, end_time)
    if detailed_output:
        print('Steiner tree solving took %s days, %s hours, %s minutes, %s seconds' % (days, hours, minutes, seconds))

        if model.status != GRB.status.OPTIMAL:
            print('Warning: Steiner tree solving did not result in an optimal model')

    return subgraphs


def generate_mSteiner_model(graph, source, destinations):
	"""
	Generates a Gurobi instance along with its corresponding parameters of interest to be optimized over

	:param graph: a directed graph with attribute 'weight' on all edges
	:param source: source/root node for Steiner Tree
	:param destinations: Set of terminal nodes for Steiner Tree problem
	:return: a Gurobi model pertaining to the Steiner Tree instance, and the edge_variables involved
	"""

	# Source get +len(destination) sourceflow, destinations get -1, other nodes 0
	sourceflow = {v: 0 for v in graph.nodes()}
	destinations = list(destinations)
	sourceflow[source] = len(destinations)

	if source in destinations:
		destinations.remove(source)
		sourceflow[source] -= 1

	for destination in destinations:
		sourceflow[destination] = -1


	# Create empty optimization model
	model = Model('steiner')


	# Flow for edges
	edge_variables = {}
	for u, v in graph.edges():
		edge_variables[u, v] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(destinations), name='edge_%s_%s' % (u, v))


	# 0-1 if edge was used
	edge_variables_binary = {}
	for u, v in graph.edges():
		edge_variables_binary[u, v] = model.addVar(vtype=GRB.BINARY, name='edge_binary_%s_%s' % (u, v))

	model.update()

	# CONSTRAINTS

	#Check if edge used
	for u,v in graph.edges():
		model.addConstr(edge_variables_binary[u, v] >= edge_variables[u, v] / len(destinations))
		model.addConstr(edge_variables_binary[u, v] <= edge_variables[u, v] )

	# Make sure we have a well defined tree
	for v in graph.nodes():
		model.addConstr(quicksum(edge_variables_binary[u, v] for u in graph.predecessors(v)) <= 1)


	# Flow conservation constraints
	for v in graph.nodes():
		model.addConstr(
			quicksum(edge_variables[u, v] for u in graph.predecessors(v)) +
			sourceflow[v] ==
			quicksum(edge_variables[v, w] for w in graph.successors(v))
		)




	# OBJECTIVE
	# Minimize total path weight

	objective_expression = quicksum(edge_variables_binary[u, v] * graph[u][v]['weight'] for u, v in graph.edges())
	model.setObjective(objective_expression, GRB.MINIMIZE)

	return model, edge_variables

def retreive_and_print_subgraph(model, graph, edge_variables, detailed_output):
	"""
	Extracts the optimal subgraphs associated with the Gurobi Steiner Tree model.

	:param model: an optimized gurobi model
	:param graph: a directed graph with attribute 'weight' on all edges
	:param edge_variables: a dictionary of variables corresponding to the variables d_v,w
	:param detailed_output: flag which when True will print the edges in the optimal subgraph
	:return List of most optimal subgraphs discovered while solving for the Gurobi Steiner Tree model
	"""
	# Recover minimal subgraphs
	subgraphs = []
	for i in range(0, model.SolCount):
		model.params.SolutionNumber = i
		subgraph = networkx.DiGraph()
		if True or model.status == GRB.status.OPTIMAL: #AW: I removed the restriction that requires only an optimal model
			value_for_edge = model.getAttr('xn', edge_variables)
			for u,v in graph.edges():
				if value_for_edge[u,v] > 0:
					subgraph.add_edge(u, v, weight=graph[u][v]['weight'], label=graph[u][v]['label'])

		# Print solution

			if detailed_output:
				print('Solved Steiner Tree instance. Optimal Solution costs ' + str(model.PoolObjVal))
				print('Edges in minimal subgraph:')
				print_edges_in_graph(subgraph)
			subgraphs.append(subgraph)
	return subgraphs



def execution_time(start_time, end_time):
	"""
	Returns the time of execution in days, hours, minutes, and seconds

	:param start_time: the start time of execution
	:param end_time: the end time of execution
	:return:
	"""
	execution_delta = datetime.timedelta(seconds=end_time - start_time)
	return execution_delta.days, execution_delta.seconds // 3600, (execution_delta.seconds // 60) % 60, execution_delta.seconds % 60


def print_edges_in_graph(graph, edges_per_line=5):
	"""
	Given a graph, prints all edges
	:param graph: a directed graph with attribute 'weight' on all edges
	:param edges_per_line: number of edges to print per line
	:return:
	"""
	edges_string = ''
	edges_printed_in_line = 0

	for u,v in graph.edges():
		edges_string += '%s -> %s        ' % (u, v)
		edges_printed_in_line += 1
		if edges_printed_in_line >= edges_per_line:
			edges_printed_in_line = 0
			edges_string += '\n'

	print(edges_string)
