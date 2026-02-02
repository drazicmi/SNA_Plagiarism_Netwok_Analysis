import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

filePath = 'data-set/Results 1_anonymized.csv'

# SECTION_1: Dataset analysis

# Load dataset and display data
data = pd.read_csv(filePath)
print(data)

# Check for empty fields
print(f'Are there empty fields?\n{data.isna().any()}')

# Calculate number of students
unique_students_df = pd.concat([data['acter1'], data['acter2']]).to_frame().drop_duplicates()
num_of_students = unique_students_df.shape[0]
print(f'Total number of students: {num_of_students}')

# Calculate occurrences for each student
print(f'Are student occurrences in pairs unique? Answer: { data["acter1"].is_unique }')  # False

student_Occurrence = pd.concat([data['acter1'], data['acter2']]).to_frame('acter')

student_Occurrence = student_Occurrence.groupby('acter')
# Note that agg(np.size) and size() do the same, in future callables such as np.size will be replaced with function calls
print(student_Occurrence['acter'].agg(np.size).sort_values(ascending=False))

# Calculate total number of similarity lines
student1_lines = data[['acter1', 'lines_matched']].rename(columns={'acter1': 'acter'})
student2_lines = data[['acter2', 'lines_matched']].rename(columns={'acter2': 'acter'})

student_similarity = pd.concat([student1_lines[['acter', 'lines_matched']], student2_lines[['acter', 'lines_matched']]])
student_similarity = student_similarity.groupby('acter')
# Avoided agg(np.sum) and swaped it for sum() because of the runtime warning
print(student_similarity.sum().sort_values(by='lines_matched', ascending=False))


# SECTION_2 : Research questions

"""
    Start by creating a graph using networkX.
    Graph is directed. Edges are weighted using actor's percentage in a pair.
    Nodes represent students, Edges represent connection between two students.
    If a pair exists, we create a edge directed from actor1 to actor2 and edge from actor2 to actor1.
    Since the pairs are 
"""


graph = nx.DiGraph()

# First add each student as a node (44 unique students)
graph.add_nodes_from(unique_students_df[0])

# Then proceed by adding edges in a already described way
for _, row in data.iterrows():
    graph.add_edge(row['acter1'], row['acter2'], weight=row['acter1_percentage'])
    graph.add_edge(row['acter2'], row['acter1'], weight=row['acter2_percentage'])

print(f'Number of nodes in graph: {graph.number_of_nodes()}')
print(f'Number of edges in graph: {graph.number_of_edges()}')

# print(graph) -> Graph with 44 nodes and 250 edges

# Question 1
network_density = nx.density(graph)
print(f"\n--- Question 1 Results ---")
print(f'Network density: {network_density}\n')

# Question 2
# Calculate average_shortest_path_length and network diameter

# Question 3
# We know graph isn't fully connected, because of the network density (it's not 1)
# We can calculate number of connections and the length for each component

# 1. Get all Weakly Connected Components (WCC)
wcc = list(nx.weakly_connected_components(graph))
wcc_sizes = [len(c) for c in wcc]
wcc_sizes.sort(reverse=True)

# Note: From weakly connected components we can see there is only one component, and it contains all the nodes
# so special analysis and search for giant component isn't necessary, but we have done it for learning purposes

# 2. Identify the Giant Component
giant_component_nodes = max(nx.weakly_connected_components(graph), key=len)
G_giant = graph.subgraph(giant_component_nodes)

# 3. Calculate Q2 Metrics (Distance & Diameter) on the Giant Component
avg_dist = nx.average_shortest_path_length(G_giant)
diameter = nx.diameter(G_giant)


# 4. Calculate Centralization (Degree Centralization)
def degree_centralization(graph):
    degrees = dict(graph.degree())
    max_deg = max(degrees.values())
    n = len(graph.nodes())
    denominator = (n - 1) * (n - 2)
    if denominator <= 0: return 0
    sum_diff = sum(max_deg - d for d in degrees.values())
    return sum_diff / denominator


centralization = degree_centralization(graph)

print(f"--- Question 3 Results ---")
print(f"Number of Components: {len(wcc)}")
print(f"Component Sizes: {wcc_sizes}")
print(f"Is there a Giant Component? {'Yes' if wcc_sizes[0] > (44/2) else 'No'}")
print(f"Network Centralization: {centralization:.4f}")

print(f"\n--- Question 2 Results (based on Giant Component) ---")
print(f"Average Distance: {avg_dist:.4f}")
print(f"Network Diameter: {diameter}")

# Question 4
# What is the average and what is the global clustering coefficient of the network?
# What is the distribution of the local clustering coefficient of its nodes?
# Is the clustering pronounced or not?
# The answer is given by comparison with randomly generated Erdos-Renyi and scale free networks of the same dimensions.

# Question 5
# By calculating relevant network metrics, formally explain whether the network exhibits small-world properties.

# 1. Calculate for your Plagiarism Graph
avg_clust_real = nx.average_clustering(graph)
global_clust_real = nx.transitivity(graph)
local_clust_values = list(nx.clustering(graph).values())

# 2. Generate Erdős-Rényi (Random)
# Probability p = edges / possible_edges
n = graph.number_of_nodes()
e = graph.number_of_edges()
p = e / (n * (n - 1))
random_graph = nx.fast_gnp_random_graph(n, p, directed=True)
avg_clust_rand = nx.average_clustering(random_graph)

# 3. Generate Scale-Free (Barabási–Albert)
# Note: BA is undirected by default, we convert to directed for comparison
m = round(e / n)  # Average edges per new node
scale_free_graph = nx.barabasi_albert_graph(n, m).to_directed()
avg_clust_sf = nx.average_clustering(scale_free_graph)

print(f"\n--- Question 4 and 5 Results ---")
print(f"Real Graph - Avg Clustering: {avg_clust_real:.4f}, Global: {global_clust_real:.4f}")
print(f"Random Graph - Avg Clustering: {avg_clust_rand:.4f}")
print(f"Scale-Free Graph - Avg Clustering: {avg_clust_sf:.4f}")

# print(f"Local clust values: {local_clust_values}")
# 4. Plotting the Local Distribution
plt.hist(local_clust_values, bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Local Clustering Coefficients")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.savefig(os.path.join("pictures_from_code", "DZ1NetDistribution of Local Clustering Coefficients.png"), bbox_inches='tight')
plt.close()

# Question 6
# Perform an assortative analysis by node degree and give an answer, whether, and to what extent assortative mixing is expressed.
# Note that node degree is a numerical and not a categorical variable and choose an adequate metric for the measure of assortative mixing.
# Attach the visualization.

# 1. Calculate the Degree Assortativity Coefficient (r)
# This measures the correlation between the degrees of nodes at either end of an edge.
assortativity_r = nx.degree_assortativity_coefficient(graph)

# 2. Visualization: Average Neighbor Degree vs. Node Degree
# This scatter plot is the standard way to visualize assortative mixing.
degrees = dict(graph.degree())
avg_neighbor_degrees = nx.average_neighbor_degree(graph)

x = [degrees[n] for n in graph.nodes()]
y = [avg_neighbor_degrees[n] for n in graph.nodes()]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, c='royalblue', edgecolors='k')
# Add a trend line to visualize the correlation
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, label=f"Trend (r={assortativity_r:.4f})")
plt.title("Assortative Mixing by Node Degree")
plt.xlabel("Node Degree (k)")
plt.ylabel("Average Neighbor Degree (knn)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join("pictures_from_code", "DZ1NetAssortativity_Degree_Analysis.png"), bbox_inches='tight')
plt.close()

print(f"\n--- Question 6 Results ---")
print(f"Assortativity Coefficient (r): {assortativity_r:.4f}\n")


# Question 7
# What is the distribution of nodes by degree and does it follow a power law distribution?
# To receive full points for this question, it is necessary to formally support the answer mathematically.

import powerlaw

# 1. Get degrees
degrees = [d for n, d in graph.degree()]

# 2. Fit the data
fit = powerlaw.Fit(degrees, discrete=True)

# 3. Get formal metrics
alpha = fit.power_law.alpha  # The gamma exponent
xmin = fit.power_law.xmin  # The threshold where power law starts

# Compare Power Law vs Exponential distribution
R, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

# 4. Visualization
plt.figure(figsize=(10, 6))
fig = fit.plot_pdf(color='b', linewidth=2, label='Empirical Data')
fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig, label='Power Law Fit')
plt.title(f"Degree Distribution & Power Law Fit (alpha={alpha:.2f})")
plt.xlabel("Degree (k)")
plt.ylabel("P(k)")
plt.legend()
plt.savefig(os.path.join("pictures_from_code", "DZ1NetDegree_Distribution_PowerLaw.png"), bbox_inches='tight')
plt.close()

print(f"\n--- Question 7 Results ---")
print(f"Alpha (gamma): {alpha}")
print(f"Xmin: {xmin}")
print(f"Likelihood Ratio (R): {R}")
print(f"p-value: {p_value}")


# Question 8
# Conduct degree centrality, closeness and relational centrality analyses.
# Give an overview of the most important actors for each of them.

# 1. Calculate Centralities
degree_cent = nx.degree_centrality(graph)
closeness_cent = nx.closeness_centrality(graph)
betweenness_cent = nx.betweenness_centrality(graph)

# 2. Combine into a DataFrame for easy viewing
centrality_data = {
    'Node': list(graph.nodes()),
    'Degree': list(degree_cent.values()),
    'Closeness': list(closeness_cent.values()),
    'Betweenness': list(betweenness_cent.values())
}

df = pd.DataFrame(centrality_data)

# 3. Identify Top 3 Actors for each metric
top_degree = df.nlargest(3, 'Degree')
top_closeness = df.nlargest(3, 'Closeness')
top_betweenness = df.nlargest(3, 'Betweenness')

print(f"\n--- Question 8 Results ---")
print("--- Top Acters by Degree ---")
print(top_degree[['Node', 'Degree']])
print("\n--- Top Acters by Closeness ---")
print(top_closeness[['Node', 'Closeness']])
print("\n--- Top Acters by Betweenness ---")
print(top_betweenness[['Node', 'Betweenness']])


# Question 9
# Who are the most important actors by centrality according to their own vector or equivalent metrics (hub score, authority score) if the network is modeled as a directed graph?
# What does that tell us about them?

# 1. Calculate Hub and Authority scores
# This automatically accounts for the directed nature of your similarity graph
hubs, authorities = nx.hits(graph)

# 2. Organize the data
hits_df = pd.DataFrame({
    'Node': list(hubs.keys()),
    'Hub_Score': list(hubs.values()),
    'Authority_Score': list(authorities.values())
})

# 3. Get the top 3 for each
top_authorities = hits_df.nlargest(3, 'Authority_Score')
top_hubs = hits_df.nlargest(3, 'Hub_Score')

print(f"\n--- Question 9 Results ---")
print("--- Top Authorities (The Sources) ---")
print(top_authorities[['Node', 'Authority_Score']])
print("\n--- Top Hubs (The Distributors) ---")
print(top_hubs[['Node', 'Hub_Score']])


# Question 10
# Based on the previous two questions, propose and construct a heuristic (composite measure of centrality) to find the most important actors and find them.
# Pay attention to the type of network being analyzed (directed or undirected) and, accordingly, adjust how different network metrics affect the heuristics.

# 1. Calculate the core metrics for the directed graph
in_degree = nx.in_degree_centrality(graph)
hubs, authorities = nx.hits(graph)

# 2. Create the unified DataFrame
final_eval = pd.DataFrame({
    'Node': list(graph.nodes()),
    'InDegree': [in_degree[node] for node in graph.nodes()],
    'Authority': [authorities[node] for node in graph.nodes()]
})

# 3. Normalize to a 0-1 scale (Min-Max Scaling)
# This ensures that both metrics contribute equally
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

final_eval['InDeg_Norm'] = normalize(final_eval['InDegree'])
final_eval['Auth_Norm'] = normalize(final_eval['Authority'])

# 4. Calculate the Composite Source Index (CSI)
final_eval['CSI_Score'] = (final_eval['InDeg_Norm'] + final_eval['Auth_Norm']) / 2

# 5. Get the Top 5
top_final = final_eval.sort_values(by='CSI_Score', ascending=False).head(5)
print(f"\n--- Question 10 Results ---")
print(top_final[['Node', 'CSI_Score', 'InDeg_Norm', 'Auth_Norm']])

# In order to answer other questions, we export the graph we created and worked on into a gephi-ready format
nx.write_gexf(graph, "gephi\DZ1Net_gephi.gexf")


# Question 13 and 14
# Conduct spectral analysis and evaluate potential candidates for the number of communes in the network.
# Compare the result with the dendrogram constructed by the Girvan-Newman method.
# Who are the actors that can be characterized as key brokers (bridges) in the network?
# What makes them brokers? Compare the answer with the key nodes obtained from the centrality analysis.

# We return to Python for Questions 13 and 14.
# GEPHI doesn't provide in-built Girvan-Newman method, so we will have to use NetworkX
from scipy.linalg import eigh
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# 1. Prepare the Undirected Graph
undirected_g = graph.to_undirected()

# --- SPECTRAL ANALYSIS ---
L = nx.laplacian_matrix(undirected_g).toarray()
vals, vecs = eigh(L)
vals = np.sort(vals)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), vals[:10], 'b*-')
plt.title("Spectral Analysis: First 10 Eigenvalues (Find the Eigengap)")
plt.xlabel("Number of Clusters")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.savefig(os.path.join("pictures_from_code", "DZ1NetSpectralAnalysis.png"))
plt.close()

# --- GIRVAN-NEWMAN HIERARCHY ---
comp = nx.community.girvan_newman(undirected_g)
level_1 = next(comp)
level_2 = next(comp)
level_3 = next(comp)
level_4 = next(comp)

# print(f"GN Level 1 splits into: {len(level_1)} clusters") -> 2 clusters
# print(f"GN Level 2 splits into: {len(level_2)} clusters") -> 3 clusters
# print(f"GN Level 3 splits into: {len(level_3)} clusters") -> 4 clusters
# print(f"GN Level 4 splits into: {len(level_4)} clusters") -> 5 clusters

# --- DENDROGRAM VISUALIZATION ---
# Using Ward's method on the distance matrix derived from the graph
adj_matrix = nx.to_numpy_array(undirected_g)
# We treat similarity as inverse distance: more similarity = less distance
dist_matrix = 1 / (1 + adj_matrix)
np.fill_diagonal(dist_matrix, 0)

# Generate the linkage matrix
Z = sch.linkage(squareform(dist_matrix), method='ward')

plt.figure(figsize=(12, 7))
sch.dendrogram(Z, labels=list(graph.nodes()), leaf_rotation=90, leaf_font_size=10, color_threshold=0.7 * max(Z[:,2]))

plt.title('Hierarchical Clustering Dendrogram (Girvan-Newman Structural Logic)')
plt.xlabel('Student ID')
plt.ylabel('Distance (Structural Dissimilarity)')
plt.savefig(os.path.join("pictures_from_code", "DZ1NetDendogram.png"))
plt.close()

# Question 15