import requests
import networkx
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import seaborn as sns
import scipy.spatial as spt
import igraph
# from main import *
# from vk_token import *
from functions_for_vk_users import *
from functions_for_vk_groups import *

def k_core_decompose(G):
    ### BEGIN SOLUTION
    return np.array(list(nx.core_number(G).values()))
    ### END SOLUTION

prefix = 'data/'
graph_path = 'service/myGraph.gpickle'

sns.set(style="whitegrid", color_codes=True)
g = nx.read_gpickle(graph_path)

figsize = (150, 100)
show_graph(g, size_of_nodes=100, figsize=figsize, save_file=prefix+'Network_friends_overview.png')

ext = '.png'
title = 'node_degree_graph'
node_degree = [v for k, v in g.degree]
core_g = drop_lonely_users(g, 0)
show_number = 10
df = show_graph(core_g, core_g.degree(), size_of_nodes=100, number=show_number, save_file=prefix+title+ext, figsize=figsize, parameter_name=title)
print(df[:show_number])

size_of_nodes = 10000

title = 'degree_centrality'
degree_centrality = networkx.degree_centrality(core_g)
df_new = show_graph(core_g, degree_centrality.items(), size_of_nodes=size_of_nodes, number=show_number, save_file=prefix+title+ext, figsize=figsize, parameter_name=title)
df = df.merge(df_new, how='left')
print(df[:show_number])

title = 'closeness_centrality'
closeness_centrality = networkx.closeness_centrality(core_g)
df_new = show_graph(core_g, closeness_centrality.items(), size_of_nodes=size_of_nodes, number=show_number, save_file=prefix+title+ext, figsize=figsize, parameter_name=title)
df = df.merge(df_new, how='left')
print(df[:show_number])

title = 'betweenness_centrality'
betweenness_centrality = networkx.betweenness_centrality(core_g)
df_new = show_graph(core_g, betweenness_centrality.items(), size_of_nodes=size_of_nodes, number=show_number, save_file=prefix+title+ext, figsize=figsize, parameter_name=title)
df = df.merge(df_new, how='left')
print(df[:show_number])

title = 'pagerank'
pagerank = networkx.pagerank(core_g, alpha=0.85)
df_new = show_graph(core_g, pagerank.items(), size_of_nodes=size_of_nodes, number=show_number, save_file=prefix+title+ext, figsize=figsize, parameter_name=title)
df = df.merge(df_new, how='left')
print(df[:show_number])


base_title = 'k_core_decompose'


lespos = nx.kamada_kawai_layout(g)

# plt.figure(figsize=(8*2, 8*4))
x_max, y_max = np.array(list(lespos.values())).max(axis=0)
x_min, y_min = np.array(list(lespos.values())).min(axis=0)

for i in range(8):
    # plt.subplot(4, 2, i+1)
    subG = nx.k_core(g, i+1)
    node_colors = k_core_decompose(subG)
    title = str(i)+base_title
    # Networkx plot
    parameter = [(node_key, color)  for node_key, color in zip(subG.nodes.keys(), node_colors)]
    df_new = create_df_with_param(subG, parameter, parameter_name=title)
    df = df.merge(df_new, how='left')
    labels_df = df_new[['first_name', 'last_name', 'city_title']]
    node_labels = get_node_labels(labels_df)
    eps = (x_max - x_min) * 0.05
    plt.xlim(x_min-eps, x_max+eps)
    plt.ylim(y_min-eps, y_max+eps)
    plt.figure(figsize=(150,100))
    nx.draw(
        subG,
        lespos,
        cmap=plt.cm.rainbow,
        node_color=node_colors,
        font_size=25,
        node_size=1000,
        linewidths=1,
        edgecolors='black',
        labels=node_labels,
    )
    plt.title('k-shells on {}-core'.format(i+1), fontsize=40)
    plt.savefig(prefix+title+ext)
    plt.axis('off')
    # plt.show()

print(df[:show_number])
df.to_excel(prefix+'DF'+".xlsx") 
