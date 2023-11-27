import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.io
import scipy.stats
import scipy.spatial as spt
from IPython.display import SVG

import vk_api
from secret import login, password
from main import auth_handler
from functions_for_vk_users import show_graph, get_graph_with_friends_connections, get_friends_childs, get_friends_ids


vk_session = vk_api.VkApi(login, password, app_id=2685278, auth_handler=auth_handler)
try:
    vk_session.auth(token_only=True)
except vk_api.AuthError as error_msg:
    print(error_msg)

my_id = vk_session.token['user_id']
post_id = "315268227_509" # Некит.
my_id = int(post_id.split("_")[0])
print(my_id)
friends = get_friends_ids(vk_session, [my_id])
parent_childs_ids, parent_ids = get_friends_childs(friends)
# print(len(parent_childs_ids), [len(i) for i in parent_childs_ids], parent_childs_ids)
# print(len(parent_ids), parent_ids)

friends_1 = get_friends_ids(vk_session, friends, req_count=5)

print(len(friends_1[my_id]['childs']), friends_1[my_id]['childs'])

parent_childs_ids, parent_ids = get_friends_childs(friends_1)

# print(len(parent_childs_ids), parent_childs_ids)
# print(len(parent_childs_ids[0]), parent_childs_ids[0])
# print(len(parent_childs_ids[0][1]), parent_childs_ids[0][1])
# print(len(parent_childs_ids[0][0]), parent_childs_ids[0][0])
# print(len(parent_childs_ids[0][0][0]), parent_childs_ids[0][0][0])
# print(len(parent_ids), parent_ids)

g = get_graph_with_friends_connections(friends_1)
g.remove_node(my_id)

nx.write_gpickle(g,'service/myGraph.gpickle')