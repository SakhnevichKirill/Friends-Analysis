import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import scipy.io
import scipy.spatial as spt
import scipy.stats
import seaborn as sns
import vk_api
from IPython.display import SVG
from main import auth_handler
from secret import login, password


def fetch_members_ids():
    ids = []
    for i in [1, 1000, 2000]:
        members_url = "https://api.vk.com/method/groups.getMembers?group_id=49815762&offset={}"
        print(requests.get(members_url.format(i)).json())
        json_response = requests.get(members_url.format(i)).json()["response"]["users"]
        ids.extend(json_response)
    return ids


def get_members_friends(members_ids):
    vk_session = vk_api.VkApi(login, password, app_id=2685278, auth_handler=auth_handler)
    try:
        vk_session.auth(token_only=True)
    except vk_api.AuthError as error_msg:
        print(error_msg)
    # members_friends = {}
    with vk_api.VkRequestsPool(vk_session) as pool:
        members_friends = pool.method_one_param("friends.get", key="user_id", values=members_ids)
    return members_friends


def find_deactivated_members(friends):
    deactivated_members = [element for element in friends if element["is_closed"] == True]
    print("From {} members {} are deactivated:\n".format(len(friends), len(deactivated_members)))
    return deactivated_members


def drop_deactivated_members(friends):
    new_friends = [element for element in friends if element["is_closed"] == False]
    return new_friends


def drop_members_with_hidden_friends(members):
    new_members = {k: v for k, v in members.items() if v["count"] != 0}
    return new_members


def create_members_graph(members):
    graph = nx.Graph(directed=False)
    for i in members:
        graph.add_node(i)
        for j in members[i]["items"]:
            if i != j and j in members:
                graph.add_edge(i, j)
    return graph


def make_list_with_members_info(g):
    vk_session = vk_api.VkApi(login, password)
    vk_session.authorization()
    vk = vk_session.get_api()
    portions_of_ids = [int(len(g.nodes()) / 4) * i for i in range(0, 4)] + [len(g.nodes())]
    response = list()
    for i in range(0, 4):
        members_ids = ", ".join(map(str, g.nodes()[portions_of_ids[i] : portions_of_ids[i + 1]]))
        response += vk.users.get(user_ids=members_ids, fields="sex, city, education", lang="en")
    return response


def set_attributes_to_nodes(graph, response, members_friends):
    # name
    g = graph
    member_name = [i["first_name"] + " " + i["last_name"] for i in response]
    member_name = dict(zip(g.nodes(), member_name))
    nx.set_node_attributes(g, "name", member_name)
    # gender
    member_gender = [i["sex"] for i in response]
    member_gender = dict(zip(g.nodes(), member_gender))
    nx.set_node_attributes(g, "gender", member_gender)
    # city title
    member_city = [i["city"]["title"] if "city" in i else "-" for i in response]
    member_city = dict(zip(g.nodes(), member_city))
    nx.set_node_attributes(g, "city", member_city)
    # university id
    member_university = [i["university"] if "university" in i else 0 for i in response]
    member_university = dict(zip(g.nodes(), member_university))
    nx.set_node_attributes(g, "university", member_university)
    # number of friends (popularity)
    member_friends_count = [members_friends[i]["count"] for i in g.nodes()]
    member_friends_count = dict(zip(g.nodes(), member_friends_count))
    nx.set_node_attributes(g, "friends", member_friends_count)
    return g


def drop_lonely_users(graph, number_of_connections):
    to_remove = [k for k, v in graph.degree if v <= number_of_connections]
    graph.remove_nodes_from(to_remove)
    return graph


def get_basic_information(g):
    print(
        "'Friends' network has {} active members with {} connections between each other.".format(
            g.number_of_nodes(), g.number_of_edges()
        )
    )
    print("Number of connected components = {}".format(nx.number_connected_components(g)))


def get_nodes_degree(g):
    node_degree = [v for k, v in g.degree]
    print("Max value of node degree = {}".format(max(node_degree)))
    print("Mean value of node degree = {}".format(np.mean(node_degree)))
    return node_degree
