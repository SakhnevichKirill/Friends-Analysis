import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy.spatial as spt
import vk_api


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_items_list_ids(user_ids: dict, id: int | str):
    parent_childs_ids = [(element["id"], element) for element in user_ids[id]["items"] if element["is_closed"] == False]
    # print("get_items_list_ids", parent_childs_ids)
    return parent_childs_ids


def get_childs_list_ids(user_ids: dict, id: int | str, intersected: bool = True):
    new_user_ids = user_ids[id]["childs"]
    parent_ids = [k for k, v in new_user_ids.items()]
    parent_childs_ids = [get_items_list_ids(new_user_ids, id) for id in parent_ids]
    if intersected:
        intersected_parent_childs_ids = [
            [child for child in childs if parent_id != child[0] and child[0] in parent_ids]
            for parent_id, childs in zip(parent_ids, parent_childs_ids)
        ]
        print(
            "intersected get_childs_list_ids", len(intersected_parent_childs_ids), len(intersected_parent_childs_ids[0])
        )
        return (intersected_parent_childs_ids, parent_ids)
    else:
        print("get_childs_list_ids", len(parent_childs_ids), len(parent_childs_ids[0]))
        return (parent_childs_ids, parent_ids)


def get_friends_childs(user_ids: dict):
    parent_ids = [k for k, v in user_ids.items()]
    # Вытаскиваем только не закрытые аккаунты
    parent_childs_ids = [
        get_items_list_ids(user_ids, id) if "items" in user_ids[id] else get_childs_list_ids(user_ids, id)
        for id in parent_ids
    ]
    return (parent_childs_ids, parent_ids)


def __get_parent_friends_ids(vk_session, user_ids: list, req_count: int=15) -> dict:
    if len(user_ids) > req_count:
        friends_1 = __get_parent_friends_ids(vk_session, user_ids[:req_count], req_count)
        friends_2 = __get_parent_friends_ids(vk_session, user_ids[req_count:], req_count)
        friends = {}
        friends.update(friends_1)
        friends.update(friends_2)
    else:
        if type(user_ids) == list:
            user_ids = [user[0] if type(user) == tuple else user for user in user_ids]
        friends, errors = vk_api.vk_request_one_param_pool(
            vk_session,
            "friends.get",  # Метод
            key="user_id",  # Изменяющийся параметр
            values=user_ids,
            # Параметры, которые будут в каждом запросе
            default_values={
                "fields": [
                    "photo_id",
                    "status",
                    "sex",
                    "bdate",
                    "can_post",
                    "can_see_all_posts",
                    "can_write_private_message",
                    "city",
                    "contacts",
                    "country",
                    "domain",
                    "education",
                    "has_mobile",
                    "timezone",
                    "last_seen",
                    "nickname",
                    "online",
                    "relation",
                    "universities",
                ]
            },
        )
        if len(errors) != 0:
            print(errors)
            print(user_ids)
            # raise
    return friends


def get_friends_ids(vk_session, user_ids, req_count: int=15):
    if type(user_ids) == int:
        user_ids = [user_ids]
    elif type(user_ids) == list and len(user_ids) != 0:
        if type(user_ids[0]) == int:
            pass
        elif type(user_ids[0]) == str:
            pass
        else:
            raise f"Unexpected type: {type(user_ids[0])}"
    elif type(user_ids) == dict:
        parent_user_ids, parent_ids = get_friends_childs(user_ids)
        values = [__get_parent_friends_ids(vk_session, user_ids, req_count) for user_ids in parent_user_ids]
        print(values)
        res = {}
        for k, v in zip(parent_ids, values):
            res[k] = {"count": len(v), "childs": v}
        return res

    return __get_parent_friends_ids(vk_session, user_ids, req_count)


def decompose_keys(friend: dict, key1: str, key2: list[str] = []):
    if len(key2) == 0 and type(friend[key1]) == dict:
        for k, v in friend[key1].items():
            friend[f"{key1}_{k}"] = friend[key1][k]
        friend.pop(key1)
    else:
        for k in key2:
            friend[f"{key1}_{k}"] = friend[key1][k]
    return friend


def process_friend_info(friend: dict):
    new_friend = friend.copy()
    for k, v in friend.items():
        new_friend = decompose_keys(new_friend, k, [])
    # friend = decompose_keys(friend, 'city', [])
    return new_friend


def pack_to_graph(g: nx.Graph, parent_user_ids, parent_ids):
    for parent, friends in zip(parent_ids, parent_user_ids):
        g.add_node(parent)
        if type(friends) == tuple:
            # print(friends[0], friends[1])
            g = pack_to_graph(g, friends[0], friends[1])
        else:
            for friend_id in friends:
                g.add_node(friend_id[0], **process_friend_info(friend_id[1]))
                g.add_edge(parent, friend_id[0])
    return g


def get_graph_with_friends_connections(friend_ids):
    """
    this function returns graph of friends and connection between them
    """
    parent_user_ids, parent_ids = get_friends_childs(friend_ids)
    g = nx.Graph(directed=False)
    return pack_to_graph(g, parent_user_ids, parent_ids)


def draw_graph(g, parameter, size_of_nodes, number):
    """
    draws plot according to parameter of interest and size of nodes
    """
    node_labels = find_top_nodes(g, parameter, number)
    plt.xkcd()
    plt.figure(1, figsize=(30, 25))
    coord = nx.spring_layout(g)
    nx.draw(
        g,
        pos=coord,
        nodelist=parameter.keys(),
        node_size=[d * size_of_nodes for d in parameter.values()],
        node_color=list(parameter.values()),
        font_size=25,
        cmap=plt.cm.get_cmap("RdBu_r"),
        labels=node_labels,
    )

def create_df_with_param(g, parameter: dict={}, parameter_name: str='parameter'):
    df = pd.DataFrame(g.nodes.values())
    df = df[df['id'].notna()]
    df['id'] = df['id'].astype('int')
    if len(parameter) != 0:
        print(parameter)
        df_degree = pd.DataFrame(parameter, columns=['id', parameter_name])
        print(df_degree)
        df = pd.merge(df, df_degree,how="left")
        df = df.sort_values(parameter_name, ascending=False)
        columns = ["id", "domain", "sex", "first_name", "last_name", "university_name", "city_title", parameter_name]
        for i, col in enumerate(columns):
            if col not in df.columns:
                columns.pop(i)
                print("Scipping: ", col)
        df = df[columns]
        df = df.set_index('id')
    else:
        df =df.set_index('id')
    return df

def get_node_labels(df: pd.DataFrame, number: int=-1):
    node_labels = df[:number].T.to_dict('list')
    node_labels = {k: f'{label[0]}_{label[1]}_{label[2]}' if type(label[2]) == str else f'{label[0]}_{label[1]}' for k, label in node_labels.items()}
    return node_labels

from matplotlib.lines import Line2D

def show_graph(g, parameter: dict={}, size_of_nodes=1000, number=15, figsize=(40,25), parameter_name: str='parameter', save_file: str = '', show: bool=True):
    lespos = nx.kamada_kawai_layout(g)
    sorted_df = create_df_with_param(g, parameter, parameter_name=parameter_name)
    labels_df = sorted_df[['first_name', 'last_name', 'city_title']]
    plt.figure(1, figsize=figsize)
    if len(parameter) != 0:
        node_labels = get_node_labels(labels_df, number)
        nx.draw(
            g,
            pos=lespos,
            node_size=[v * size_of_nodes for k, v in parameter],
            node_color=[v for k, v in parameter],
            font_size=25,
            cmap=plt.cm.get_cmap("RdBu_r"),
            labels=node_labels,
        )
        # nodes = nx.draw_networkx_nodes(
        #     g,
        #     lespos,
        #     cmap=plt.cm.OrRd,
        #     node_color=[v for k, v in parameter],
        #     node_size=100,
        #     edgecolors='black'
        # )
        # plt.legend(*nodes.legend_elements())
        # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markersize=10) 
        #             for label in node_labels.keys()]

        # plt.legend(handles=legend_elements, title='Top Nodes', loc='upper right')
    else:
        node_labels = get_node_labels(labels_df)
        nx.draw(
                g,
                pos=lespos,
                node_color='white',
                edgecolors='black',
                node_size=size_of_nodes,
                font_size=25,
                # cmap=plt.cm.get_cmap("RdBu_r"),
                labels=node_labels,
            )
    plt.title("Graph of Friends Connections", fontsize=40)
    if save_file != '':
        plt.savefig(save_file)
    if show:
        plt.show()
    return sorted_df


def print_full_name_for_id(id_interest):
    """
    :param id_interest: int
    :return: string
    """
    response = requests.get("https://api.vk.com/method/users.get?user_ids={}".format(id_interest)).json()["response"]
    print(response[0]["first_name"].strip() + " " + response[0]["last_name"].strip())


def count_likes(ph):
    owner_id = int(ph.split("_")[0])
    photo_id = int(ph.split("_")[1])
    likes = requests.get(
        "https://api.vk.com/method/likes.getList?type=photo&owner_id=%d&item_id=%d" % (owner_id, photo_id)
    ).json()
    print(likes)
    return likes["count"]


def info_about(id_api_url, id_interest, access_token, v):
    url = id_api_url + str(id_interest) + access_token + v
    resp = requests.get(url).json()["response"]
    if "photo_id" in resp[0].keys():
        resp[0]["popularity"] = count_likes(resp[0]["photo_id"].strip())
    else:
        resp[0]["popularity"] = 0
    resp[0]["name"] = resp[0]["first_name"].strip() + " " + resp[0]["last_name"].strip()
    if "schools" in resp[0].keys():
        try:
            resp[0]["school"] = resp[0]["schools"][0]["id"]
        except:
            resp[0]["school"] = 0
    else:
        resp[0]["school"] = 0
    if "university" not in resp[0].keys():
        resp[0]["university"] = 0
    del resp[0]["first_name"]
    del resp[0]["last_name"]
    return resp[0]


def get_friends_information(g):
    for i in nx.nodes(g):
        information = info_about(i)
        if "deactivated" in information.keys():
            g.remove_node(i)
        else:
            g.node[i]["name"] = information["name"]
            g.node[i]["popularity"] = information["popularity"]
            g.node[i]["school"] = information["school"]
            g.node[i]["university"] = information["university"]
            g.node[i]["city"] = information["city"]
            time.sleep(0.25)
            print(g.node[i]["name"] + "is fine")


def find_top_nodes(g, values, number):
    print(values)
    sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
    print(sorted_values)
    best = {i[0]: g.nodes[i[0]]["domain"] for i in sorted_values[0:number]}
    return best

def get_sparse_matrix(graph):
    g_sparse = nx.utils.reverse_cuthill_mckee_ordering(graph)
    reorder_nodes = list(g_sparse)
    a = nx.to_numpy_matrix(graph, nodelist=reorder_nodes, dtype=int)
    a = np.asarray(a)
    return a


def plot_similarity(a):
    f, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].imshow(a, cmap="Greens", interpolation="None")
    ax[0, 0].set_title("Adjacency Matrix")

    d = np.corrcoef(a)
    ax[1, 0].imshow(d, cmap="Greens", interpolation="None")
    ax[1, 0].set_title("Correlation coefficient")

    distance = spt.distance.pdist(a, metric="euclidean")
    d = spt.distance.squareform(distance)
    ax[0, 1].imshow(d, cmap="Greens", interpolation="None")
    ax[0, 1].set_title("Euclidean Distance")

    distance = spt.distance.pdist(a, metric="cosine")
    d = spt.distance.squareform(distance)
    ax[1, 1].imshow(d, cmap="Greens", interpolation="None")
    ax[1, 1].set_title("Cosine Distance")


def compare_graphs(graph):
    n = nx.number_of_nodes(graph)
    m = nx.number_of_edges(graph)
    k = np.mean([v for k, v in graph.degree()])
    erdos = nx.erdos_renyi_graph(n, p=m / float(n * (n - 1) / 2))
    barabasi = nx.barabasi_albert_graph(n, m=int(k) - 7)
    small_world = nx.watts_strogatz_graph(n, int(k), p=0.04)
    print(" ")
    print("Compare the number of edges")
    print(" ")
    print("My network: " + str(nx.number_of_edges(graph)))
    print("Erdos: " + str(nx.number_of_edges(erdos)))
    print("Barabasi: " + str(nx.number_of_edges(barabasi)))
    print("SW: " + str(nx.number_of_edges(small_world)))
    print(" ")
    print("Compare average clustering coefficients")
    print(" ")
    print("My network: " + str(nx.average_clustering(graph)))
    print("Erdos: " + str(nx.average_clustering(erdos)))
    print("Barabasi: " + str(nx.average_clustering(barabasi)))
    print("SW: " + str(nx.average_clustering(small_world)))
    print(" ")
    print("Compare average path length")
    print(" ")
    for i, c in enumerate(nx.connected_components(graph)):
        conn_g = graph.subgraph(c)
        print(f'My network {i}: ' + str(nx.average_shortest_path_length(conn_g)))
    print("Erdos: " + str(nx.average_shortest_path_length(erdos)))
    print("Barabasi: " + str(nx.average_shortest_path_length(barabasi)))
    print("SW: " + str(nx.average_shortest_path_length(small_world)))
    print(" ")
    print("Compare graph diameter")
    print(" ")
    for i, c in enumerate(nx.connected_components(graph)):
        conn_g = graph.subgraph(c)
        print(f'My network {i}: ' + str(nx.diameter(conn_g)))
    print("Erdos: " + str(nx.diameter(erdos)))
    print("Barabasi: " + str(nx.diameter(barabasi)))
    print("SW: " + str(nx.diameter(small_world)))
