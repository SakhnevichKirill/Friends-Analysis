import json
import os
import pickle
import re
import string
import time

import emoji
import networkx as nx
import numpy as np
import pymorphy2
import requests
import secret
import vk_api

morph = pymorphy2.MorphAnalyzer()


# Функция для двухфакторной аутентификации
def auth_handler():
    # Код двухфакторной аутентификации
    key = input("Enter authentication code: ")
    # Если: True - сохранить, False - не сохранять.
    remember_device = True
    return key, remember_device


# Получение постов с личной стены
def get_personal_wall_posts(vk_session):
    vk = vk_session.get_api()
    response = vk.wall.get(count=1)
    if response["items"]:
        print(response["items"][0])


# Получение опреденного количества постов со стены группы по id
def get_public_posts_by_count(vk_session, owner_id, count=100):
    vk = vk_session.get_api()
    response = vk.wall.get(owner_id=owner_id, count=count)
    print(response)


# Получение всех постов со стены группы по id
def get_public_posts_all(vk_session, owner_id):
    vk = vk_session.get_api()
    rez_responce = []
    offs = 0
    cur_response = vk.wall.get(owner_id=owner_id, offset=offs, count=100)
    all_posts = cur_response.get("count")
    rez_responce.extend(cur_response.get("items"))
    offs += 101

    while offs < all_posts:
        print(offs)
        cur_response = vk.wall.get(owner_id=owner_id, offset=offs, count=100)
        rez_responce.extend(cur_response.get("items"))
        offs += 101

    count_iter = 0
    print("----", len(rez_responce))
    comments_strings = []
    for post in rez_responce:
        print(count_iter)
        comments = vk.wall.getComments(owner_id=owner_id, post_id=post["id"], count=100)["items"]
        cur_ids = []
        for comment in comments:
            cur_ids.append(comment["from_id"])
        for i in cur_ids:
            for i1 in range(len(cur_ids)):
                if cur_ids[i1] != i:
                    comments_strings.append(str(i) + " " + str(cur_ids[i1]))
        count_iter += 1
    return rez_responce, comments_strings


# Удаление всех рекламных постов
def posts_dataset(all_posts):
    for i in all_posts:
        if i["marked_as_ads"] == 1:
            all_posts.remove(i)


# Нормализация текста
def text_preparation(dataset):
    rezult_text = []
    for i in dataset:
        cur_text = i["text"]
        cur_text = cur_text.lower()  # Lower text
        cur_text = emoji.replace_emoji(cur_text, " ")  # Delete emoji
        cur_text = re.sub(r"^https?:\/\/.*[\r\n]*", "", cur_text, flags=re.MULTILINE)  # Delete links
        cur_text = re.sub(r"\S*@\S*\s?", "", cur_text, flags=re.MULTILINE)  # Delete emails
        cur_text = cur_text.translate(str.maketrans("", "", string.punctuation))  # Delete punctuation
        cur_text = cur_text.translate(str.maketrans("", "", string.digits))  # Delete digits
        cur_text = cur_text.strip()  # Delete spaces
        rezult_text.append(cur_text)
    return rezult_text


# Лемматизация текста
def lemmatization(text):
    rez_words = {}
    for i in text:
        for j in i.split():
            p = morph.parse(j)[0]
            rez_words[p.normal_form] = rez_words.setdefault(p.normal_form, 0) + 1
    sort_words = sorted(rez_words.items(), key=lambda x: x[1], reverse=True)
    return sort_words


# Part 2-----------------------------------
group_api_url = "https://api.vk.com/method/groups.getMembers?group_id="
id_api_url = "https://api.vk.com/method/friends.get?user_id="
fields = "&fields=sex,bdate,city,country"
count = "&count=1000"
offset = "&offset="
v = "&v=5.131"
access_token = "&access_token=vk1."
# Id группы ВК

# 29842742
#  297836

# Рифмы и Панчи
vk_group_id = 28905875


# Извлечение участников группы
def extract_members(group_id):
    list_of_members = []
    j = 1
    url = group_api_url + str(group_id) + offset + str(0) + count + access_token + v
    json_response = requests.get(url).json()
    users = json_response["response"]["items"]
    list_of_members += users
    i = len(users)
    while json_response["response"]["next_from"] != "":
        time.sleep(0.12)
        url = group_api_url + str(group_id) + offset + str(i) + count + access_token + v
        print(i)
        json_response = requests.get(url).json()
        users = json_response["response"]["items"]
        list_of_members += users
        i = i + len(users)
        j = j + 1
        print(j)
    return list_of_members


# Получение списка друзей участников группы
def user_friends_list(user_id, group_members):
    url = id_api_url + str(user_id) + access_token + v

    time.sleep(0.12)

    try:
        json_response = requests.get(url).json()
    except requests.exceptions.RequestException:
        print("error")
        return []

    if "error" in json_response.keys():
        print("error")
        return []

    friends_inside_community_list = list(set(json_response["response"]["items"]).intersection(group_members))
    print("success")
    return friends_inside_community_list


if __name__ == "__main__":
    vk_session = vk_api.VkApi(secret.login, secret.password, app_id=2685278, auth_handler=auth_handler)
    try:
        vk_session.auth(token_only=True)
    except vk_api.AuthError as error_msg:
        print(error_msg)

    get_personal_wall_posts(vk_session)
    print("---------------------------")
    get_public_posts_by_count(vk_session, f"-{vk_group_id}", 2)
    print("---------------------------")
    rez1 = get_public_posts_all(vk_session, f"-{vk_group_id}")

    posts_dataset(rez1[0])
    print(len(rez1[0]))
    rez_text1 = text_preparation(rez1[0])

    # Загрузка текста в JSON для дальнейшей визуализации
    with open(os.path.join("./data", "data2.json"), "w", encoding="utf-8") as f:
        json.dump(rez_text1, f, ensure_ascii=False, indent=4)

    # Загрузка текста в файл для дальнейшей визуализации
    my_file1 = open(os.path.join("./data", "text_public1.txt"), "w")
    for i in rez_text1:
        my_file1.write(i + "\n")
    my_file1.close()

    # Загрузка id для графа
    my_file3 = open(os.path.join("./data", "ids_public1.txt"), "w")
    for i in rez1[1]:
        my_file3.write(i + "\n")
    my_file3.close()

    # Запись участников группы и друзей среди них
    group_members = extract_members(vk_group_id)
    print(len(group_members))
    print(group_members[4000])

    output = open(os.path.join("./data", "group_members_" + str(vk_group_id) + ".pkl"), "wb")
    pickle.dump(group_members, output)
    output.close()
    with open(os.path.join("./data", "group_members_" + str(vk_group_id) + ".pkl"), "rb") as f:
        group_members = pickle.load(f)
    url = id_api_url + str(61781867) + access_token + v
    print(requests.get(url).json())

    f1 = open(os.path.join("./data", "friends_inside_" + str(vk_group_id) + ".txt"), "w")

    for member_id in group_members:
        a = user_friends_list(member_id, group_members)

        if len(a) != 0:
            for friend_id in a:
                f1.write("%d" % member_id)
                f1.write(" ")
                f1.write("%d" % friend_id)
                f1.write("\n")

    f1.close()
