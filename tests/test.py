import vk_api
from functions_for_vk_users import get_friends_ids
from main import auth_handler, get_public_posts_by_count
from secret import login, password

vk_session = vk_api.VkApi(login, password, app_id=2685278, auth_handler=auth_handler)
try:
    vk_session.auth(token_only=True)
except vk_api.AuthError as error_msg:
    print(error_msg)


# print(vk_session.token['user_id'])
friends, errors = vk_api.vk_request_one_param_pool(
    vk_session,
    "friends.get",  # Метод
    key="user_id",  # Изменяющийся параметр
    values=[vk_session.token["user_id"]],
    # Параметры, которые будут в каждом запросе
    default_values={"fields": "photo"},
)

print(friends)
print(errors)


# test = get_friends_ids(393551038)
# print(test)
# get_public_posts_by_count(vk_session, '-297836', 1)
