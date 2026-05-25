from db.loader import load_all_data
from config import USER1_ID, USER2_ID

for user_id in [USER1_ID, USER2_ID]:
    data = load_all_data(user_id)

    print(f"\nUser {user_id} loaded successfully")
    print("Foods:", data["foods"].shape)
    print("Food nutrients:", data["food_nutrients"].shape)
    print("DRI:")
    print(data["dri"])