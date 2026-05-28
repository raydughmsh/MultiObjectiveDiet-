import pymysql
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from config import DB_CONFIG, NUTRIENT_IDS


def get_connection():
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )


def clean_numeric_id(df, column_name):
    """
    Removes fake header rows by keeping only numeric IDs.
    """
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df = df.dropna(subset=[column_name])
    df[column_name] = df[column_name].astype(int)
    return df


def load_foods():
    query = """
        SELECT
            id,
            name,
            foodGroupId,
            caseStudy,
            cost,
            preference,
            preference2,
            preparingTime,
            COALESCE(cookingTime, 0) AS cookingTime,
            co2
        FROM foods;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df = clean_numeric_id(df, "id")

    numeric_cols = [
        "foodGroupId",
        "caseStudy",
        "cost",
        "preference",
        "preference2",
        "preparingTime",
        "cookingTime",
        "co2",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("id").reset_index(drop=True)


def load_food_groups():
    query = """
        SELECT id, name
        FROM food_group;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df = clean_numeric_id(df, "id")
    return df.sort_values("id").reset_index(drop=True)


def load_nutrients():
    query = """
        SELECT id, name
        FROM nutrients;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df = clean_numeric_id(df, "id")
    return df.sort_values("id").reset_index(drop=True)


def load_food_nutrients():
    query = """
        SELECT
            foodId,
            nutrientId,
            quantity AS value
        FROM food_nutrients;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df["foodId"] = pd.to_numeric(df["foodId"], errors="coerce")
    df["nutrientId"] = pd.to_numeric(df["nutrientId"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["foodId", "nutrientId", "value"])

    df["foodId"] = df["foodId"].astype(int)
    df["nutrientId"] = df["nutrientId"].astype(int)

    required_ids = list(NUTRIENT_IDS.values())
    df = df[df["nutrientId"].isin(required_ids)]

    return df.sort_values(["foodId", "nutrientId"]).reset_index(drop=True)


def load_user_preferences(user_id):
    query = """
        SELECT
            userId,
            foodId,
            preference AS user_preference
        FROM user_foods;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df["userId"] = pd.to_numeric(df["userId"], errors="coerce")
    df["foodId"] = pd.to_numeric(df["foodId"], errors="coerce")
    df["user_preference"] = pd.to_numeric(df["user_preference"], errors="coerce")

    df = df.dropna(subset=["userId", "foodId"])
    df["userId"] = df["userId"].astype(int)
    df["foodId"] = df["foodId"].astype(int)

    df = df[df["userId"] == int(user_id)]

    return df.reset_index(drop=True)


def load_users():
    query = """
        SELECT
            id,
            name,
            surname,
            username,
            age,
            gender,
            height,
            weight
        FROM user;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df = clean_numeric_id(df, "id")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    df = df.dropna(subset=["age"])

    return df.sort_values("id").reset_index(drop=True)


def load_user_info(user_id):
    users = load_users()
    user_row = users[users["id"] == int(user_id)]

    if user_row.empty:
        raise ValueError(f"No real user found with id={user_id}")

    return user_row.iloc[0].to_dict()


def load_dri_for_user(user_id):
    user = load_user_info(user_id)

    age = int(user["age"])
    gender = str(user["gender"]).lower()

    query = """
        SELECT
            nutrient_id AS nutrientId,
            low_age,
            up_age,
            gender,
            RLL,
            RUL
        FROM dri;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    df["nutrientId"] = pd.to_numeric(df["nutrientId"], errors="coerce")
    df["low_age"] = pd.to_numeric(df["low_age"], errors="coerce")
    df["up_age"] = pd.to_numeric(df["up_age"], errors="coerce")
    df["RLL"] = pd.to_numeric(df["RLL"], errors="coerce")
    df["RUL"] = pd.to_numeric(df["RUL"], errors="coerce")

    df = df.dropna(subset=["nutrientId", "low_age", "up_age", "RLL", "RUL"])

    df["nutrientId"] = df["nutrientId"].astype(int)

    required_ids = list(NUTRIENT_IDS.values())

    df = df[
        (df["nutrientId"].isin(required_ids)) &
        (df["low_age"] <= age) &
        (df["up_age"] >= age) &
        (df["gender"].str.lower() == gender)
    ]

    if df.empty:
        raise ValueError(
            f"No DRI data found for user_id={user_id}, age={age}, gender={gender}"
        )

    return df[["nutrientId", "RLL", "RUL"]].sort_values("nutrientId").reset_index(drop=True)


def load_all_data(user_id):
    foods = load_foods()
    user_preferences = load_user_preferences(user_id)

    foods = foods.merge(
        user_preferences[["foodId", "user_preference"]],
        left_on="id",
        right_on="foodId",
        how="left"
    )

    foods["final_preference"] = foods["user_preference"].fillna(foods["preference"])

    food_nutrients = load_food_nutrients()
    nutrients = load_nutrients()
    dri = load_dri_for_user(user_id)
    food_groups = load_food_groups()

    return {
        "foods": foods,
        "food_nutrients": food_nutrients,
        "nutrients": nutrients,
        "dri": dri,
        "food_groups": food_groups,
    }


if __name__ == "__main__":
    print("Testing database loader...")

    users = load_users()
    print("\nUsers:")
    print(users.head())
    print("Users count:", len(users))

    foods = load_foods()
    print("\nFoods count:", len(foods))
    print(foods.head())

    food_nutrients = load_food_nutrients()
    print("\nFood nutrients sample:")
    print(food_nutrients.head())

    data = load_all_data(user_id=2)
    print("\nAll data loaded successfully for user_id=2")
    print("Foods:", data["foods"].shape)
    print("Food nutrients:", data["food_nutrients"].shape)
    print("DRI:")
    print(data["dri"])