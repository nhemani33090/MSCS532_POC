import numpy as np
from scipy.spatial import KDTree

# Hash Table for User Data
class UserRegistry:
    """
    A class to manage user information using a hash table.
    """
    def __init__(self):
        # Initialize an empty dictionary called user_records.
        self.user_records = {}

    def add_user(self, user_id, user_info):
        """
        Adds or updates user information.
        :param user_id: Unique identifier for the user.
        :param user_info: Dictionary containing user details.
        """
        self.user_records[user_id] = user_info

    def get_user(self, user_id):
        """
        Retrieves user information.
        :param user_id: Unique identifier for the user.
        :return: User details if found, otherwise None.
        """
        return self.user_records.get(user_id)

    def delete_user(self, user_id):
        """
        Removes a user from the registry.
        :param user_id: Unique identifier for the user.
        """
        if user_id in self.user_records:
            del self.user_records[user_id]

# Matrix Representation for User-Product Interactions
class InteractionStore:
    """
    A class to manage user-product interaction data using a matrix.
    """
    def __init__(self, num_users, num_products):
        self.matrix = np.zeros((num_users, num_products))

    def update_interaction(self, user_id, product_id, value):
        """
        Updates the rating for a specific user-product interaction.
        :param user_id: ID of the user.
        :param product_id: ID of the product.
        :param value: Interaction value or rating.
        """
        self.matrix[user_id][product_id] = value

    def get_interaction(self, user_id, product_id):
        """
        Retrieves the interaction value for a specific user-product interaction.
        :param user_id: ID of the user.
        :param product_id: ID of the product.
        :return: Interaction value or rating.
        """
        return self.matrix[user_id][product_id]

class ProductMatcher:
    """
    A class to find similar products using KD-Tree.
    """
    def __init__(self, product_features):
        # Initialize a data structure to store product features for similarity queries.
        self.tree = KDTree(product_features)

    def find_similar_products(self, query_features, k=5):
        """
        Identifies the top k most similar products to the given query features.
        :param query_features: Feature vector of the product.
        :param k: Number of similar products to find.
        :return: Indices and distances of the nearest neighbors.
        """
        distances, indices = self.tree.query(query_features, k=k)
        return indices, distances

# Testing the implemented functionalities
if __name__ == "__main__":
    # UserRegistry Test
    print("\n--- UserRegistry Test ---")
    user_registry = UserRegistry()
    user_registry.add_user(101, {"name": "Alice", "preferences": ["electronics", "books"]})
    user_registry.add_user(102, {"name": "Bob", "preferences": ["clothing", "shoes"]})
    print("Retrieving User 101:", user_registry.get_user(101))
    print("Registered Users:")
    for user_id, info in user_registry.user_records.items():
        print(f"User ID: {user_id}, Info: {info}")

    # InteractionStore Test
    print("\n--- InteractionStore Test ---")
    interaction_store = InteractionStore(num_users=5, num_products=5)
    interaction_store.update_interaction(0, 2, 4.5)  # User 0 rates Product 2 with a 4.5 rating
    print("Interaction Value (User 0, Product 2):", interaction_store.get_interaction(0, 2))
    print("User-Product Interaction Matrix:")
    print(interaction_store.matrix)

    # ProductMatcher Test
    print("\n--- ProductMatcher Test ---")
    product_features = np.array([[1.0, 0.5], [0.9, 0.4], [0.8, 0.3]])
    product_matcher = ProductMatcher(product_features)
    query = np.array([[1.0, 0.6]])
    indices, distances = product_matcher.find_similar_products(query, k=2)
    print("Similar Product Indices:", indices)
    print("Similar Product Distances:", distances)
