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
    def __init__(self, max_users, num_products):
        self.matrix = np.zeros((max_users, num_products))
        self.user_index_map = {}  # Mapping real user IDs to matrix indices
        self.next_available_index = 0

    def add_user(self, user_id):
        """Assigns a user ID to an available matrix index."""
        if user_id not in self.user_index_map:
            if self.next_available_index < self.matrix.shape[0]:
                self.user_index_map[user_id] = self.next_available_index
                self.next_available_index += 1
            else:
                print(f"Warning: Maximum users reached, cannot add user {user_id}.")

    def update_interaction(self, user_id, product_id, value):
        """Logs user-product interactions if the user exists."""
        if user_id in self.user_index_map:
            matrix_index = self.user_index_map[user_id]
            self.matrix[matrix_index][product_id] = value
        else:
            print(f"Warning: User {user_id} does not exist in interaction store.")

    def get_interaction(self, user_id, product_id):
        """Retrieves the interaction value if the user exists."""
        if user_id in self.user_index_map:
            matrix_index = self.user_index_map[user_id]
            return self.matrix[matrix_index][product_id]
        return 0

# KD-Tree for Product Similarity
class ProductMatcher:
    """
    A class to find similar products using KD-Tree.
    """
    def __init__(self, product_features):
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

    print("Registered Users (Before Deletion):")
    for user_id, info in user_registry.user_records.items():
        print(f"User ID: {user_id}, Info: {info}")

    # Remove User 101
    user_registry.delete_user(101)

    print("Registered Users (After Deletion of User 101):")
    for user_id, info in user_registry.user_records.items():
        print(f"User ID: {user_id}, Info: {info}")

    # InteractionStore Test
    print("\n--- InteractionStore Test ---")
    interaction_store = InteractionStore(max_users=5, num_products=5)

    # Register Users
    interaction_store.add_user(102)  # Register Bob in the interaction store

    # Test Storing an Interaction
    interaction_store.update_interaction(102, 2, 4.5)  # Bob rates Product 2 with 4.5
    interaction_store.update_interaction(101, 3, 3.0)  # Alice was removed, should warn

    # Retrieve stored and non-existing interactions
    print("Interaction Value (User 102, Product 2):", interaction_store.get_interaction(102, 2))
    print("Interaction Value (User 101, Product 3 - No Interaction):", interaction_store.get_interaction(101, 3))

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
