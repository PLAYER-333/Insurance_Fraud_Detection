import pickle
import matplotlib.pyplot as plt
from sklearn import tree

MODEL_PATH = "outputs/random_forest_model.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("=" * 60)
print("RANDOM FOREST VISUALIZER")
print("=" * 60)
print(f"Trees in forest: {len(model.estimators_)}")

# ---------------------------
# Feature Importance
# ---------------------------
# if hasattr(model, "feature_importances_"):
#     plt.figure(figsize=(12, 6))
#     plt.bar(
#         range(len(model.feature_importances_)),
#         model.feature_importances_
#     )
#     plt.title("Random Forest Feature Importance")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Importance")
#     plt.tight_layout()
#     plt.show()

# ---------------------------
# Visualize First Tree
# ---------------------------
plt.figure(figsize=(24, 12))
tree.plot_tree(
    model.estimators_[0],
    filled=True,
    rounded=True,
    max_depth=5
)
plt.title("Decision Tree #1 (First 5 Levels)")
plt.show()


plt.figure(figsize=(24, 12))
tree.plot_tree(
    model.estimators_[0],
    filled=True,
    rounded=True,
)
plt.title("Decision Tree #1 Original Size")
plt.show()

# ---------------------------
# Forest Summary
# ---------------------------
print("\nForest Summary")
print("-" * 40)
print(f"Number of Trees: {len(model.estimators_)}")
print(f"Input Features: {model.n_features_in_}")

print("\nHow Random Forest Works:")
print("1. Each tree receives the same input.")
print("2. Every tree makes its own prediction.")
print("3. All 200 trees vote.")
print("4. Majority vote becomes the final prediction.")