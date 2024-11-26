import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from pygad import GA
import joblib

# Load your dataset
df = pd.read_csv("DataFiles/dataset.csv")
df = df.dropna()

X = df.drop('Target', axis=1).values  
y = df['Target'].values  

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Grey Wolf Optimization for feature selection
def feature_selection_obj_function(solution):
    selected_features = np.where(solution > 0.5)[0]
    if len(selected_features) == 0:
        return float('inf')
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    clf = RandomForestClassifier()
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy

# Grey Wolf Optimization (GWO) implementation
def gwo(num_agents, max_iter, dim, lb, ub, obj_func):
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    positions = np.random.uniform(low=lb, high=ub, size=(num_agents, dim))

    for l in range(max_iter):
        print(f"Iteration {l+1}/{max_iter}")
        
        for i in range(num_agents):
            fitness = obj_func(positions[i])
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i].copy()

        a = 2 - l * (2 / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i][j] = np.clip((X1 + X2 + X3) / 3, lb[j], ub[j])

    return alpha_pos

# Genetic Algorithm (GA) for feature selection
def fitness_func(ga_instance, solution, solution_idx, X_scaled, y):
    selected_features = np.where(solution == 1)[0]  
    X_selected = X_scaled[:, selected_features]  
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def run_ga(X_scaled, y, num_generations=10, num_parents_mating=5, sol_per_pop=10):
    num_genes = X_scaled.shape[1]
    
    ga_instance = GA(num_generations=num_generations,
                     num_parents_mating=num_parents_mating,
                     fitness_func=lambda ga_instance, solution, solution_idx: fitness_func(ga_instance, solution, solution_idx, X_scaled, y),
                     sol_per_pop=sol_per_pop,
                     num_genes=num_genes,
                     gene_type=int,
                     gene_space=[0, 1])

    ga_instance.run()

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    selected_features = np.where(best_solution == 1)[0]
    
    return best_solution, selected_features

# GWO parameters
num_agents = 10
max_iter = 50
dim = X_train.shape[1]
lb, ub = [0] * dim, [1] * dim

# Run GWO
print("\nRunning Grey Wolf Optimization (GWO)...")
best_solution_gwo = gwo(num_agents, max_iter, dim, lb, ub, feature_selection_obj_function)
selected_features_gwo = np.where(best_solution_gwo > 0.5)[0]

# Subset dataset based on GWO-selected features
X_train_gwo = X_train[:, selected_features_gwo]
X_test_gwo = X_test[:, selected_features_gwo]

# Run GA for fine-tuning feature selection or hyperparameters
print("\nRunning Genetic Algorithm (GA)...")
best_solution_ga, selected_features_ga = run_ga(X_scaled, y)

# Final features selected (using GA)
X_train_final = X_train[:, selected_features_ga]
X_test_final = X_test[:, selected_features_ga]



# Classifiers to evaluate
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "LightGBM": lgb.LGBMClassifier(verbose=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Evaluate performance of classifiers with GWO-selected features
print("\nEvaluating classifiers with GWO-selected features...")
metrics_gwo = {}
for name, clf in classifiers.items():
    if name == "ANN":
        clf.fit(X_train_gwo, y_train, epochs=10, batch_size=10, verbose=0)
        y_pred = (clf.predict(X_test_gwo) > 0.5).astype("int32")
    else:
        clf.fit(X_train_gwo, y_train)
        y_pred = clf.predict(X_test_gwo)
    
    metrics_gwo[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }

# Evaluate performance of classifiers with GA-selected features
print("\nEvaluating classifiers with GA-selected features...")
metrics_ga = {}
for name, clf in classifiers.items():
    if name == "ANN":
        clf.fit(X_train_final, y_train, epochs=10, batch_size=10, verbose=0)
        y_pred = (clf.predict(X_test_final) > 0.5).astype("int32")
    else:
        clf.fit(X_train_final, y_train)
        y_pred = clf.predict(X_test_final)
    
    metrics_ga[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }

# Hybrid evaluation using features selected from both GWO and GA
combined_selected_features = np.unique(np.concatenate((selected_features_gwo, selected_features_ga)))
X_train_hybrid = X_train[:, combined_selected_features]
X_test_hybrid = X_test[:, combined_selected_features]

# Evaluate performance of classifiers with hybrid-selected features
print("\nEvaluating classifiers with Hybrid-selected features...")
metrics_hybrid = {}
for name, clf in classifiers.items():
    if name == "ANN":
        clf.fit(X_train_hybrid, y_train, epochs=10, batch_size=10, verbose=0)
        y_pred = (clf.predict(X_test_hybrid) > 0.5).astype("int32")
    else:
        clf.fit(X_train_hybrid, y_train)
        y_pred = clf.predict(X_test_hybrid)
    
    metrics_hybrid[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }

# Output performance metrics
print("\nPerformance metrics with GWO-selected features:")
for name, metrics in metrics_gwo.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

print("\nPerformance metrics with GA-selected features:")
for name, metrics in metrics_ga.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

print("\nPerformance metrics with Hybrid-selected features:")
for name, metrics in metrics_hybrid.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


final_model = RandomForestClassifier()  
final_model.fit(X_train_hybrid, y_train)

# Save the model to an .h5 file
joblib.dump(final_model, 'static/newmodel.pkl')

