from rgcn_map_encoder import load_graph
TOWN = "Town05"
data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")
print(f"mean: {mean}")
print(f"std:  {std}")
