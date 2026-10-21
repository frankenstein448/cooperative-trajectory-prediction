import pickle

with open('topology/Town05_topology.pkl', 'rb') as f:
    edges = pickle.load(f)

xs = [e['start_pos'][0] for e in edges] + [e['end_pos'][0] for e in edges]
ys = [e['start_pos'][1] for e in edges] + [e['end_pos'][1] for e in edges]

print(f"Town05 topology x range: {min(xs):.2f} to {max(xs):.2f}")
print(f"Town05 topology y range: {min(ys):.2f} to {max(ys):.2f}")
print(f"\nV2X-Sim agent position to check: (87.72, 105.93)")