import numpy as np

path = '/home/karra/V2X-Sim-det/train/agent1/77_23/0.npy'
d = np.load(path, allow_pickle=True).item()

target_id = d['target_agent_id']
trans = d['trans_matrices']  # [6, 4, 4]

agent_pos_world = trans[target_id, :3, 3]  # (x, y, z)
print(f"target_agent_id: {target_id}")
print(f"trans_matrices shape: {trans.shape}")
print(f"agent world position (x,y,z): {agent_pos_world}")

# also print all 6 agents' positions for sanity check
for i in range(trans.shape[0]):
    print(f"  agent {i} pos: {trans[i, :3, 3]}")