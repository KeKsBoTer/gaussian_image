import torch
import numpy as np
import sys

if __name__ == "__main__":
        
    file_in = sys.argv[1]
    data = torch.load(file_in)

    width = int(sys.argv[2])
    height = int(sys.argv[2])

    pos = data["_xyz"].tanh().cpu().numpy()
    color = (data["_features_dc"]).cpu().numpy()
    scaling = torch.abs(data["_scaling"]+data["bound"]).cpu().numpy()
    rotation = torch.sigmoid(data["_rotation"]).cpu().numpy()*2*torch.pi    

    mask = (scaling > 0).all(axis=1)

    np.savez_compressed(sys.argv[4],**{
        "xyz":pos[mask].astype(np.float16),
        "scaling":scaling[mask].astype(np.float16),
        "rotation":rotation[mask].astype(np.float16),
        "color":color[mask].astype(np.float16),
        "resolution":np.array([width,height],dtype=np.uint32)
    })