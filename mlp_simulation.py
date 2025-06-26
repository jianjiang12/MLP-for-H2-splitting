import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import fs as fs_conversion

# Define the Neural Network model
class NeuralNet(nn.Module):
    def __init__(self, layer_size=128):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Function to update velocities based on MLP forces
def update_velocities(velocities, positions, timestep, model, device):
    fratio = 4.9256340596086E5  # Unit conversion factor
    data = torch.tensor(positions, dtype=torch.float32).to(device)
    
    if positions[0] <= 15:
        force = model(data)
    else:
        data = torch.tensor([15, 15 + positions[1] - positions[0]], dtype=torch.float32).to(device)
        force = model(data)
    
    f_value = force.detach()
    velocities[0][2] += f_value[0] * fratio * timestep
    velocities[1][2] += f_value[1] * fratio * timestep
    return velocities

# Load the MLP model
def load_mlp_model(model_path, device="cpu"):
    model = NeuralNet()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Main Molecular Dynamics simulation
def run_md_simulation():
    # Simulation parameters
    initial_velocity = 1E5 * 1E-2  # Å/ps
    initial_positions = np.array([[5.0, 6.15, 5.0], [5.0, 6.15, 6.1]])
    velocities = np.array([[0.0, 0.0, initial_velocity], [0.0, 0.0, initial_velocity]])
    timestep = 1E-8  # ps
    max_steps = 1000000
    output_interval = 1000
    end_position = 60.0
    max_separation = 4.2

    # Initialize system
    system = Atoms("H2", positions=initial_positions)
    natoms = len(system)
    trajectory = []
    
    # Load MLP model
    device = "cpu"  # Use "cuda" if GPU is available
    model = load_mlp_model("force_128_linear.pth", device)
    
    # Simulation loop
    step = 0
    total_time = 0.0
    while step < max_steps:
        # Update velocities and positions
        velocities = update_velocities(
            velocities,
            [system[0].position[2], system[1].position[2]],
            timestep,
            model,
            device
        )
        
        # Adaptive timestep
        timestep = min(
            abs(0.0001 / velocities[0][2]),
            abs(0.0001 / velocities[1][2]),
            1E-5
        )
        
        # Update positions
        system[0].position += velocities[0] * timestep
        system[1].position += velocities[1] * timestep
        total_time += timestep
        step += 1
        
        # Check termination conditions
        if (system[natoms-2].position[2] >= end_position or 
            abs(system[0].position[2] - system[1].position[2]) > max_separation):
            break
            
        # Save trajectory snapshot
        if step % output_interval == 0:
            trajectory.append(system.copy())
    
    # Save final trajectory
    write("traj_linear_1e5.xyz", trajectory)
    print(f"Simulation completed after {step} steps and {total_time:.2e} ps")

if __name__ == "__main__":
    run_md_simulation()