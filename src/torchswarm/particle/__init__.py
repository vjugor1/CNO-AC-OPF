import torch
import numpy as np
from pyomo.environ import *

from torchswarm.utils.rotation_utils import (
    get_rotation_matrix,
    get_inverse_matrix,
    get_phi_matrix,
)
from torchswarm.utils.parameters import SwarmParameters


class CNOPyomoParticle:
    def __init__(self, dimensions, w=0.5, c1=2, c2=2, **kwargs):
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        classes = kwargs.get("classes") if kwargs.get("classes") else 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = kwargs.get("device") if kwargs.get("device") else self.device
        self.model = kwargs.get("init_model")()
        self.solver = kwargs.get("solver")
        # if kwargs.get("bounds"):
        #     self.bounds = kwargs.get("bounds")
        #     self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(
        #         dimensions, classes
        #     ).to(self.device) + self.bounds[1]
        # else:
        #     self.bounds = None
        #     self.position = torch.rand(dimensions, classes).to(self.device)
        self.solver.solve(self.model)
        self.position = self.vars_to_position()
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)

    def vars_to_position(self):
        dump = []
        for k in self.model.component_objects(Var):
            dump.append(np.array(list(k.get_values().values())))
        dump = np.concatenate(dump)
        return dump

    def position_to_vars(self):
        dump = self.position.numpy()
        curr_position = 0
        for k in self.model.component_objects(Var):
            curr_vals = k.get_values()
            var_chunk = dump[curr_position : curr_position + len(k)]
            fill_dict = {key: -val for key, val in zip(curr_vals.keys(), var_chunk)}
            k.set_values(fill_dict)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.pbest_position[i] - self.position[i])
                + self.c2 * r2 * (gbest_position[i] - self.position[i])
            )

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters

    def move(self):
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
        if self.bounds:
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])


class Particle:
    def __init__(self, dimensions, w=0.5, c1=2, c2=2, **kwargs):
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        classes = kwargs.get("classes") if kwargs.get("classes") else 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = kwargs.get("device") if kwargs.get("device") else self.device
        if kwargs.get("bounds"):
            self.bounds = kwargs.get("bounds")
            self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(
                dimensions, classes
            ).to(self.device) + self.bounds[1]
        else:
            self.bounds = None
            self.position = torch.rand(dimensions, classes).to(self.device)
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.pbest_position[i] - self.position[i])
                + self.c2 * r2 * (gbest_position[i] - self.position[i])
            )

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters

    def move(self):
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
        if self.bounds:
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])


class RotatedParticle(Particle):
    def __init__(self, dimensions, w, c1=2, c2=2, **kwargs):
        super(RotatedParticle, self).__init__(dimensions, w, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = (
            self.w * self.velocity
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c1, r1)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (self.pbest_position - self.position).float().to(self.device),
            )
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c2, r2)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (gbest_position - self.position).float().to(self.device),
            )
        )
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class ExponentiallyWeightedMomentumParticle(Particle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(ExponentiallyWeightedMomentumParticle, self).__init__(
            dimensions, 0, c1, c2, **kwargs
        )
        self.beta = beta
        self.momentum = torch.zeros((dimensions, 1)).to(self.device)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            momentum_t = (
                self.beta * self.momentum[i] + (1 - self.beta) * self.velocity[i]
            )
            self.velocity[i] = (
                momentum_t
                + self.c1 * r1 * (self.pbest_position[i] - self.position[i])
                + self.c2 * r2 * (gbest_position[i] - self.position[i])
            )
            self.momentum[i] = momentum_t
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class RotatedEWMParticle(ExponentiallyWeightedMomentumParticle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(RotatedEWMParticle, self).__init__(dimensions, beta, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = (
            momentum_t
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c1, r1)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (self.pbest_position - self.position).float().to(self.device),
            )
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c2, r2)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (gbest_position - self.position).float().to(self.device),
            )
        )

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters
