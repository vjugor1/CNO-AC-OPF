import numpy as np
from numba import njit
from pyomo.environ import *
from torchswarm.functions import Function
import time
from copy import deepcopy
import torch
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
        self.bounds = None
        res = self.solver.solve(self.model)
        if str(res.solver.termination_condition) != 'ok':
            self.model = kwargs.get("init_model")()
        self.position = torch.tensor(torch.rand(dimensions, classes).to(self.device).flatten())
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)

    def vars_to_position(self):
        dump = []
        for k in self.model.component_objects(Var):
            dump.append(np.array(list(k.get_values().values())))
        dump = np.concatenate(dump)
        return dump
    
    def move_to_KKT(self):
        res = self.solver.solve(self.model)
        if str(res.solver.termination_condition) != 'infeasible':
            new_position = torch.tensor(self.vars_to_position())
            # if len(new_position) != 62:
                # print("lmao")
            self.position = new_position

    def position_to_vars(self):
        dump = self.position.numpy()
        curr_position = 0
        for k in self.model.component_objects(Var):
            curr_vals = k.get_values()
            var_chunk = dump[curr_position : curr_position + len(k)]
            curr_position += len(k)
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


class ObjectiveCNO(Function):
    def __init__(
        self,
        model_example
    ) -> None:
        super().__init__()
        self.model = model_example

    def __call__(self, inp_):
        curr_position = 0
        inp = inp_.numpy()
        for k in self.model.component_objects(Var):
            curr_vals = k.get_values()
            var_chunk = inp[curr_position : curr_position + len(k)]
            curr_position += len(k)
            fill_dict = {key: val for key, val in zip(curr_vals.keys(), var_chunk)}
            k.set_values(fill_dict)
        return self.model.obj()

    def evaluate(self, inp):
        return self.__call__(inp)

class SwarmOptimizerCNOPyomo:
    def __init__(self, dimensions, swarm_size, particle=None, **kwargs):
        self.swarm_size = swarm_size
        if not particle:
            self.particle = CNOPyomoParticle
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.swarm = []
        self.gbest_position = torch.Tensor([0]).to(device)
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        for i in range(self.swarm_size):
            self.swarm.append(self.particle(dimensions, **kwargs))

    def optimize(self, function):
        self.fitness_function = function

    def run(self, verbosity=True):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0
        # --- Run
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            # --- Move to KKT
            for particle in self.swarm:
                particle.move_to_KKT()
            # --- Set PBest
            for particle in self.swarm:
                fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if (particle.pbest_value > fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_position = particle.position.clone()
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_cadidate:
                    self.gbest_value = best_fitness_cadidate
                    self.gbest_position = particle.position.clone()
                    self.gbest_particle = deepcopy(particle)
            r1s = []
            r2s = []
            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            toc = time.monotonic()
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()
            if verbosity == True:
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                      .format(iteration + 1, self.gbest_value.item(), toc - tic))
        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        swarm_parameters.c1 = self.gbest_particle.c1
        swarm_parameters.c2 = self.gbest_particle.c2
        return swarm_parameters
