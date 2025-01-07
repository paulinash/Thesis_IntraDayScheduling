''' This module contains the base class for the optimization model. '''
import pyomo.environ as pyo
import pandas as pd

from optimization_model import BaseOptimizationModel
from utils import simpsons_rule, cdf_formula, cdf_formula_numpy, pdf_formula, dynamic_bounds


class IntraDayOptimizationModel(BaseOptimizationModel):
    ''' Base class for the optimization model. '''

    def __init__(self, input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max):
        self.day_ahead_schedule = day_ahead_schedule
        self.e_nom_old = e_nom
        self.e_prob_min_old = e_prob_min
        self.e_prob_max_old = e_prob_max

        super().__init__(input_data)
        self.name = "Intra Day Optimization Model"
                                                            

    def _define_constraints(self):
        ''' Defines the constraints of the model. This includes equality and inequality constraints. '''
        super()._define_constraints()

        ######################################## battery constraints ###################################################
        def constr_e_limit_min(model, t):
            ''' e_nom[t] + e_min[t] >= e_nom_old + e_prob_min_old '''
            return model.e_nom[t] + model.e_min[t] >= self.e_nom_old[t] + self.e_prob_min_old[t]
        self.model.constr_e_limit_min = pyo.Constraint(self.model.time_e, rule=constr_e_limit_min)

        def constr_e_limit_max(model, t):
            ''' e_nom[t] + e_max[t] <= e_nom_old + e_prob_max_old '''
            return model.e_nom[t] + model.e_max[t] <= self.e_nom_old[t] + self.e_prob_max_old[t]
        self.model.constr_e_limit_max = pyo.Constraint(self.model.time_e, rule=constr_e_limit_max)


    def _define_objective_function(self):
        def objective(model):
            return sum(
                (model.pg_nom[t] - self.day_ahead_schedule[t])**2
                + (-model.prob_low[t]*model.exp_pg_low[t]
                + model.prob_high[t]*model.exp_pg_high[t])
                + self.c11 * model.pg_nom_plus[t]**2 
                + self.c21 * model.pg_nom_minus[t]**2 
                for t in model.time) 
        self.model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)



    def solve(self):
        solver = pyo.SolverFactory('ipopt')
        #solver.options['mu_strategy'] = 1       
        #solver.options['tol'] = 1e-8            
        #solver.options['acceptable_tol'] = 1e-8 
        #solver.options['max_step'] = 1e-1 
        solver.options['max_iter'] = 5000
        result = solver.solve(self.model, tee=True)
        return result
    