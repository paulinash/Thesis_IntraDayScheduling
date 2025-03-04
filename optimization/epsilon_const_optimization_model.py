''' This module contains the base class for the optimization model. '''
import pyomo.environ as pyo
import pandas as pd

from optimization_model import BaseOptimizationModel
from utils import simpsons_rule, cdf_formula, cdf_formula_numpy, pdf_formula, dynamic_bounds


class EpsilonConstraintOptimizationModel(BaseOptimizationModel):
    ''' This model represents the optimization model for the implementation of the epsilon constraint method in MOO '''

    def __init__(self, input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilon, self_suff=True):
        self.day_ahead_schedule = day_ahead_schedule
        self.e_nom_old = e_nom
        self.e_prob_min_old = e_prob_min
        self.e_prob_max_old = e_prob_max
        self.epsilon = epsilon
        # To decide whether to promot self sufficiency or cost efficiency
        self.self_suff = self_suff

        super().__init__(input_data)
        self.name = "Epsilon Constraint Intra Day Optimization Model"
                                                            

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

        ####################################### epsilon constraints #####################################################
        def constr_epsilon(model):
            # self sufficiency ist constrained
            #sum(pg_nom^+**2 + pg_nom^-**2) <= epsilon_self_suff
            if self.self_suff:
                return sum(model.pg_nom_plus[t]**2 + model.pg_nom_minus[t]**2 for t in model.time) <= self.epsilon
            else:#sum(pg_nom^+**2 - pg_nom^-**2) <= epsilon_cost_eff
                return sum(model.pg_nom_plus[t]**2 - model.pg_nom_minus[t]**2 for t in model.time) <= self.epsilon
        self.model.constr_epsilon = pyo.Constraint(rule=constr_epsilon)

    def _define_objective_function(self):
        def objective(model):
            return sum((model.pg_nom[t] - self.day_ahead_schedule[t])**2 + 
                       (-model.prob_low[t]*model.exp_pg_low[t]+ 
                        model.prob_high[t]*model.exp_pg_high[t]) 
                        for t in model.time)
        self.model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)


