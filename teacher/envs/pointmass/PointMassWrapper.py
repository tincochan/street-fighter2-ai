import csv
import numpy as np
import wandb
from envs.pointmass.pm_sparse.binary_pointmass import BinaryContextualPointMass
from envs.pointmass.pm_dense.original_pointmass import ContextualPointMass
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper
import random


class PointmassWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the PointMass environment.
        Inherits from our abstract class so that curriculums can be used"""
    def __init__(self, cur, env_type, beta, metrics, beta_plr, rho_plr):
        path = "envs/pointmass/task_datasets/cpm_train_100.csv"
        #path = "envs/pointmass/task_datasets/hard_data.csv"
        self.type_env = env_type
        self.metrics = metrics
        super(PointmassWrapper, self).__init__(cur, env_type, beta, path, beta_plr, rho_plr)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space_context  # context obs space for PM
        self.context1_ls = []
        self.context2_ls = []
        self.context3_ls = []

    def load_envs(self, path, env_type):
        self.contexts = self.read_csv_tasks(path)
        # Each context is considered a task. We create the environment as a multi-task collection of single tasks.
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]

            # Create a new single environment for each task
            if env_type == "binary":
                single_env = BinaryContextualPointMass(context=context)
            else:
                single_env = ContextualPointMass(context=context)

            single_env.reset()
            self.collection_envs.append(single_env)

    @staticmethod
    # Generate contexts and save them. Used only for harder tasks
    def _generate_contexts(num, harder_bool=True):
        if harder_bool:
            target_task_percentage = 0.5
            with open("task_datasets/hard_data.csv", "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(["ID", "c1", "c2", "c3"])
                for i in range(num):
                    if i < (1 - target_task_percentage) * num:
                        context1 = random.uniform(-4, 4)
                        context2 = random.uniform(0.5, 8)
                        context3 = random.uniform(0, 4)
                    elif i < (1 - target_task_percentage / 2) * num:
                        context1 = np.random.normal(loc=3, scale=0.5)
                        context2 = np.random.normal(loc=1, scale=0.5)
                        context3 = random.uniform(0, 4)  # context3 = abs(np.random.normal(loc=0, scale=1e-4))
                    else:
                        context1 = np.random.normal(loc=-3, scale=0.5)
                        context2 = np.random.normal(loc=1, scale=0.5)
                        context3 = random.uniform(0, 4)  # context3 = abs(np.random.normal(loc=0, scale=1e-4))
                    writer.writerow([i, context1, context2, context3])

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

    def step(self, action):
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        x = self.curr_task.step(action=action)
        self.num_steps += 1
        return x

    def reset(self):
        super(PointmassWrapper, self).select_next_task()
        self.num_steps = 0

        if self.metrics:
            # Record features for visualization
            if self.cur == "procurl-env" or self.cur == "iid" or self.cur == "procurl-val":
                _, context1, context2, context3 = self.get_context()
                self.context1_ls.append(abs(float(context1)))
                self.context2_ls.append(float(context2))
                self.context3_ls.append(float(context3))
                if len(self.context1_ls) == 100:  # Take the average of 100 episodes
                    wandb.log({"context1": sum(self.context1_ls)/len(self.context1_ls),
                               "context2": sum(self.context2_ls)/len(self.context2_ls),
                               "context3": sum(self.context3_ls)/len(self.context3_ls)})
                    self.context1_ls, self.context2_ls, self.context3_ls = [], [], []

        return self.curr_task.reset()


class PointmassWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the PointMass environment """
    def __init__(self, env_type):
        path = "envs/pointmass/task_datasets/cpm_train_100.csv"
        #path = "envs/pointmass/task_datasets/cpm_test_100.csv"  # Test set
        #path = "envs/pointmass/task_datasets/hard_data.csv"
        super(PointmassWrapperEval, self).__init__(env_type, path)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space_context

    def step(self, action):
        x = self.curr_task.step(action=action)
        return x

    def reset(self):
        next_id = self.pick_next_id()
        self.curr_task = self.collection_envs[next_id]
        return self.curr_task.reset()

    def load_envs(self, path, env_type):
        self.contexts = self.read_csv_tasks(path)
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]

            # Create a new environment for each task
            if env_type == "binary":
                single_env = BinaryContextualPointMass(context=context)
            else:
                single_env = ContextualPointMass(context=context)

            single_env.reset()
            self.collection_envs.append(single_env)

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

