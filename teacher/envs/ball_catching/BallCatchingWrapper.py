import math
import csv
import numpy as np
import random
import os.path
from envs.ball_catching.ball_catching_env import ContextualBallCatching
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper


class BallCatchingWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the Ballcatching environment.
        Inherits from our abstract gym wrapper class so that different
        curriculum strategies can be used on that environment"""
    def __init__(self, cur, beta, beta_plr, rho_plr):
        path = "envs/ball_catching/task_datasets/ball_catching_train_data.csv"
        self.type_env = "info-based"
        self.single_env = None
        super(BallCatchingWrapper, self).__init__(cur, self.type_env, beta, path, beta_plr, rho_plr)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space

    def load_envs(self, path, env_type):
        # If no context exist generate them, otherwise read them
        if os.path.exists(path):
            self.contexts = self.read_csv_tasks(path)
        else:
            self._generate_contexts(num=100)  # Generate 100 contexts based on distributions
            self.contexts = self.read_csv_tasks(path)

        # Each context is considered a task. We create the environment as a multi-task collection of single tasks.
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]
            # Create a new environment for each task
            self.single_env = ContextualBallCatching(context=context)
            self.single_env.reset_model()
            self.collection_envs.append(self.single_env)

    @staticmethod
    # Generate contexts and save them
    def _generate_contexts(num):
        with open("envs/ball_catching/task_datasets/ball_catching_train_data.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["ID", "target X", "target Y", "distance"])
            for i in range(num):

                # Context 1,2 control the target ball position
                # Phi angle
                context1 = random.uniform(0.125*math.pi, 0.5*math.pi)
                # r
                context2 = random.uniform(0.6, 1.1)
                # Distance dx from which ball is thrown
                context3 = random.uniform(0.75, 4)
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
        self.num_steps += 1
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        observation, reward, done, info = self.curr_task.real_step(action=action)
        return observation, reward, done, info

    def reset(self):
        super(BallCatchingWrapper, self).select_next_task()
        self.num_steps = 0
        return self.curr_task.reset_model()


class BallCatchingWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the Ballcatching environment """
    def __init__(self):
        self.type_env = "info-based"
        path = "envs/ball_catching/task_datasets/ball_catching_train_data.csv"  # train data
        # path = "envs/ball_catching/task_datasets/ball_catching_test_data.csv"  # test data
        self.single_env = None
        super(BallCatchingWrapperEval, self).__init__(self.type_env, path)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space

    def step(self, action):
        x = self.curr_task.real_step(action=action)
        return x

    def reset(self):
        next_id = self.pick_next_id()
        self.curr_task = self.collection_envs[next_id]
        return self.curr_task.reset_model()

    def load_envs(self, path, env_type):

        # If no context exist generate them. Otherwise read them
        if os.path.exists(path):
            self.contexts = self.read_csv_tasks(path)
        else:
            num_of_tasks = 100
            self._generate_contexts(num_of_tasks)
            self.contexts = self.read_csv_tasks(path)

        # Each context is considered a task. We create the environment as a multi-task collection of single tasks.
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]
            # Create a new environment for each task
            self.single_env = ContextualBallCatching(context=context)
            self.single_env.reset_model()
            self.collection_envs.append(self.single_env)

    @staticmethod
    # Generate contexts and save them
    def _generate_contexts(num):
        with open("envs/ball_catching/task_datasets/ball_catching_train_data.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["ID", "target X", "target Y", "distance"])
            for i in range(num):

                # Context 1,2 control the target ball position
                # Phi angle
                context1 = random.uniform(0.125*math.pi, 0.5*math.pi)
                # r
                context2 = random.uniform(0.6, 1.1)
                # Distance dx from which ball is thrown
                context3 = random.uniform(0.75, 4)
                writer.writerow([i, context1, context2, context3])

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data
