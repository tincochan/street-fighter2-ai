import csv
import numpy as np
import os.path
from gym import spaces
from envs.ant_goal.ant import AntEnv
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper


class AntGoalWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the AntGoal environment.
        Inherits from our abstract gym wrapper class so that different
        curriculum strategies can be used on that environment"""
    def __init__(self, cur, beta, beta_plr, rho_plr):
        path = "envs/ant_goal/task_datasets/ant_goal_train_data.csv"
        self.type_env = "non-binary"
        self.single_env = AntEnv()
        super(AntGoalWrapper, self).__init__(cur, self.type_env, beta, path, beta_plr, rho_plr)
        self.action_space = self.single_env.action_space
        obs_size = 2 + len(self.single_env.observation_space.low)
        self.observation_space = spaces.Box(-np.inf * np.ones(obs_size), np.inf * np.ones(obs_size))
        self.num_steps = 0
        self.x_goal = 0
        self.y_goal = 0

    def load_envs(self, path, env_type):
        # If no context exist generate them, otherwise read them
        if os.path.exists(path):
            self.contexts = self.read_csv_tasks(path)
        else:
            self._generate_contexts(num=50)  # Generate 50 contexts based on distribution
            self.contexts = self.read_csv_tasks(path)
        self.collection_envs = self.contexts

    @staticmethod
    def _generate_contexts(num):
        with open("envs/ant_goal/task_datasets/ant_goal_train_data.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["ID,x,y"])
            for i in range(num):
                a = np.random.random() * 2 * np.pi
                r = 3 * np.random.random() ** 0.5
                goal = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
                writer.writerow([i, goal[0], goal[1]])

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

    def set_goal(self, cur_id):
        self.x_goal = float(self.contexts[cur_id][1])
        self.y_goal = float(self.contexts[cur_id][2])

    def step(self, action):
        self.num_steps += 1
        self.single_env.do_simulation(action, self.single_env.frame_skip)
        xposafter = np.array(self.single_env.get_body_com("torso"))
        self.set_goal(self.cur_id)
        goal_reward = 10 * np.exp(- 2 * np.linalg.norm(np.array([self.x_goal, self.y_goal]) - xposafter[:2]))
        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.single_env.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.single_env.state_vector()
        done = False
        ob = self._get_obs()
        if self.num_steps >= 200:
            done = True
        next_state = np.concatenate(([self.x_goal, self.y_goal], ob))
        return next_state, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward
        )

    def reset(self):
        super(AntGoalWrapper, self).select_next_task()
        self.num_steps = 0
        self.set_goal(self.cur_id)
        obs = self.single_env.reset()
        next_state = np.concatenate(([self.x_goal, self.y_goal], obs))
        return next_state

    def _get_obs(self):
        return np.concatenate([
            self.single_env.sim.data.qpos.flat,
            self.single_env.sim.data.qvel.flat,
        ])


class AntGoalWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the AntGoal environment """
    def __init__(self):
        self.type_env = "non-binary"
        path = "envs/ant_goal/task_datasets/ant_goal_train_data.csv"
        self.single_env = AntEnv()
        super(AntGoalWrapperEval, self).__init__(self.type_env, path)
        self.action_space = self.single_env.action_space
        obs_size = 2 + len(self.single_env.observation_space.low)
        self.observation_space = spaces.Box(-np.inf * np.ones(obs_size), np.inf * np.ones(obs_size))
        self.done = False
        self.num_steps = 0
        self.x_goal = 0
        self.y_goal = 0
        self.curr_eval_id = 0

    def set_goal(self, cur_id):
        self.x_goal = float(self.contexts[cur_id][1])
        self.y_goal = float(self.contexts[cur_id][2])

    def step(self, action):
        self.num_steps += 1
        self.single_env.do_simulation(action, self.single_env.frame_skip)
        xposafter = np.array(self.single_env.get_body_com("torso"))
        self.set_goal(self.curr_eval_id)

        goal_reward = 10 * np.exp(- 2 * np.linalg.norm(np.array([self.x_goal, self.y_goal]) - xposafter[:2]))

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.single_env.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.single_env.state_vector()
        done = False
        ob = self._get_obs()
        if self.num_steps >= 200:
            done = True
        next_state = np.concatenate(([self.x_goal, self.y_goal], ob))

        return next_state, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def reset(self):
        self.curr_eval_id = self.pick_next_id()
        self.num_steps = 0
        self.set_goal(self.curr_eval_id)
        obs = self.single_env.reset()
        next_state = np.concatenate(([self.x_goal, self.y_goal], obs))
        return next_state

    def load_envs(self, path, env_type):
        # If no context exist generate them, otherwise read them
        if os.path.exists(path):
            self.contexts = self.read_csv_tasks(path)
        else:
            self._generate_contexts(num=50)
            self.contexts = self.read_csv_tasks(path)
        self.collection_envs = self.contexts

    @staticmethod
    def _generate_contexts(num):
        with open("envs/ant_goal/task_datasets/ant_goal_train_data.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["ID,x,y"])
            for i in range(num):
                a = np.random.random() * 2 * np.pi
                r = 3 * np.random.random() ** 0.5
                goal = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
                writer.writerow([i, goal[0], goal[1]])

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

    def _get_obs(self):
        return np.concatenate([
            self.single_env.sim.data.qpos.flat,
            self.single_env.sim.data.qvel.flat,
        ])
