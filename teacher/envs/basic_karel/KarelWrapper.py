from gym import spaces
import numpy as np
import json
from glob import glob
import copy
import wandb
from envs.basic_karel.utils import symb_2_bitmap
from envs.basic_karel.KarelCompiler import SymbolicKarel
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper


class KarelGymWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the Basic Karel environment.
        Inherits from our abstract gym wrapper class so that curriculums can be used"""
    def __init__(self, cur, beta, metrics, beta_plr, rho_plr):

        self.actions = {"move": 0, "turnLeft": 1, "turnRight": 2, "pickMarker": 3, "putMarker": 4, "finish": 5}
        self.opt_actions = []
        self.trajectories = []
        path = "envs/basic_karel/data/train/"
        self.type_env = "binary"
        self.metrics = metrics

        super(KarelGymWrapper, self).__init__(cur, self.type_env, beta, path, beta_plr, rho_plr)
        print("Number of train tasks", self.num_of_tasks)
        # Deepcopy original tasks for safe resetting
        self.original_envs = copy.deepcopy(self.collection_envs)

        # To log features
        self.ls_features0 = []
        self.ls_features1 = []
        self.ls_features2 = []
        self.ls_features3 = []
        self.ls_features4 = []
        self.ls_features5 = []
        self.ls_features6 = []

        # Initialization to define input size
        self.cur_id = 0
        self.curr_task = copy.deepcopy(self.collection_envs[self.cur_id])

        self.input_size = 5 * self.curr_task["gridsz_num_rows"] * self.curr_task["gridsz_num_cols"] + 2 * 4
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 1, [self.input_size], np.int32)
        self.Done = False
        self.reward = 0
        # Horizon added for training
        self.H = 100

    def load_envs(self, path, env_type):
        # Load all tasks for training
        for t_name in sorted(glob(f'{path}/task/*.json')):
            self.collection_envs.append(self.read_csv_tasks(t_name))
        for tr_name in sorted(glob(f'{path}/seq/*.json')):
            act_t = []
            # Transform dictionary to sequence of optimal actions
            actions = self.read_csv_tasks(tr_name)["sequence"]
            for action in actions:
                act_t.append(self.actions[action])
            self.opt_actions.append(act_t)
            self.trajectories.append(self.read_csv_tasks(tr_name))

    @staticmethod
    def read_csv_tasks(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data

    # To get features for visualization
    def get_features(self, cur_id):
        task_grid = self.collection_envs[cur_id]
        trajectory = self.trajectories[cur_id].get('sequence')
        feature0 = len(trajectory)  # Total steps
        if "putMarker" not in trajectory and "pickMarker" not in trajectory:
            feature1 = 1
        else:
            feature1 = 0
        if "pickMarker" in trajectory and "putMarker" not in trajectory:
            feature2 = 1
        else:
            feature2 = 0
        if "putMarker" in trajectory and "pickMarker" not in trajectory:
            feature3 = 1
        else:
            feature3 = 0
        if "putMarker" in trajectory or "pickMarker" in trajectory:
            feature4 = 1
        else:
            feature4 = 0
        feature5 = max(len(task_grid.get('pregrid_markers')), len(task_grid.get("postgrid_markers"))) \
            - abs(len(task_grid.get("pregrid_markers")) - len(task_grid.get("postgrid_markers")))

        feature6 = len(task_grid.get("walls"))

        return feature0, feature1, feature2, feature3, feature4, feature5, feature6

    def step(self, action):

        assert self.action_space.contains(action)
        self.num_steps += 1
        self.task_id = self.cur_id
        self.curr_task, finish, crash, no_marker = SymbolicKarel(self.curr_task).action_in_grid(action)
        reward = self.get_reward(finish, crash, no_marker)
        # Add horizon to avoid infinite rollout(used for stable baselines)
        if self.num_steps == self.H:
            self.Done = True
        return symb_2_bitmap(self.curr_task), reward, self.Done, {}

    def check_space(self):
        x = self.observation_space.sample()
        assert self.observation_space.contains(x)

    def reset_task(self, cur_id):
        self.curr_task = copy.deepcopy(self.original_envs[cur_id])

    def reset(self):
        super(KarelGymWrapper, self).select_next_task()
        self.Done = False
        self.num_steps = 0
        # Reset selected task
        self.reset_task(self.cur_id)

        if self.metrics:
            # Record features
            if self.cur == "iid" or self.cur == "procurl-env" or self.cur == "procurl-val":
                feat0, feat1, feat2, feat3, feat4, feat5, feat6 = self.get_features(self.cur_id)
                self.ls_features0.append(feat0), self.ls_features1.append(feat1), self.ls_features2.append(feat2), \
                self.ls_features3.append(feat3), self.ls_features4.append(feat4), self.ls_features5.append(feat5), \
                self.ls_features6.append(feat6)

                if len(self.ls_features0) == 500:  # Take the average of 500 episodes
                    wandb.log({"context1": sum(self.ls_features0) / len(self.ls_features0),
                               "context2": sum(self.ls_features1) / len(self.ls_features1),
                               "context3": sum(self.ls_features2) / len(self.ls_features2),
                               "context4": sum(self.ls_features3) / len(self.ls_features3),
                               "context5": sum(self.ls_features4) / len(self.ls_features4),
                               "context6": sum(self.ls_features5) / len(self.ls_features5),
                               "context7": sum(self.ls_features6) / len(self.ls_features6)
                               })
                    self.ls_features0, self.ls_features1, self.ls_features2, self.ls_features3, self.ls_features4, \
                        self.ls_features5, self.ls_features6 = [], [], [], [], [], [], []

        # Return the bitmap representation of the new task to the agent
        return symb_2_bitmap(self.curr_task)

    # Reward function based on state and action
    def get_reward(self, finish, crash, no_marker):
        # Action finished is taken
        if finish:
            self.Done = True
            # Curr_grid is equal to post_grid(agent location/orientation and markers)
            if [self.curr_task["pregrid_agent_row"], self.curr_task["pregrid_agent_col"]] == \
                [self.curr_task["postgrid_agent_row"], self.curr_task["postgrid_agent_col"]] and \
                self.curr_task["pregrid_agent_dir"] == self.curr_task["postgrid_agent_dir"] and \
                    sorted(self.curr_task["pregrid_markers"]) == sorted(self.curr_task["postgrid_markers"]):   # Sorted because order o# f markers does not matter
                # Reward is 1
                return 1
            else:
                return 0
        # Agent took invalid move
        elif crash:
            self.Done = True
            return 0
        # Agent picked marker but there is no marker
        elif no_marker:
            self.Done = True
            return 0
        # In any other case r=0
        else:
            self.Done = False
            return 0

    def render(self, mode="human"):
        SymbolicKarel(self.curr_task).draw()


class ValGymWrapper(AbstractCurriculumEvalGymWrapper):
    """ Wrapper for the validation environment. Enumerate over all test tasks """
    def __init__(self):

        self.type_env = "binary"
        path = "envs/basic_karel/data/val/"  # test data
        # path = "../code/neural_karel/data/train/"  # train data
        self.Done = False
        super(ValGymWrapper, self).__init__(self.type_env, path)
        print("Number of test tasks", self.num_of_tasks)

        self.H = 100
        self.original_envs = copy.deepcopy(self.collection_envs)
        # Initialization to define input size
        self.cur_id = 0
        self.curr_task = copy.deepcopy(self.collection_envs[self.cur_id])

        self.input_size = 5 * self.curr_task["gridsz_num_rows"] * self.curr_task["gridsz_num_cols"] + 2 * 4
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 1, [self.input_size], np.int32)

    def step(self, action):

        assert self.action_space.contains(action)
        self.num_steps += 1
        self.curr_task, finish, crash, no_marker = SymbolicKarel(self.curr_task).action_in_grid(action)
        reward = self.get_reward(finish, crash, no_marker)
        # Add horizon to avoid infinite rollout(used for stable baselines_bck)
        if self.num_steps == self.H:
            self.Done = True
        return symb_2_bitmap(self.curr_task), reward, self.Done, {}

    def reset_task(self):
        self.curr_task = copy.deepcopy(self.original_envs[self.curr_eval_id])

    def reset(self):
        self.Done = False
        self.num_steps = 0
        # Reset selected task
        self.reset_task()
        # Reset with next task
        self.curr_task = copy.deepcopy(self.original_envs[self.pick_next_id()])
        # Return the bitmap representation of the new task to the agent
        return symb_2_bitmap(self.curr_task)

        # Reward function based on state and action

    def get_reward(self, finish, crash, no_marker):
        # Action finished is taken
        if finish:
            self.Done = True
            # Curr_grid is equal to post_grid(agent location/orientation and markers)
            if [self.curr_task["pregrid_agent_row"], self.curr_task["pregrid_agent_col"]] == \
                    [self.curr_task["postgrid_agent_row"], self.curr_task["postgrid_agent_col"]] and \
                    self.curr_task["pregrid_agent_dir"] == self.curr_task["postgrid_agent_dir"] and \
                    sorted(self.curr_task["pregrid_markers"]) == sorted(
                    self.curr_task["postgrid_markers"]):  # Sorted because order o# f markers does not matter
                # Reward is 1
                return 1
            else:
                return 0
        # Agent took invalid move
        elif crash:
            self.Done = True
            return 0
        # Agent picked marker but there is no marker
        elif no_marker:
            self.Done = True
            return 0
        # In any other case r=0
        else:
            self.Done = False
            return 0

    def load_envs(self, path, env_type):
        # Load all tasks for test
        for t_name in sorted(glob(f'{path}/task/*.json')):
            self.collection_envs.append(self.read_csv_tasks(t_name))

    @staticmethod
    def read_csv_tasks(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data
