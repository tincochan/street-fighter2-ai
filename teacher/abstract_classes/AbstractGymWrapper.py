from abc import ABC, abstractmethod
from gym import Env
import numpy as np
import random
import copy


class AbstractCurriculumGymWrapper(Env, ABC):
    """ Abstract class for a curriculum OpenAI Gym wrapper.
        Provides the template on how the concrete Gym environments
        should be created in order to run curriculum experiments.
        Any environment can be derived from this abstract class."""
    def __init__(self, cur, env_type, beta, path, beta_plr, rho_plr):

        self.cur = cur
        self.task_id, self.curr_task, self.cur_id = None, None, None
        self.collection_envs, self.contexts = [], []
        self.ep = 0  # To log episodes

        self.load_envs(path, env_type)
        self.num_steps = 0
        self.num_of_tasks = len(self.collection_envs)

        # CurriculumMultiple properties
        self.post = np.ones(self.num_of_tasks)  # POSTeacher=1
        self.posl = np.zeros(self.num_of_tasks)  # PoSLearner=0 and will be updated by the teacher callback
        self.posL_estim = np.zeros(self.num_of_tasks)  # PoSL estimation, used for updating PosL using training rollouts

        self.beta = beta
        self.iid_trigger = True

        # Used for SPACE. It is used and updated by the teacher
        self.cur_set = [self.collection_envs[0]]
        self.cur_set_ids = [0]
        self.indices = [0]
        self.instance_set_size = 1 / len(self.collection_envs)
        self.space_id = 0
        self.dt = np.zeros(self.num_of_tasks)  # Difference in Values, i.e V_{t} - V_{t-1}

        # SPDL parameters
        self.pv = np.divide(np.ones(self.num_of_tasks), self.num_of_tasks)  # Initial uniform task distribution

        # PLR parameters
        self.binomial_p = 0.0
        self.seen = []
        self.unseen = list(range(self.num_of_tasks))
        self.p_scores = np.zeros(self.num_of_tasks)
        self.global_timestamps = np.zeros(self.num_of_tasks)
        self.rho_plr = rho_plr # Hyperparameter for P_replay
        #self.P_s = np.zeros(self.num_of_tasks)
        #self.P_c = np.zeros(self.num_of_tasks)
        self.beta_plr = beta_plr # Hyperparameter for P_replay


    @abstractmethod
    def load_envs(self, path, env_type):
        pass

    @staticmethod
    @abstractmethod
    def read_csv_tasks(path):
        pass

    @abstractmethod
    def step(self, action):
        """
        Define an example step function of the environment
        """
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        obs, reward, done, info = self.curr_task.step(action=action)
        return obs, reward, done, info

    @abstractmethod
    def reset(self):
        """
        A reset function example of the environment
        """
        self.select_next_task()
        return self.curr_task.reset()  # Returns reset observation of the newly selected task

    def select_next_task(self):
        """
        Selects next tasks based on the specified curriculum strategy
        """
        if self.cur == "procurl-env":
            self.cur_id = self.pick_curr_id()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "procurl-val":
            self.cur_id = self.pick_curr_id()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "iid":
            self.cur_id = self.pick_random_id()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "space":
            self.curr_task = self.pick_space()
        elif self.cur == "space-alt":
            self.cur_id = self.pick_space_alt()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "procurl-iid":
            self.cur_id = self.cur_iid_combo()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "hard":
            self.cur_id = self.pick_hard_id()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "easy":
            self.cur_id = self.pick_easy_id()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "spdl":
            self.cur_id = self.pick_spdl()
            self.curr_task = self.collection_envs[self.cur_id]
        elif self.cur == "plr":
            self.cur_id = self.pick_plr()
            self.curr_task = self.collection_envs[self.cur_id]
        else:
            raise ValueError("Invalid curriculum strategy")
        self.ep += 1  # A new episode

        return

    def render(self, mode="human"):
        pass

    # Ways of selecting next tasks. Same across different environments


    def pick_random_id(self):
        return random.randint(0, self.num_of_tasks - 1)

    def pick_curr_id(self):
        """
        Select next task based on a soft selection from equation PoSL*(PosT-PoSL)
        """
        return random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=np.exp(self.beta * self.posl * (self.post - self.posl)), k=1)[0]

    def pick_easy_id(self):
        return random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=np.exp(self.beta * self.posl), k=1)[0]

    def pick_hard_id(self):
        return random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=np.exp(self.beta * (1 - self.posl)), k=1)[0]

    def pick_space_alt(self):
        """
        Select next task based on V_{t} - V_{t-1}
        """
        return random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=np.exp(self.beta * self.dt), k=1)[0]

    def pick_space(self):
        # Round-robin inside the cur set of space
        space_task = copy.deepcopy(self.cur_set[self.space_id])
        self.space_id = (self.space_id + 1) % len(self.cur_set)
        self.cur_id = self.space_id
        return space_task

    def cur_iid_combo(self):
        """
        Alternate selection between iid and procurl-env
        """
        if self.iid_trigger:
            return random.randint(0, self.num_of_tasks - 1)
        else:
            return random.choices(population=np.arange(0, self.num_of_tasks),
                                  weights=np.exp(self.beta * self.posL_estim * (self.post - self.posL_estim)), k=1)[0]

    def pick_spdl(self):
        task = random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=self.pv, k=1)[0]
        return task

    # Space functions
    def get_instance_set(self):
        return self.indices, self.cur_set

    def increase_set_size(self, kappa):
        self.instance_set_size += kappa / len(self.collection_envs)
        return

    def set_instance_set(self, indices):
        size = int(np.ceil(len(self.collection_envs) * self.instance_set_size))
        if size <= 0:
            size = 1
        self.cur_set = np.array(self.collection_envs)[indices[:size]]
        self.cur_set_ids = [indices[:size]]
        self.indices = indices

    def get_context(self):
        return self.contexts[self.cur_id]

    def pick_plr(self):

        # Anneal the bernoulli probability
        self.binomial_p = len(self.seen) / self.num_of_tasks

        # Sample replay decision from bernoulli distribution
        d = np.random.binomial(1, self.binomial_p)

        # Sample unseen task
        if d == 0 and self.unseen != []:

            # Sample uniform from list self.unseen
            task_id = random.choice(self.unseen)
            # Update seen and unseen
            self.seen.append(task_id)
            self.unseen.remove(task_id)

            #print(self.seen, "seen")
            #print(self.unseen, "unseen")
        else:
            # Update the staleness of the tasks
            self.update_staleness()
            self.update_p_scores()
            p_replay = (1-self.rho_plr) * self.P_s + self.rho_plr * self.P_c
            # Sample from seen tasks with prob Preplay
            task_id = random.choices(population=self.seen, weights=p_replay, k=1)[0]

        # Update the global timestamps for the task with episode number
        self.global_timestamps[task_id] = self.ep + 1
        return task_id

    def update_staleness(self):

        # Update the staleness of the seen tasks in self.seen
        seen_timestamps = self.global_timestamps[self.seen]
        self.P_c = (self.ep + 1 - seen_timestamps)/np.sum(self.ep + 1 - seen_timestamps, axis=0)
        #self.P_c = (self.ep + 1 - self.global_timestamps)/np.sum(self.ep + 1 - self.global_timestamps, axis=0)
        return

    def update_p_scores(self):

        # Get the rank of each entry in self.scores
        seen_scores = self.p_scores[self.seen]
        temp = np.flip(np.argsort(seen_scores))
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(seen_scores)) + 1

        # Compute the weights and normalize
        weights = 1 / rank ** (1. / self.beta_plr)
        z = np.sum(weights)
        weights /= z
        self.P_s = weights
        return


class AbstractCurriculumEvalGymWrapper(Env, ABC):
    """Abstract Evaluation Wrapper. It is similar to AbstractCurriculumGymWrapper and
    tasks are selected sequentially for evaluation"""
    def __init__(self, env_type, path):

        self.collection_envs, self.contexts = [], []
        self.ep = -1  # To log episodes
        self.load_envs(path, env_type)
        self.num_steps = 0
        self.num_of_tasks = len(self.collection_envs)
        self.curr_eval_id = -1
        self.curr_task = None

    def pick_next_id(self):
        self.curr_eval_id = (self.curr_eval_id + 1) % self.num_of_tasks
        return self.curr_eval_id

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def load_envs(self, path, env_type):
        pass


