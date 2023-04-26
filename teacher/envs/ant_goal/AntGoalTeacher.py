from abstract_classes.AbstractTeacher import AbstractTeacherCallback
import numpy as np
from tqdm import tqdm


class AntGoalTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the AntGoal environment.
    Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, cur, eps, n_steps, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon, metrics=True, verbose=1):
        self.F = 81920  # Frequency for PoS evaluation
        self.type_env = env.type_env
        norm_strategy = "static"
        super(AntGoalTeacher, self).__init__(env, cur, eps, n_steps, self.F, spdl_pthresh, eta, space_freq, kappa, Na,
                                             alpha, epsilon, norm_strategy, metrics, verbose)

    def get_context_obs(self, curr_task):
        obs = self.env.single_env.reset_model()  # this reset returns concatenated obs
        self.env.cur_id = int(curr_task[0])
        # Sets new context
        obs = np.concatenate(([float(curr_task[1]), float(curr_task[2])], obs))
        return obs

    @staticmethod
    def take_step(curr_task, action):
        s1, r, done, info = curr_task.step(action)
        return s1, r, done, info

    @staticmethod
    def reset_task(curr_task, cur_id):
        return curr_task.single_env.reset_model()

    @staticmethod
    def check_horizon(step, done):
        if step >= 200:
            done = True
        return done

    # Simple normalization
    @staticmethod
    def get_vmax():
        return 300

    def non_binary_success(self, num_times, p_model):
        """
        Compute probability of success for all tasks by doing extra rollouts for a non-binary environment
        :param num_times: number of rollouts to do for each task
        :param p_model: current policy
        :return: numpy array posl, i.e., probability of success for each task
        """
        posl = np.zeros(self.env.num_of_tasks)
        for index, curr_task in tqdm(enumerate(self.instances)):
            total_rewards = []
            for ep in range(num_times):
                step = 0
                done = False
                rewards = []
                s0 = self.get_context_obs(curr_task)
                while not done:
                    action, states = p_model.predict(s0, deterministic=False)
                    step += 1
                    s1, r, done, info = self.take_step(self.env, action)
                    rewards.append(r)
                    s0 = s1
                    s0[0] = float(curr_task[1])
                    s0[1] = float(curr_task[2])
                    self.pos_steps += 1
                total_rew = np.sum(rewards)
                total_rewards.append(total_rew)
                self.env.num_steps = 0
            posl[index] = (np.sum(total_rewards) / num_times)
        vm = self.get_vmax()
        posl = posl / vm
        posl = np.clip(posl, a_min=0, a_max=1)
        return posl
