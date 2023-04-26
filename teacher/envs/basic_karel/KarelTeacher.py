import copy
from abstract_classes.AbstractTeacher import AbstractTeacherCallback
from envs.basic_karel.utils import symb_2_bitmap


class KarelTeacher(AbstractTeacherCallback):
    """ Concrete teacher callback class for the BasicKarel environment.
        Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, cur, eps, n_steps, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon, metrics=True,
                 verbose=1):
        self.F = 102400
        self.type_env = env.type_env
        norm_strategy = None
        super(KarelTeacher, self).__init__(env, cur, eps, n_steps, self.F, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon,
                                           norm_strategy, metrics, verbose)

    def get_context_obs(self, curr_task):
        self.env.curr_task = copy.deepcopy(curr_task)
        s0 = symb_2_bitmap(self.env.curr_task)
        return s0

    def take_step(self, curr_task, action):
        s1, r, done, info = self.env.step(action)
        return s1, r, done, info

    def reset_task(self, curr_task, cur_id):
        self.env.reset_task(cur_id=cur_id)
        return

    @staticmethod
    def get_vmax():
        pass

    @staticmethod
    def check_horizon(step, done):
        if step == 100:
            done = True
        return done

