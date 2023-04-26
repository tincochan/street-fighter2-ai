from abstract_classes.AbstractTeacher import AbstractTeacherCallback


class BallCatchingTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the BallCatching environment.
    Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, cur, eps, n_steps, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon, metrics=True, verbose=1):
        self.F = 20480
        self.type_env = env.type_env
        print(self.type_env)
        norm_strategy = "static"
        super(BallCatchingTeacher, self).__init__(env, cur, eps, n_steps, self.F, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon,
                                                  norm_strategy, metrics, verbose)

    @staticmethod
    def get_context_obs(curr_task):
        s0 = curr_task.get_obs()
        return s0

    @staticmethod
    def take_step(curr_task, action):
        s1, r, done, info = curr_task.real_step(action)
        return s1, r, done, info

    @staticmethod
    def reset_task(curr_task, cur_id):
        curr_task.reset_model()
        return

    @staticmethod
    def check_horizon(step, done):
        if step == 100:
            done = True
        return done

    # Simple normalization
    @staticmethod
    def get_vmax():
        return 60
