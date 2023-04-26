from abstract_classes.AbstractTeacher import AbstractTeacherCallback


class PointMassTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the PointMass environment.
        Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, type_env, cur, eps, n_steps, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon,
                 metrics=True, verbose=1):
        self.F = 5120
        self.type_env = type_env  # Point-mass can be binary and non-binary
        if self.type_env == "non-binary":
            norm_strategy = "adaptive"
        else:
            norm_strategy = None
        super(PointMassTeacher, self).__init__(env, cur, eps, n_steps, self.F, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon,
                                               norm_strategy, metrics, verbose)


    @staticmethod
    def get_context_obs(curr_task):
        s0 = curr_task.context_state
        return s0

    @staticmethod
    def take_step(curr_task, action):
        s1, r, done, info = curr_task.step(action)
        return s1, r, done, info

    @staticmethod
    def reset_task(curr_task, index):
        curr_task.reset()
        return

    def get_vmax(self):
       pass

    @staticmethod
    def check_horizon(step, done):
        if step == 100:
            done = True
        return done
