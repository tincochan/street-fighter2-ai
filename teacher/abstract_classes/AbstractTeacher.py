from abc import ABC, abstractmethod
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
import wandb
from scipy.optimize import minimize, NonlinearConstraint, Bounds


class AbstractTeacherCallback(BaseCallback, ABC):
    """ Abstract class for the teacher callback inherits from stable-baselines3 BaseCallback.
        Template for defining a concrete teacher callback that can be used with sb3
        while training on any gym environment with a curriculum wrapper"""
    def __init__(self, env, cur, eps, n_steps, F, spdl_pthresh, eta, space_freq, kappa, Na, alpha, epsilon,
                 norm_strategy, metrics=True, verbose=1):
        super(AbstractTeacherCallback, self).__init__(verbose)

        self.env = env  # A gym env that is defined based on our AbstractGymWrapper class
        self.F = F  # Hyperparameter frequency F
        self.instances = self.env.collection_envs
        self.cur = cur
        self.eps = eps  # Noise added to the critic estimate
        self.n_steps = n_steps  # nb_steps for model update
        self.pos_steps = 0  # To measure the environment steps of the extra rollouts
        self.metrics = metrics
        self.type_env = self.env.type_env  # Type of env based on success and reward. Can be binary, non-binary, info based success
        self.nb_eval_rolls = 20  # Hyperparameter nb_rollouts to evaluate learner
        self.step_budget = np.inf  # Can be used to restrict budget of steps for PoS evaluation
        self.norm_strategy = norm_strategy

        # For procurl-iid
        self.pos_est = np.zeros(len(self.instances))
        self.counts = np.zeros(len(self.instances))
        self.alter_cur = 0

        # Space parameters
        self.size = 1
        self.kappa = kappa
        self.old_mean = -np.inf
        self.eta = eta
        self.last_evals = np.zeros(len(self.instances))
        self.space_freq = space_freq

        # Spdl parameters
        self.Na = Na
        self.alpha = alpha
        self.epsilon = epsilon
        self.spdl_pthresh = spdl_pthresh  # performance threshold for SPDL

        # For PLR
        self.rollout_tasks = []

    @staticmethod
    @abstractmethod
    def get_context_obs(curr_task):
        pass

    @staticmethod
    @abstractmethod
    def take_step(curr_task, action):
        pass

    @staticmethod
    @abstractmethod
    def reset_task(curr_task, cur_id):
        pass

    @abstractmethod
    def get_vmax(self):
        pass

    @staticmethod
    @abstractmethod
    def check_horizon(cur_step, done):
        pass

    @staticmethod
    def define_success(rewards, infos, success_count, s_type):
        """
        Define rollout success based on the type of environment.
        :param rewards: list of rewards for each step of the previous episode
        :param infos: list of infos for each step
        :param success_count: previous success count
        :param s_type: type of success
        :return: updated success count
        """
        if s_type == "info":
            lst_info = [info["success"] for info in infos]
            if any(lst_info):
                success_count += 1
            return success_count
        elif s_type == "binary_reward":
            if rewards[-1] == 1:
                success_count += 1
            return success_count

    def p_success(self, num_times, p_model, env_type):
        """
        Compute probability of success for all tasks by doing extra rollouts
        :param num_times: number of rollouts to do for each task
        :param p_model: current policy
        :param env_type: type of environment, binary, non-binary etc.
        :return: numpy array posl, i.e., probability of success for each task
        """
        posl = np.zeros(self.env.num_of_tasks)
        for index, curr_task in tqdm(enumerate(self.instances)):
            success_count = 0
            for ep in range(num_times):
                step = 0
                done = False
                rewards = []
                infos = []
                s0 = self.get_context_obs(curr_task)
                while not done:
                    action, states = p_model.predict(s0, deterministic=False)
                    step += 1
                    s1, r, done, info = self.take_step(curr_task, action)
                    rewards.append(r)
                    infos.append(info)
                    s0 = s1
                    self.pos_steps += 1
                    done = self.check_horizon(step, done)
                success_count = self.define_success(rewards, infos, success_count, env_type)
                self.reset_task(curr_task, index)
            posl[index] = success_count / num_times
        return posl

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
                    s1, r, done, info = self.take_step(curr_task, action)
                    rewards.append(r)
                    s0 = s1
                    self.pos_steps += 1
                    done = self.check_horizon(step, done)
                total_rew = np.sum(rewards)
                total_rewards.append(total_rew)
                self.reset_task(curr_task, index)
            posl[index] = (np.sum(total_rewards) / num_times)
        # Normalize it to [0,1] PoS by dividing with a large value and then clip it to Vmin, Vmax range.
        # Any normalization technique can be used to make the value prediction range [0,1]
        if self.norm_strategy == "static":
            vm = self.get_vmax()
            posl = posl / vm
        elif self.norm_strategy == "adaptive":
            posl = (posl - np.min(posl)) / (np.max(posl) - np.min(posl))  # Alternative normalization
        posl = np.clip(posl, a_min=0, a_max=1)
        return posl

    def value_forward_pass(self):
        """
        Do forward pass of the value network for all tasks and return value estimates
        :return: numpy array evals, the value prediction for each task
        """
        evals = []
        for index, curr_task in tqdm(enumerate(self.instances)):
            s0 = self.get_context_obs(curr_task)
            s_tensor, _ = self.model.policy.obs_to_tensor(s0)
            max_future_q = self.model.policy.predict_values(s_tensor)
            evals.append(max_future_q.cpu().detach().numpy()[0][0])
        evals = np.array(evals)
        return evals

    def q_evals(self):
        """
        Get task value improvement and order them. Used for space selection
        """
        evals = []
        mean_q = []
        for index, curr_task in tqdm(enumerate(self.instances)):
            s0 = self.get_context_obs(curr_task)
            s_tensor, _ = self.model.policy.obs_to_tensor(s0)
            max_future_q = self.model.policy.predict_values(s_tensor)
            evals.append(max_future_q.cpu().detach().numpy()[0][0])
            if curr_task in self.env.cur_set:
                mean_q.append(max_future_q.cpu().detach().numpy()[0][0])
        # Order instances improvements
        evals = np.array(evals)
        improvement = evals - self.last_evals
        self.last_evals = evals
        # Reverse argsort to give the largest values first
        return np.argsort(improvement)[::-1], np.mean(mean_q), evals

    # Used for space variant
    def new_q_evals(self):
        evals = self.value_forward_pass()

        if self.norm_strategy == "static":
            # Normalize values
            vm = self.get_vmax()
            evals = evals/vm
        # Compute improvement
        improvement = evals - self.last_evals

        if self.norm_strategy == "adaptive":
            # Scale improvement values
            improvement = (improvement - np.min(improvement)) / (np.max(improvement) - np.min(improvement))

        self.last_evals = evals
        # Reverse argsort to give the largest values first
        return improvement

    def value_critic(self):
        """
        Compute values for each task. Normalize values
        and clip them to range [Vmin, Vmax]
        """
        evals = self.value_forward_pass()
        if self.norm_strategy == "adaptive":
            evals = (evals - np.min(evals)) / (np.max(evals) - np.min(evals))

        elif self.norm_strategy == "static":
            vm = self.get_vmax()
            evals = evals / vm

        # Noise is added to the critic estimation. Used for ablation
        if self.eps > 0:
            noise = np.random.uniform(-self.eps, self.eps, self.env.num_of_tasks)  # Add uniform noise to the evaluations
            evals = np.add(evals, noise)

        norm_val = np.clip(evals, a_min=0, a_max=1)  # Clipping between Vmin and Vmax
        return norm_val

    # SPDL Update
    @staticmethod
    def logsumexp(x, axis=None):
        x_max = np.max(x, axis=axis)
        return np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max

    @staticmethod
    def interp_ll(old_ll, target_ll, values, x):
        eta, alpha = x
        interp_ll = (target_ll + eta * values) * alpha + old_ll * (1 - alpha)
        interp_ll -= AbstractTeacherCallback.logsumexp(interp_ll)
        return interp_ll

    @staticmethod
    def expected_value(old_ll, target_ll, values, x):
        return np.sum(np.exp(AbstractTeacherCallback.interp_ll(old_ll, target_ll, values, x)) * values)

    @staticmethod
    def kl_divergence(old_ll, target_ll, ref_ll, values, x):
        interp_ll = AbstractTeacherCallback.interp_ll(old_ll, target_ll, values, x)
        return np.sum(np.exp(interp_ll) * (interp_ll - ref_ll))

    def spdl_update(self):
        """
        SPDL optimization step
        """
        evals = self.value_forward_pass()
        min_eval = np.min(evals)
        max_eval = np.max(evals)
        eval_range = max_eval - min_eval
        # We normalize the values to be between 0 and 1
        if self.type_env == "non-binary" or self.type_env == "info-based":
            print("Min: %.3e, Max: %.3e, Range: %.3e" % (min_eval, max_eval, eval_range))
            evals = (evals - min_eval) / eval_range

        # Compute ci
        if hasattr(self.env, "log_pv"):
            old_ll = self.env.log_pv
        else:
            old_ll = np.log(self.env.pv)

        # Define the spdl objective
        from functools import partial

        if self.type_env == "non-binary" or self.type_env == "info-based":
            delta = (self.spdl_pthresh - min_eval) / eval_range
        else:
            delta = self.spdl_pthresh
        print("Iteration Delta: %.3e" % delta)
        # Want a uniform distribution over the contexts
        target_ll = -np.log(old_ll.shape[0]) * np.ones(old_ll.shape[0])
        perf_con = NonlinearConstraint(partial(self.expected_value, old_ll, target_ll, evals), delta, np.inf)
        kl_con = NonlinearConstraint(partial(self.kl_divergence, old_ll, target_ll, old_ll, evals), -np.inf,
                                     self.epsilon)

        # If we are below the performance threshold, we optimize the performance in a first run
        avg_perf = np.sum(np.exp(old_ll) * evals)
        if avg_perf <= delta:
            neg_objective = partial(self.expected_value, old_ll, target_ll, evals)
            res = minimize(lambda x: -neg_objective(x), np.array([1., 1.]), method='trust-constr', jac="3-point",
                           constraints=[kl_con], options={'verbose': 1, "gtol": 1e-4, "xtol": 1e-6},
                           bounds=Bounds(1e-3 * np.ones(2), 1e4 * np.ones(2)))

            intermediate_ll = self.interp_ll(old_ll, target_ll, evals, res.x)
            x0 = res.x

            avg_perf = np.sum(np.exp(intermediate_ll) * evals)
            if res.success:
                # In this case we either set the optimized performance distribution as the new sampling distributoin
                if avg_perf < delta:
                    print("Optimized performance as performance constraint not met: %.3e vs %.3e" % (avg_perf, delta))
                    self.env.log_pv = intermediate_ll
                    self.env.pv = np.exp(intermediate_ll)
                    return
            else:
                print("Warning! Optimization not successful")
                return
            # Only if the optimization was successful and the optimized result fulfills the performance constraint
            # we continue with the optimization.
        else:
            intermediate_ll = old_ll
            x0 = np.array([1., 1.])

        # If we start above the performance threshold, we minimize the KL
        if avg_perf > delta:
            constraints = [perf_con, kl_con]
            objective = partial(self.kl_divergence, old_ll, target_ll, target_ll, evals)

            res = minimize(objective, x0, method='trust-constr', jac="3-point", constraints=constraints,
                           options={'verbose': 1, "gtol": 1e-8, "xtol": 1e-8}, #options={'verbose': 1, "gtol": 1e-4, "xtol": 1e-6},
                           bounds=Bounds(1e-4 * np.ones(2), 1e4 * np.ones(2)))

            if res.success:
                print("New Target KL-Divergence: %.3e" % res.fun)
                self.env.log_pv = self.interp_ll(old_ll, target_ll, evals, res.x)
            else:
                print("Warning! Optimization not successful!")
                self.env.log_pv = intermediate_ll
        else:
            raise RuntimeError("Should not happen!")

        self.env.pv = np.exp(self.env.log_pv)
        print("Expected Performance: %.3e" % self.expected_value(old_ll, target_ll, evals, res.x))

    def posl_eval(self):
        """
        Custom interface to evaluate the PosL based on different types of environments
        """
        if self.type_env == "binary":
            self.env.posl = self.p_success(num_times=self.nb_eval_rolls, p_model=self.model, env_type="binary_reward")
        elif self.type_env == "info-based":
            self.env.posl = self.p_success(num_times=self.nb_eval_rolls, p_model=self.model, env_type="info")
        elif self.type_env == "non-binary":
            self.env.posl = self.non_binary_success(num_times=self.nb_eval_rolls, p_model=self.model)
        return self.env.posl


    def _on_training_start(self) -> None:
        print(f"Instantiate {self.cur} Teacher")
        # Domain knowledge assumed pos*=1
        if self.cur == "procurl-env" or self.cur == "hard" or self.cur == "easy":
            if self.step_budget == 1000000 or self.step_budget == 3000000:
                pass
            else:
                self.env.posl = self.posl_eval()
        # PosL is estimated by a value critic
        elif self.cur == "procurl-val":
            self.env.posl = self.value_critic()

    def _on_rollout_start(self) -> None:
        self.rollout_tasks = []

    def _on_rollout_end(self) -> None:

        if self.cur == "plr":
            # These are the indices of the start of each task in the rollout
            task_starts = np.where(self.locals['rollout_buffer'].episode_starts == 1)[0]

            # if task_tasks does not contain 0 we add it
            if task_starts[0] != 0:
                task_starts = np.insert(task_starts, 0, 0)

            # If rollout_tasks has more elements than task_starts we remove the last task in rollout_tasks
            if len(self.rollout_tasks) > len(task_starts):
                self.rollout_tasks = self.rollout_tasks[:-1]

            # Compute the mean advantage of the rollout for each task in the rollout
            for i, task in enumerate(self.rollout_tasks):
                if i < len(task_starts) - 1:
                    # Update the environment p_scores if it is not nan
                    temp = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:task_starts[i+1] -1])
                    if not np.isnan(temp):
                        self.env.p_scores[task] = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:task_starts[i+1] -1])
                    else:
                        # Keep the old p_score if the new one is nan
                        #print("Warning: nan value in p_scores. Keeping old p_score.")
                        self.env.p_scores[task] = self.env.p_scores[task]
                else:
                    # Update the environment p_scores if it is not nan
                    temp = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:])
                    if not np.isnan(temp):
                        self.env.p_scores[task] = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:])
                    else:
                        # Keep the old p_score if the new one is nan
                        #print("Warning: nan value in p_scores. Keeping old p_score.")
                        self.env.p_scores[task] = self.env.p_scores[task]

            # Check if the p_scores contain nan values and raise an error if so
            if np.isnan(self.env.p_scores).any():
                raise ValueError("p_scores contain nan values")

    def _on_step(self) -> bool:

        # Log metrics(total number of steps)
        if self.metrics:
            if self.n_calls % self.n_steps == 0:
                wandb.log({"env_n_calls": self.n_calls, "global_env_steps": self.n_calls + self.pos_steps})

        # If new task is selected or new rollout starts
        if self.env.num_steps == 0 or self.locals['n_steps'] == 0:
            self.rollout_tasks.append(self.env.cur_id)

        # Evaluate Pos with external rollouts with a quering frequency F for procurl-env
        if self.n_calls % self.F == 0:
            if self.cur == "procurl-env" or self.cur == "hard" or self.cur == "easy":
                if self.pos_steps < self.step_budget:  # Condition for budget constraint
                    self.env.posl = self.posl_eval()
                else:
                    self.env.posl = np.ones(self.env.num_of_tasks)  # No evaluation, becomes random

        if self.n_calls % self.n_steps == 0:

            # Value critic evaluation
            if self.cur == "procurl-val":
                # Estimated posl from value critic
                self.env.posl = self.value_critic()

            # Spdl optimization step
            elif self.cur == "spdl":
                self.spdl_update()

            # Space variant algorithm
            elif self.cur == "space-alt":
                delta_q = self.new_q_evals()
                self.env.dt = delta_q

            # Original space
            elif self.cur == "space":
                indices, mean_q, _ = self.q_evals()
                delta_q = np.abs(np.abs(mean_q) - np.abs(self.old_mean))
                # Last condition is added because for BasicKarel sorting all the tasks is very slow
                if delta_q < np.abs(self.old_mean) * self.eta and len(self.env.cur_set) < len(self.instances) \
                        and len(self.env.cur_set) < 1000:
                    self.env.increase_set_size(kappa=self.kappa)
                self.old_mean = mean_q
                self.env.set_instance_set(indices)
        return True
