import abc
import logging
import time
import numpy as np

from typing import Iterable, List, Union, Tuple, Optional

from smac.configspace import get_one_exchange_neighbourhood, \
    convert_configurations_to_array, Configuration, ConfigurationSpace
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.utils.constants import MAXINT

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None
    ):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__
        )
        self.acquisition_function = acquisition_function
        self.config_space = config_space

        if rng is None:
            self.logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory : ~smac.runhistory.runhistory.RunHistory

        stats : ~smac.stats.stats.Stats

        num_points : int

        Returns
        -------
        iterable
            An iterable consisting of :class:`smac.configspace.Configuration`.
        """
        return [t[1] for t in self._maximize(runhistory, stats, num_points)]

    @abc.abstractmethod
    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        runhistory : ~smac.runhistory.runhistory.RunHistory

        stats : ~smac.stats.stats.Stats

        num_points : int

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]


class LocalSearch(AcquisitionFunctionMaximizer):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional

    epsilon: float
        In order to perform a local move one of the incumbent's neighbors
        needs at least an improvement higher than epsilon
    max_iterations: int
        Maximum number of iterations that the local search will perform

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            epsilon: float = 0.00001,
            max_iterations: Optional[int] = None
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            *args
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        start_point:  np.array(1, D)
            The point from where the local search starts
        *args:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        num_configurations_by_local_search = self._calculate_num_points(
            num_points, stats, runhistory
        )
        init_points = self._get_initial_points(
            num_configurations_by_local_search, runhistory)
        configs_acq = []

        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(
                start_point)

            configuration.origin = "Local Search"
            configs_acq.append((acq_val, configuration))

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq

    def _calculate_num_points(self, num_points, stats, runhistory):
        if stats._ema_n_configs_per_intensifiy > 0:
            num_configurations_by_local_search = (
                min(
                    num_points,
                    np.ceil(0.5 * stats._ema_n_configs_per_intensifiy) + 1
                )
            )
        else:
            num_configurations_by_local_search = num_points
        num_configurations_by_local_search = min(
            len(runhistory.data),
            num_configurations_by_local_search
        )
        return num_configurations_by_local_search

    def _get_initial_points(self, num_configurations_by_local_search,
                            runhistory):
        if runhistory.empty():
            init_points = self.config_space.sample_configuration(
                size=num_configurations_by_local_search)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = runhistory.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_configurations_by_local_search)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )
        return init_points

    def _one_iter(
            self,
            start_point: Configuration,
            *args
    ) -> Tuple[float, Configuration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], *args)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                self.logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.randint(MAXINT))

            for neighbor in all_neighbors:
                s_time = time.time()
                acq_val = self.acquisition_function([neighbor], *args)
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent + self.epsilon:
                    self.logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_iterations is not None and
                             local_search_steps == self.max_iterations):
                self.logger.debug("Local search took %d steps and looked at %d "
                                  "configurations. Computing the acquisition "
                                  "value for one configuration took %f seconds"
                                  " on average.",
                                  local_search_steps, neighbors_looked_at,
                                  np.mean(time_n))
                break

        return acq_val_incumbent, incumbent


class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            _sorted: bool = False,
            *args
    ) -> List[Tuple[float, Configuration]]:

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(
                size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function, config_space, rng
        )
        self.local_search = LocalSearch(
            acquisition_function, config_space, rng
        )

    def maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            *args
    ) -> Iterable[Configuration]:
        next_configs_by_local_search = self.local_search._maximize(
            runhistory, stats, 10,
        )

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            runhistory,
            stats,
            num_points - len(next_configs_by_local_search),
            _sorted=True,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
            next_configs_by_random_search_sorted
            + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])
        )
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space)
        return challengers

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class ChallengerList(object):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    challengers : list
        List of challengers (without interleaved random configurations)

    configuration_space : ConfigurationSpace
        ConfigurationSpace from which to sample new random configurations.
    """

    def __init__(self, challengers, configuration_space):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._next_is_random = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers) and not self._next_is_random:
            raise StopIteration
        elif self._next_is_random:
            self._next_is_random = False
            config = self.configuration_space.sample_configuration()
            config.origin = 'Random Search'
            return config
        else:
            self._next_is_random = True
            config = self.challengers[self._index]
            self._index += 1
            return config


class EASearch(AcquisitionFunctionMaximizer):
    """Implementation of EA-SMAC's search.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_generations: int
        Maximum number of generations in evolutionary process

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            max_generations: int,
            generation_size: int,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_generations = max_generations
        self.generation_size = generation_size
        self.elite_rate = 1.0  # TODO make it a parameter, use trails to select the rest
        self.result_configurations = 20
        self.crossover_probability = 0.5

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            *args
    ) -> List[Tuple[float, Configuration]]:
        """
        Performs EA search

        Parameters
        ----------
        num_points:  int:
            number of points to start with
        *args:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        acq_vals: np.array(1,1)
            The acquisition value of the incumbent
        survivals: np.array(1, D)
            The best configurations found among population

        """

        generation = self._get_initial_points(
            max(num_points, self.generation_size), runhistory)

        scored_survivals = []  # : [Tuple[float, Configuration]]

        # perform evolutionary computations
        for g in range(0, self.max_generations):
            scored_survivals = self._one_iter(generation)
            generation = list(map(lambda x: x[1], scored_survivals[:self.result_configurations]))

        # if self.result_configurations > self.generation_size:
        #     generation += self._get_initial_points(
        #         self.result_configurations - self.generation_size, runhistory)

        # shuffle for random tie-break
        self.rng.shuffle(scored_survivals)

        # sort according to acq value
        scored_survivals.sort(reverse=True, key=lambda x: x[0])

        # return required amount of configurations
        return scored_survivals[:self.result_configurations]

    def _get_initial_points(self, num_configurations, runhistory):
        if runhistory.empty():
            init_points = self.config_space.sample_configuration(
                size=num_configurations)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = runhistory.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_initial_configs = int(min(
                len(configs_previous_runs_sorted),
                num_configurations)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_initial_configs])
            )
        return init_points

    def _one_iter(
            self,
            start_generation: [Configuration],
            *args
    ) -> [Tuple[float, Configuration]]:

        # if we have too little to start with
        if self.generation_size > len(start_generation):
            start_generation = start_generation + \
                               self.config_space.sample_configuration(size=self.generation_size - len(start_generation))

        ## Breed phase:
        population = []

        # Add parents:
        population += [start_generation[i] for i in range(len(start_generation))]

        # Generate new individuals with crossover:
        for i in range(0, len(start_generation) - 1):
            for j in range(i + 1, len(start_generation)):
                if (self.rng.standard_normal() > self.crossover_probability):
                    population += self.crossover(start_generation[i], start_generation[j])

        ## Mutation phase (produces new individuals too):
        for incumbent in start_generation:
            neighbours = get_one_exchange_neighbourhood(incumbent, seed=self.rng.randint(MAXINT))
            population += neighbours

        # Make population contain only unique individuals:
        arrays = list(map(lambda x: x.get_array(), population))
        indices = np.unique(arrays, return_index=True, axis=0)
        population = [population[ind] for ind in indices[1][::-1]]

        ## Selection phase:
        survivals = []
        sorted = self._sort_configs_by_acq_value(population)

        elite_num = int(self.generation_size * self.elite_rate)
        for i in range(0, elite_num):
            individual = sorted[i]
            survivals.append(individual)

        return survivals

    def crossover(self, a: Configuration, b: Configuration) -> [Configuration]:
        hyperparameters_list = list(
            list(a.configuration_space._hyperparameters.keys())
        )

        c_point = int(self.rng.uniform(0, 1) * len(hyperparameters_list))

        a_array = a.get_array()
        b_array = b.get_array()

        for i in range(c_point, len(hyperparameters_list)):
            t = a_array[i]
            a_array[i] = b_array[i]
            b_array[i] = t

        # new configurations with new arrays
        new_a = Configuration(configuration_space=a.configuration_space, vector=a_array, origin="Crossover")
        new_b = Configuration(configuration_space=b.configuration_space, vector=b_array, origin="Crossover")

        return [new_a, new_b]
