import numpy as np
from cpso_particle import COMB_Particle

# NOTE: I need to map out the progression of the algorithm and see if
# dynamic programming would be helpful here. If eval_fitness is called
# on the same thing more than once, it is worth the memory usage to
# store fitness values and only calculate them once.

class COMB_Swarm:

    """COMB-Particle Swarm Optimization (COMB-PSO) Swarm.

    COMB_Swarm is an implementation of a swarm object in
    Combined-Particle Swarm Optimization. It contains a group of
    COMB_Particle objects and maintains group attributes.

    Attributes
    ----------
    swarm : group of COMB_Particles
    
    gbest : pass

    gbest_counter : pass

    abest : pass

    t : pass

    c1 : pass

    c2 : pass

    c3 : pass

    ndim : pass

    x_bounds : pass

    v_bounds : pass

    w_bounds : pass

    t_bounds : pass


    Functions
    ---------
    __init__ : Initializes a COMB_Swarm object and assigns all attributes.

    init_classifier : Initializes the classifier used in evaluate_fitness.

    eval_fitness : pass

    # NOTE: This might be a bad idea. It would be modular and clean, but have
    # too much information hiding and make the script using the COMB_Swarm
    # unreadable. Pros: software is usable for different algorithms,
    # Cons: software is harder to read through and understand in one easy
    # shot.
    advance_time : Move the swarm forward one time step

    shuffle_gbest : pass
    """
    
    def __init__(self, c1, c2, c3, ndim,
                 x_bounds, v_bounds, w_bounds, t_bounds):
        """Initialize the COMB_Swarm object.

        Parameters
        ----------
        pass

        Returns
        -------
        pass

        Raises
        ------
        pass
        """
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ndim = ndim
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        self.t_bounds = t_bounds
        self.gbest_counter = 0

        self.swarm = []
        gbest_updated = False
        for i in range(ndim):
            # Fix the class instantiation if I add t_bounds. Also,
            # I might remove w as an argument for instantiation because
            # it might just start at w_bounds[1].
            self.swarm.append(COMB_Particle(w, c1, c2, c3, ndim,
                                            x_bounds, v_bounds, w_bounds))
            if i == 0:
                self.gbest = self.swarm[i].x.copy()
            elif eval_fitness(self.swarm[i].x) > eval_fitness(self.swarm.gbest):
                self.gbest = self.swarm[i].x.copy()
        self.abest = self.gbest.copy()

    def shuffle_gbest():
        # method should only be used when gbest has been updated 3 times and
        # has stagnated.
        assert(self.gbest_counter = 3)
        self.gbest = np.random.uniform(low=self.x_bounds[0],
                                       high=self.x_bounds[1],
                                       size=self.ndim)
        if eval_fitness(self.gbest) > eval_fitness(self.abest):
            self.abest = self.gbest.copy()
        self.gbest_counter = 0

    def eval_fitness(self, classifier, classifier_args):
        # NOTE: Upon further reflection, I think this does not belong
        # in this class. evaluate_fitness should be a method in the swarm
        # level that takes a COMB_Particle.x as an argument. That fitness
        # function may then itself outsource the classification accuracy
        # test to another function before incorporating it into the
        # Fitness Function.
        """Evaluate the fitness of the current position vector.

        Evaluates the fitness of the current position vector or feature
        subset by using the current position vector to train a classifier.

        Fitness Function:

        f(X, y, b, alpha) = alpha * Pb + (1-alpha) * (|X|-|b|)/|X|

        where X is the set of all features,
              y is the set of class label values,
              b is the subset of features selected
              alpha is a weight factor that denotes importance to size of the
                    subset and accuracy of the classifier
              Pb is the classification performance given only b (decoded by y
                 and the position vector)
              |X| is the number of features (size of position vector)
              |b| is the number of features in b

        Parameters
        ----------
        classifier : object, classifier to be trained on the current position.

        classifier_args : list, additional arguments for classifier

        Returns
        -------
        f : the value of the fitness function with the given parameters

        Raises
        ------
        None
        """
        # Decode self.x to a feature subset
        # 10-fold cross validation on data using classifier and feature subset
        # Return classification performance
        # Calculate fitness using classification performance
        pass
    



