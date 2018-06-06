#-----------------------------------------------------------------------------+
# 
# Title: COMB-PSO Swarm Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 6, 2018
# Date Modified: June 6, 2018, by Nathan Fox <nathanfox@miami.edu>
#
#-----------------------------------------------------------------------------+

# TODO: Docstrings, initialize_classifier, eval_fitness, check the math on
# the stagnation counter, proofread, and write test cases.
#
# Evolutionary Functionality: 
#   It should record the progression of gbest and abest, and possibly
#   also each particle's pbest. It also maybe should pickle the classifier?
#   It also might want to take freezes of the initial states, and record
#   all the positions and velocities for each particle over time and export.
#   The fitness values should almost certainly be recorded and exported
#   because they take so much time to compute.

import numpy as np
from cpso_particle import COMB_Particle

class COMB_Swarm:

    """COMB-Particle Swarm Optimization (COMB-PSO) Swarm.

    COMB_Swarm is an implementation of a swarm object in
    Combined-Particle Swarm Optimization. It contains a group of
    COMB_Particle objects and maintains group attributes.

    Attributes
    ----------
    swarm : list, size npart; holds the swarm of COMB_Particle objects.
    
    gbest : 1-Dimensional ndarray, size ndim; Holds the global best position
            found by any particle in the swarm. Subject to reshuffling based
            on the "shuffle and archive" behavior in the COMB-PSO algorithm.

    g_fitness : float; the fitness value returned by eval_fitness for the
                current gbest.

    gbest_counter : integer; stagnation counter for gbest. If it reaches 3,
                    shuffle_gbest is called.

    abest : 1-Dimensional ndarray, size ndim; Holds the archived best
            position found by any particle in the swarm, even if the global
            best has been shuffled.

    a_fitness : float; the fitness value returned by eval_fitness for the
                current abest.

    t : integer; current time.

    w : float; inertia coefficient.

    c1 : float; acceleration constant 1, for the cognitive component

    c2 : float; acceleration constant 2, for the social component

    c3 : float; acceleration constant 3, for the diversity component

    ndim : integer; number of dimensions in the search space; also the size
           of the position and velocity vectors.

    x_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the position vectors.

    v_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the velocity vectors.

    w_bounds : float tuple, size 2; Holds lower/upper bounds for the
               inertia coefficient.

    t_bounds : integer tuple, size 2; Holds beginning and end times for
               the algorithm.


    Functions
    ---------
    __init__ : Initializes a COMB_Swarm object and assigns all attributes.

    initialize_particles : Fills the swarm with initialized COMB_Particles. 

    initialize_classifier : Initializes the classifier used in eval_fitness.

    execute_search : Execute one full run of the COMB-PSO algorithm.

    update_inertia : Update the inertia coefficient, time-decreasing.

    shuffle_gbest : "Shuffle and Archive", randomizes gbest after stagnation.

    eval_fitness : Evaluates the fitness function for a position vector.
    """
    
    def __init__(self, npart, c1, c2, c3, ndim,
                 x_bounds, v_bounds, w_bounds, t_bounds):
        """Initialize the COMB_Swarm object.

        Initializes a COMB_Swarm object. The new COMB_Swarm then assigns its
        attributes and creates an empty Python list that will eventually hold
        the COMB_Particle objects. The actual initialization of the particles
        that make up the swarm was separated and moved to initialize_particles
        because it involves npart full fitness calls where npart is the number
        of particles. This is potentially computationally expensive and should
        not be called during first initialization, but manually afterward.

        Parameters
        ----------
        npart : integer; number of particles in the swarm.

        c1 : float; acceleration coefficient for the cognitive component.

        c2 : float; acceleration coefficient for the social component.

        c3 : float; acceleration coefficient for the diversity component.

        ndim : integer; number of dimensions or features in the search space.
               Also the length of the position and velocity vectors.

        x_bounds : tuple of floats, size 2; x_bounds[0] is the minimum value
                   that an element of COMB_Particle.x can be; x_bounds[1]
                   is the maximum value that an element of COMB_Particle.x
                   can be.

        v_bounds : tuple of floats, size 2; v_bounds[0] is the minimum value
                   that an element of COMB_Particle.v can be; v_bounds[1]
                   is the maximum value that an element of COMB_Particle.v
                   can be.

        w_bounds : tuple of floats, size 2; w_bounds[0] is the minimum value
                   that w can be; w_bounds[1] is the maximum value that w can
                   be.

        t_bounds : tuple of integers, size 2; t_bounds[0] is the starting time
                   and should essentially always be 0, t_bounds[1] is the
                   end time or maximum time allowed (stopping condition).

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.npart = npart
        self.w = w_bounds[1]
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
        self.g_fitness = 0.0
        self.a_fitness = 0.0

        # self.initialize_classifer()

    def initialize_particles(self):
        """Initialize the particles that comprise the swarm.

        Actually fills the empty list, swarm, with COMB_Particle objects.
        Separated initialization functionality from the __init__ function
        because of a high computational time cost. Wrapper scripts using
        the COMB_Swarm class should manually initialize the particles.

        NOTE: See the comment block below explaining why
              COMB_Particle.p_fitness attributes are initialized here
              instead of inside the COMB_Particle.__init__ method.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        for i in range(self.npart):
            self.swarm.append(COMB_Particle(self.c1, self.c2, self.c3,
                                            self.ndim, self.x_bounds,
                                            self.v_bounds, self.w_bounds))
            f = self.eval_fitness(self.swarm[i].x)
            # Initialized outside of COMB_Particle.__init__ because the
            # readibility in execute search of "for p in swarm" is worth it.
            # Storing each particle's pbest fitness within the particle
            # allows the main algorithm loop to be far more readable and
            # only means that there's a tricky line of initialization
            # code below, and that the stored fitness values are distributed
            # instead of in one place.
            self.swarm[i].p_fitness = f
            if i == 0:
                self.gbest = self.swarm[i].x.copy()
                self.g_fitness = self.eval_fitness(self.gbest)
            elif f > self.g_fitness:
                self.gbest = self.swarm[i].x.copy()
                self.g_fitness = f
        self.abest = self.gbest.copy()
        self.a_fitness = self.eval_fitness(self.abest)

    def initialize_classifer(self):
        pass

    def execute_search(self):
        """Execute a full run of the COMB-PSO Algorithm.

        Completes one full run of the COMB-PSO Algorithm using an internal
        classifier object and a swarm of COMB_Particle objects, returning
        a 1-Dimensional ndarray containing the best position found
        by the algorithm.

        NOTE: initialize_particles and initialize_classifer MUST be called
              before this method will run correctly.

        Parameters
        ----------
        None

        Returns
        -------
        abest : 1-Dimensional ndarray, size ndim; Holds the archived best
                position found by any particle in the swarm, even if the global
                best has been shuffled.

        Raises
        ------
        None
        """
        for i in range(1, t_bounds[1]):
            self.t = i
            self.update_inertia()
            for p in swarm:
                p.update_velocity(self.w, self.gbest, self.abest)
                p.update_position()
                p.update_binary_position()
                f = self.eval_fitness(p.x)
                if f > p.p_fitness:
                    p.pbest = p.x.copy()
                    p.p_fitness = f
                if f > self.g_fitness:
                    self.gbest_counter = 0 # NOTE: Maybe -1? Check the math
                    self.gbest = p.x.copy()
                    self.g_fitness = f
            if self.g_fitness > self.a_fitness:
                self.abest = self.gbest.copy()
                self.a_fitness = self.g_fitness
            self.gbest_counter += 1
            if self.gbest_counter >= 3: # NOTE: Make sure the math lines up
                self.shuffle_gbest()
        return self.abest

    def update_inertia(self):
        """Update inertia according to a time-dependent decreasing model.

        Decreases inertia based on time. A time-dependent decreasing inertia
        has been shown to exhibit better performance over a fixed inertia.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        # NOTE: Potential typo in the paper, unsure if the important ratio
        # is supposed to be t/t_max or v/v_max. Update after checking with
        # Hassen.
        # 
        # self.w = self.w_bounds[1]
        #        - ((self.t / self.t_bounds[1])
        #           * (self.w_bounds[1]-self.w_bounds[0]))
        pass 

    def shuffle_gbest(self):
        """Randomize gbest after stagnation.

        Randomly reassigns gbest and saves the old gbest in abest. This method
        implements the "shuffle and archive" functionality described in the
        COMB-PSO algorithm. This method should only be used when gbest has
        been updated 3 times and has not changed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        assert(self.gbest_counter = 3)
        self.gbest = np.random.uniform(low=self.x_bounds[0],
                                       high=self.x_bounds[1],
                                       size=self.ndim)
        self.g_fitness = self.eval_fitness(self.gbest)
        if g > self.a_fitness:
            self.abest = self.gbest.copy()
            self.a_fitness = self.g_fitness
        self.gbest_counter = 0 # NOTE: Make sure the math lines up

    def eval_fitness(self, classifier, classifier_args):
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
    



