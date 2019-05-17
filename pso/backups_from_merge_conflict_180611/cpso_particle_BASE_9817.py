#------------------------------------------------------------------------------+
#
# Title: COMB-PSO Particle Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 1, 2018
# Date Modified: June 6, by Nathan Fox <nathanfox@miami.edu>
#
#------------------------------------------------------------------------------+

# TODO (nathanfox@miami.edu): See about saving seeds/random number progression,
# update the docstrings, proofread, and write test cases.

import numpy as np
from scipy.special import expit

class COMB_Particle:

    """COMB-Particle Swarm Optimization (COMB-PSO) Particle.

    COMB_Particle is an implementation of a particle object in
    Combined-Particle Swarm Optimization (COMB-PSO). It is designed to be
    used by a larger swarm class or swarm program.

    Attributes
    ----------
    x : 1-Dimensional ndarray; Holds the current particle position.
    
    v : 1-Dimensional ndarray; Holds the current particle velocity.

    b : 1-Dimensional ndarray; Holds the current binary position.

    pbest : 1-Dimensional ndarray; Holds the particle's best position to date.

    w : float; inertia coefficient.
    
    c1 : float; acceleration constant 1, for the cognitive component.

    c2 : float; acceleration constant 2, for the social component.

    c3 : float; acceleration constant 3, for the diversity component.

    ndim : integer; number of dimensions in the search space; also the size
           of x, v, b, and pbest.

    x_bounds : float tuple; size 2; Holds upper/lower bounds for elements in x.

    v_bounds : float tuple; size 2; Holds upper/lower bounds for elements in v.

    w_bounds : float tuple; size 2; Holds upper/lower bounds for w.

   
    Functions
    ---------
    __init__ : Initializes a COMB_Particle object and assigns all attributes.

    update_position : Updates position based on velocity.

    update_velocity : Updates velocity based on the COMB-PSO velocity equation.

    initialize_position : Randomly initializes position.

    initialize_velocity : Randomly initializes velocity.

    update_binary_position : Updates binary position by converting current x.

    update_inertia : Updates inertia based on time.
    """

    def __init__(self, c1, c2, c3, ndim, x_bounds, v_bounds, w_bounds):
        """Initialize the COMB_Particle object.

        Initializes a COMB_Particle object. The new COMB_Particle then randomly
        initializes its position, binary position, velocity, and pbest vectors,
        then assigns the inertia, c1, c2, c3 variables and the respective
        bounds.
        
        NOTE: p_fitness must be initialized in the swarm class because
        of a readability-based design decision.

        Parameters
        ----------
        c1 : float, acceleration coefficient for the cognitive component.

        c2 : float, acceleration coefficient for the social component.

        c3 : float, acceleration coefficient for the diversity component.

        ndim : integer, number of dimensions or features in the search space.
               Also the length of the position and velocity vectors.

        x_bounds : tuple of floats, size 2, x_bounds[0] is the minimum value
                   that an element of x can be; x_bounds[1] is the maximum
                   value that an element of x can be.

        v_bounds : tuple of floats, size 2, v_bounds[0] is the minimum value
                   that an element of v can be; v_bounds[1] is the maximum
                   value that an element of v can be.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.x = np.zeros(ndim)
        self.v = np.zeros(ndim)
        self.randomize_position(ndim)
        self.randomize_velocity(ndim)
        self.b = np.zeros(ndim, dtype=np.int8)
        self.update_binary_position()
        self.pbest = x.copy()
        self.p_fitness = 0.0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ndim = ndim
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        
    def update_position(self):
        """Update the position vector for one time step.

        Updates the position vector for a single time step, according
        to the PSO position equation.
        
        NOTE: The velocity vector must be updated BEFORE the position vector.

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
        self.x = self.x * self.v
        # Clipping position if outside bounds
        self.x[self.x < self.x_bounds[0]] = self.x_bounds[0]
        self.x[self.x > self.x_bounds[1]] = self.x_bounds[1]

    def update_velocity(self, w, gbest, abest):
        """Update the velocity vector for one time step.

        Updates the velocity vector for a single time step, according
        to the COMB-PSO velocity equation. The equation terms, in order,
        represent the inertia component, the cognitive component, the
        social component, and the diversity component.

        NOTE: The velocity vector must be updated BEFORE the position vector.

        Parameters
        ----------
        w : float, inertia coefficient.
        
        gbest : ndarray, shape: (ndim,), the best position vector found by
                any Particle so far.

        abest : ndarray, shape: (ndim,), the archived best position vector
                found by any particle so far. Used as an archive to allow
                gbest to be shuffled on stagnation.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        # NOTE: Remember to come back and make sure that you're recording the
        # random values if necessary. Or record a seed at the beginning.
        self.v = (
                   (w*self.v)
                 + (self.c1*np.random.uniform(size=self.ndim)*(self.pbest-self.x))
                 + (self.c2*np.random.uniform(size=self.ndim)*(gbest-self.x))
                 + (self.c3*np.random.uniform(size=self.ndim)*(abest-self.x))
                 )
        # Clipping velocity if outside bounds
        self.v[self.v < self.v_bounds[0]] = self.v_bounds[0]
        self.v[self.v > self.v_bounds[1]] = self.v_bounds[1]

    def initialize_position(self):
        """Initialize the position vector, x.

        Randomizes the initial position vector of the Particle. Typically used
        for Particle initialization at the beginning of a PSO search. Expects
        the position vector to already have the correct length.

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
        # This implementation cannot start at the upper bound,
        # Not sure if that's a problem
        self.x = np.random.uniform(low=self.x_bounds[0],
                                   high=self.x_bounds[1],
                                   size=self.ndim)
    
    def initialize_velocity(self, ndim):
        """Initialize the velocity vector, v.

        Randomizes the initial velocity vector of the Particle. Typically used
        for Particle initialization at the beginning of a PSO search. Expects
        the velocity vector to already have the correct length.

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
        # This implementation cannot start at the upper bound,
        # Not sure if that's a problem
        v = np.random.uniform(low=self.v_bounds[0],
                              high=self.v_bounds[1],
                              size=self.ndim)

    def update_binary_position(self):
        """Convert continuous position vector to binary position vector.

        Updates the binary position vector by using a logistic function
        to convert a continuous position vector, x, into a binary position
        vector, b. This allows the particle to explore a continuous space,
        but still report a position as a subset of features.

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
        # NOTE: Remember to come back and make sure that you're recording the
        # random values if necessary. Or record a seed at the beginning.

        # Use if random numbers need to be saved
        # rand_compar = np.random.uniform(size=self.ndim)
        # Save rand_compar to file
        # self.b = (self.x < rand_compar).astype(int)
        
        # Use if random numbers do Not need to be saved
        self.b = (self.x < np.random.uniform(size=self.ndim)).astype(int)

    
