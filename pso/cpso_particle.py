#------------------------------------------------------------------------------+
#
# Title: COMB-PSO Particle Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 1, 2018
# Date Modified: June 11, by Nathan Fox <nathanfox@miami.edu>
#
#------------------------------------------------------------------------------+

# TODO (nathanfox@miami.edu): update the docstrings.

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

    b : 1-Dimensional ndarray; Holds the current binary position as
        an array of booleans.

    pbest : 1-Dimensional ndarray; Holds the particle's best position to date.

    pbinary : 1-Dimensional ndarray, size ndim; Holds the binary best position 
              converted from pbest as a list of booleans.

    p_fitness : float; the fitness value returned by COMB_Swarm.eval_fitness
                for the current pbest.

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
        self.w = 0 # Initialized in the swarm class in execute_search()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ndim = ndim
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        self.x = np.zeros(ndim)
        self.v = np.zeros(ndim)
        self.initialize_position()
        self.initialize_velocity()
        self.b = np.zeros(ndim) # Unnecessary, but a placeholder for readability
        self.update_binary_position()
        self.pbest = self.x.copy()
        self.pbinary = self.b.copy()
        self.p_fitness = 0.0 # Initialized in the swarm class.
                
    def update_position(self):
        """Update the position vector for one time step.

        Updates the position vector for a single time step, according
        to the COMB-PSO position equation. Clips to within the bounds
        given in x_bounds.
        
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
        self.x = self.x + self.v
        # Clipping position if outside bounds
        self.x[self.x < self.x_bounds[0]] = self.x_bounds[0]
        self.x[self.x > self.x_bounds[1]] = self.x_bounds[1]

    def update_velocity(self, gbest, abest):
        """Update the velocity vector for one time step.

        Updates the velocity vector for a single time step, according
        to the COMB-PSO velocity equation. The equation terms, in order,
        represent the inertia component, the cognitive component, the
        social component, and the diversity component. Clips to within the
        bounds given in v_bounds.

        NOTE: The velocity vector must be updated BEFORE the position vector.

        Parameters
        ----------
        gbest : ndarray, shape (ndim,); the best position vector found by
                any particle so far.

        abest : ndarray, shape (ndim,); the archived best position vector
                found by any particle so far. Used as an archive to allow
                gbest to be shuffled on stagnation.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.v = (
                   (self.w*self.v)
                 + (self.c1*np.random.uniform(size=self.ndim)*(self.pbest-self.x))
                 + (self.c2*np.random.uniform(size=self.ndim)*(gbest-self.x))
                 + (self.c3*np.random.uniform(size=self.ndim)*(abest-self.x))
                 )
        # Clipping velocity if outside bounds
        self.v[self.v < self.v_bounds[0]] = self.v_bounds[0]
        self.v[self.v > self.v_bounds[1]] = self.v_bounds[1]

    def initialize_position(self):
        """Initialize the position vector, x.

        Randomizes the initial position vector of the particle. Typically used
        for particle initialization at the beginning of a COMB-PSO search.

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
    
    def initialize_velocity(self):
        """Initialize the velocity vector, v.

        Randomizes the initial velocity vector of the particle. Typically used
        for particle initialization at the beginning of a COMB-PSO search.

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
        self.v = np.random.uniform(low=self.v_bounds[0],
                              high=self.v_bounds[1],
                              size=self.ndim)

    def update_binary_position(self):
        """Convert continuous position vector to binary position vector.

        Updates the binary position vector, b, by converting the current
        continuous position vector, x using a passed function.

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
        # Use if random numbers need to be saved
        # rand_compar = np.random.uniform(size=self.ndim)
        # Save rand_compar to file
        # self.b = (self.x < rand_compar).astype(int)
        
        # Use if random numbers do Not need to be saved
        self.b = self.convert_pos_to_binary(self.x)

    def convert_pos_to_binary(self, x):
        """Convert a continuous position vector to a binary position vector.

        Converts any continuous position vector to a binary position
        vector according to the following formula.

                 { True, if rand() < S(x[i,j])  
        b[i,j] = {
                 { False, otherwise

        where b is the binary position vector, b[i] is the binary position
        vector of particle i, b[i, j] is a feature (True for inclusion,
        False for exclusion) in b[i], rand() is a uniform random
        number ϵ [0.0, 1.0), S is a logistic transformation, x is the
        continuous position vector, and x[i]/x[i,j] are analogous to
        b[i]/b[i,j].

        Parameters
        ----------
        x : 1-Dimensional ndarray; holds a continuous position vector.

        Returns
        -------
        binary : ndarray, size ndim; boolean ndarray that holds the
                 binary version of x, a continuous position vector.

        Raises
        ------
        None
        """
        return (np.random.uniform(size=x.size) < expit(x))

    def update_inertia(self, gbinary):
        """Update inertia coefficient, w.

        Updates inertia coefficient according to a function that correlates
        inertia based on distance from gbinary. A binary Jaccard correlation
        index is used to determine similarity between the current binary
        position vector, b, and the binary global position vector, gbinary.
        The closer the particle is to gbinary, the lower the inertia. This
        favors exploration while far away from gbest and exploitation when
        close to gbest.

        Parameters
        ----------
        gbinary : 1-Dimensional ndarray, size ndim; Holds the binary global
                  best position.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.w = (self.w_bounds[1]
                  - ((self.w_bounds[1]-self.w_bounds[0])
                         * self.jaccard_index(gbinary)))
   
    def jaccard_index(self, gbinary):
        """Return binary Jaccard index coefficient between x and gbinary.

        Calculates the binary Jaccard index coefficient between the
        internal binary position vector, b, and the passed argument
        binary global position vector, gbinary. The following formula
        is used:
                           M11
        J(b, gbinary) = ---------
                         n - M00

        where M11 is the number of 1-1 matches between the binary strings
        and M00 is the number of 0-0 matches between the binary strings.
        n is the particle size, or length of either vector.

        NOTE: This method does not catch the exception where both strings
              are entirely False/0 which would cause division by 0. It
              should be added in a later version, but for now, it seems
              inconceivable that an implementation of COMB-PSO would ever
              return a gbinary that was all 0's (none of the features
              included in the predictive subset).

        Parameters
        ----------
        gbinary : 1-Dimensional ndarray, size ndim; Holds the binary
                  global best position.

        Returns
        -------
        j : float; binary Jaccard index coefficient calculated between
            b and gbinary. j ϵ [0.0, 1.0].

        Raises
        ------
        None
        """
        m11, m00 = 0, 0
        for i in range(self.ndim):
            if gbinary[i] == self.b[i]:
                if gbinary[i] == True:
                    m11 += 1
                else:
                    m00 += 1
        return (m11 / (self.ndim - m00))
