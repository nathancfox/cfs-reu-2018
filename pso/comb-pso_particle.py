#------------------------------------------------------------------------------+
#
# Title: COMB-PSO Particle Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 1, 2018
# Date Modified: June 1, by Nathan Fox <nathanfox@miami.edu>
#
#------------------------------------------------------------------------------+

# TODO: Add clipping, double check on random selection intervals
# see about saving seeds/random number progression, check inertia equation
# typo, write a sigmoid function, generally finish and check against paper.

import numpy as np

class COMB_Particle:
    def __init__(self, w, c1, c2, c3, ndim, x_bounds, v_bounds, w_bounds):
        """Initialize the COMB_Particle object.

        Initializes a COMB_Particle object. The new COMB_Particle gets a copy
        of the class variable, var_template, as an instance variable, var. This
        allows readers to see the expected structure of the dict holding
        the COMB_Particle state variables, but avoids mutating it for all
        instances of the class. The new COMB_Particle then randomly
        initializes its position and velocity vectors and assigns the inertia,
        c1, c2, and c3 variables.

        Parameters
        ----------
        w : float, positive inertia weight.

        c1 : float, acceleration coefficient for the cognitive component.

        c2 : float, acceleration coefficient for the social component.

        c3 : float, acceleration coefficient for the diversity component.

        ndim : integer, number of dimensions or features in the search space.
               Also the length of the position and velocity vectors.

        x_bounds : tuple of floats, size 2, x_bounds[0] is the minimum value
                   that an element of x can hold.

        v_bounds : tuple of floats, size 2, v_bounds[0] is the minimum value
                   that an element of v can hold; v_bounds[1] is the maximum
                   value that an element of v can hold.

        w_bounds : tuple of floats, size 2, w_bounds[0] is the minimum value
                   that w can be; w_bounds[1] is the maximum value that w
                   can be.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.var = var_template.copy()
        self.randomize_position(ndim)
        self.b = np.zeros(ndim)
        self.update_binary_position()
        self.randomize_velocity(ndim)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ndim = ndim
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        
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

    def update_velocity(self, gbest, abest):
        """Update the velocity vector for one time step.

        Updates the velocity vector for a single time step, according
        to the PSO velocity equation. The equation terms, in order, represent
        the inertia component, the cognitive component, and the social
        component.

        NOTE: The velocity vector must be updated BEFORE the position vector.

        Parameters
        ----------
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
                   (self.w*self.v)
                 + (self.c1*np.random.uniform()*(self.pbest-self.x))
                 + (self.c2*np.random.uniform()*(gbest-self.x))
                 + (self.c3*np.random.uniform()*(abest-self.x))
                 )

    def initialize_position(self, ndim):
        """Initialize the position vector, var['x'].

        Randomizes the initial position vector of the Particle. Typically used
        for Particle initialization at the beginning of a PSO search.

        Parameters
        ----------
        ndim : integer, position vector length, or the number of dimensions
               or features in the search space.
           
        Returns
        -------
        None

        Raises
        ------
        None
        """
        pass
    
    def initialize_velocity(self, ndim):
        """Initialize the velocity vector, var['v'].

        Randomizes the initial velocity vector of the Particle. Typically used
        for Particle initialization at the beginning of a PSO search.

        Parameters
        ----------
        ndim : integer, velocity vector length, or the number of dimensions
               or features in the search space.
           
        Returns
        -------
        None

        Raises
        ------
        None
        """
        pass

    def sigmoid(self, x_i):
        # Some sigmoid conversion function to be used in
        # update_binary_position().
        pass
    def update_binary_position(self):
        """Convert continuous position vector to binary position vector.

        Updates the binary position vector by using a sigmoid function,
        defined internally, to convert a continuous position vector, x,
        into a binary position vector, b. This allows the particle to
        explore a continuous space, but still report a position as a subset
        of features.

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
        for i in range(ndim):
            if np.random.uniform() < sigmoid(self.x[i]):
                self.b[i] = 1
            else:
                self.b[i] = 0

    def update_inertia(self, t, t_max):
        # NOTE: Potential typo in the paper, unsure if the important ratio
        # is supposed to be t/t_max or v/v_max. Update after checking with
        # Hassen.
        # 
        # self.w = w_bounds[1] - ((t / t_max) * (w_bounds[1]-w_bounds[0]))
        pass
    def evaluate_fitness(self, classifier, classifier_args):
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
