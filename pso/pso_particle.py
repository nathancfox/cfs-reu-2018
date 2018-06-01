#-----------------------------------------------------------------------------+
#
# Title: Particle Swarm Optimization Particle Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 1, 2018
# Date Modified: June 1, by Nathan Fox <nathanfox@miami.edu>
#
#-----------------------------------------------------------------------------+



import numpy as np

class Particle:
    # x:Position, v:Velocity, w:Inertia, c1/c2:Cog/Social constants,
    # pbest:particle's best vector, gbest:global best vector
    var_template = {
                    'x': np.array([]),
                    'v': np.array([]),
                    'w': 0.0,
                    'c1': 0.0,
                    'c2': 0.0,
                    'pbest': np.array([]),
                   }

    def __init__(self, w, c1, c2, ndim):
        """Initialize the Particle object.

        Initializes a Particle object. The new Particle gets a copy of the
        class variable, var_template, as an instance variable, var. This
        allows readers to see the expected structure of the dict holding
        the Particle state variables, but avoids mutating it for all instances
        of the class. The new Particle then randomly initializes its position
        and velocity vectors and assigns the inertia, c1, and c2 variables.

        Parameters
        ----------
        w : float, positive inertia weight.

        c1 : float, acceleration coefficient for the cognitive component.

        c2 : float, acceleration coefficient for the social component.

        ndim : integer, number of dimensions or features in the search space.
               Also the length of the position and velocity vectors.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.var = var_template.copy()
        self.randomize_position(ndim)
        self.randomize_velocity(ndim)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
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

    def update_velocity(self, gbest):
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

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.v = (
                   (self.w*self.v)
                 + (self.c1*np.random.uniform()*(self.pbest-self.x))
                 + (self.c2*np.random.uniform()*(gbest-self.x))
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
