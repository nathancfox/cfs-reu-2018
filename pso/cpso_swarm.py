#-----------------------------------------------------------------------------+
# 
# Title: COMB-PSO Swarm Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 6, 2018
# Date Modified: June 14, 2018, by Nathan Fox <nathanfox@miami.edu>
#
#-----------------------------------------------------------------------------+

# TODO: Update docstrings.
#
# Evolutionary Functionality: 
#
#   It should record the progression of gbest and abest, and possibly
#   also each particle's pbest. It also maybe should pickle the classifier?
#   It also might want to take freezes of the initial states, and record
#   all the positions and velocities for each particle over time and export.
#   The fitness values should almost certainly be recorded and exported
#   because they take so much time to compute.
#
#   The program should be callable as a script with arguments. This means
#   that as many currently-hard-coded parameters should be class
#   attributes as possible, so that wrapper scripts using this class
#   can pass them as arguments. Maybe pass them in a dictionary?
#
#   The program should also have options on what data to store because
#   i/o affects runtime so much. If the entire path of each particle
#   is not needed, it shouldn't be recorded to file.

import numpy as np
from scipy.special import expit
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from cpso_particle import COMB_Particle

class COMB_Swarm:

    """COMB-Particle Swarm Optimization (COMB-PSO) Swarm.

    COMB_Swarm is an implementation of a swarm object in
    Combined-Particle Swarm Optimization. It contains a group of
    COMB_Particle objects and maintains group attributes.

    Attributes
    ----------
    swarm : list, size npart; holds the swarm of COMB_Particle objects.

    npart : integer; number of particles in the swarm.
    
    gbest : 1-Dimensional ndarray, size ndim; Holds the global best position
            found by any particle in the swarm. Subject to reshuffling based
            on the "shuffle and archive" behavior in the COMB-PSO algorithm.

    g_fitness : float; the fitness value returned by eval_fitness for the
                current gbest.

    gbest_counter : integer; stagnation counter for gbest. If it reaches 3,
                    shuffle_gbest is called.

    gbinary : 1-Dimensional ndarray, size ndim; Holds the binary global
              best position as a list of booleans.

    abest : 1-Dimensional ndarray, size ndim; Holds the archived best
            position found by any particle in the swarm, even if the global
            best has been shuffled.

    a_fitness : float; the fitness value returned by eval_fitness for the
                current abest.

    abinary : 1-Dimensional ndarray, size ndim; Holds the binary archived
              best position as a list of booleans.

    t : integer; current time.

    w : float; inertia coefficient.

    c1 : float; acceleration constant 1, for the cognitive component

    c2 : float; acceleration constant 2, for the social component

    c3 : float; acceleration constant 3, for the diversity component

    ndim : integer; number of dimensions in the search space; also the size
           of the position and velocity vectors.

    alpha : float, alpha ϵ [0.0, 1.0]; weights number of features vs.
            classification performance in evaluating fitness.

    x_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the position vectors.

    v_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the velocity vectors.

    w_bounds : float tuple, size 2; Holds lower/upper bounds for the
               inertia coefficient.

    t_bounds : integer tuple, size 2; Holds beginning and end times for
               the algorithm.

    clf : sklearn classifier object, currently an SVC (Support Vector Machine
          Classifier); To be used in evaluating the fitness of a position
          vector.

    data : 2-Dimensional ndarray, size (number_of_data_points, ndim); holds the
           feature data, each row is a data point and each column is a feature.

    target : 1-Dimensional ndarray, size number_of_data_points; holds the
             correct classifications for the data points in data.

    test_size : float, test_size ϵ (0.0, 1.0); denotes the percentage of data
                that should be reserved to use for final evaluation.

    X_train : 2-Dimensional ndarray,
              size (1-test_size*number_of_data_points, ndim); feature data
              to be used for training.

    X_test : 2-Dimensional ndarray,
             size (test_size*number_of_data_points, ndim); feature data to be
             use for final evaluation.

    y_train : 1-Dimensional ndarray,
              size (1-test_size)*number_of_data_points;
              correct classifications of X_train to be used in training.

    y_test : 1-Dimensional ndarray, size test_size*number_of_data_points;
             correct classifications of X_test to be used for final evaluation.

    final_scores : n-Dimensional ndarray, size K where K-Fold Cross Validation
                   is being used for evaluation. Used to hold the CV scores
                   from the reserved test data that was unused during 
                   COMB-PSO.

    Functions
    ---------
    __init__ : Initializes a COMB_Swarm object and assigns all attributes.

    initialize_particles : Fills the swarm with initialized COMB_Particles. 

    initialize_classifier : Initializes the classifier used in eval_fitness.

    execute_search : Execute one full run of the COMB-PSO algorithm.

    shuffle_gbest : "Shuffle and Archive", randomizes gbest after stagnation.

    convert_pos_to_binary : Converts a position vector to a binary one.

    test_classify : Returns classification performance for a given position.

    eval_fitness : Evaluates the fitness function for a position vector.

    final_eval : Runs a final classification evaluation on reserved test data.
    """
    
    def __init__(self, npart, c1, c2, c3, ndim, alpha, test_size,
                 x_bounds, v_bounds, w_bounds, t_bounds,
                 data_path, target_path):
        """Initialize the COMB_Swarm object.

        Initializes a COMB_Swarm object. The new COMB_Swarm then assigns its
        attributes and creates an empty Python list that will eventually hold
        the COMB_Particle objects. The actual initialization of the particles
        that make up the swarm is separated into the initialize_particles
        method because it involves npart calls to the eval_fitness method.
        This is potentially computationally expensive and should not be called
        during first initialization, instead called manually afterward.

        Parameters
        ----------
        npart : integer; number of particles in the swarm.

        c1 : float; acceleration coefficient for the cognitive component.

        c2 : float; acceleration coefficient for the social component.

        c3 : float; acceleration coefficient for the diversity component.

        ndim : integer; number of dimensions or features in the search space.
               Also the length of the position and velocity vectors.

        alpha : float, ϵ [0.0, 1.0]; weights number of features vs.
                classification performance in evaluating fitness. If alpha
                = 1.0, classification performance is the only contributing
                factor to fitness. If alpha = 0.0, minimizing the number
                of features is the only contributing factor to fitness.

        test_size : float, ϵ [0.0, 1.0]; designates the portion of the data
                    to be reserved for final testing.

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

        data_path : string; path to the .csv file holding the feature data.
                    The file should have one data point per line. Each line
                    should have all the features for that data point, separated
                    by commas. There should be no header row or index column.

        target_path : string; path to the .csv file holding the correct
                      classifications for the data points in the file pointed
                      to by data_path. This file should have 1 line with all
                      the labels, separated by commas.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.npart = npart
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ndim = ndim
        self.alpha = alpha
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        self.t_bounds = t_bounds
        self.all_false = 0
        # Placeholders
        self.gbest = np.zeros(self.ndim)
        self.gbest_counter = 0
        self.gbinary = np.zeros(self.ndim)
        self.g_fitness = 0.0
        self.g_score = 0.0
        self.abest = np.zeros(self.ndim)
        self.abinary = np.zeros(self.ndim)
        self.a_fitness = 0.0
        self.a_score = 0.0
        self.swarm = []

        self.clf = svm.SVC()
        self.data = np.loadtxt(data_path, dtype=np.float64, delimiter=',')
        self.target = np.loadtxt(target_path, dtype=np.int8, delimiter=',')
        self.test_size = test_size
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(self.data, self.target,
                                                       test_size=self.test_size)
        self.final_scores = np.zeros(10)
        self.var_by_time = {'num_features': np.zeros(t_bounds[1]).astype(int),
                            'g_fitness': np.zeros(t_bounds[1]),
                            'a_fitness': np.zeros(t_bounds[1]),
                            'a_score': np.zeros(t_bounds[1])
                           }

    def initialize_particles(self):
        """Initialize the particles that comprise the swarm.

        Actually fills the empty list, swarm, with COMB_Particle objects.
        Separated particle initialization from the __init__ function
        because of a high computational time cost. Wrapper scripts using
        the COMB_Swarm class should manually initialize the particles.

        NOTE: The inertia coefficient, w, is not initialized here because
              it is updated based on gbinary as the very first thing in
              the actual run of the algorithm in execute_search. It is
              only initialized as a placeholder, 0.0 in the particle
              __init__. If you use these classes out of context, be
              sure to know that the w attribute may not be what you expect.

        NOTE: See "# NOTE REFERENCE" below. That line of code is manually
              initializing the p_fitness attribute for each COMB_Particle
              in the COMB_Swarm. The author recognizes that having to
              manually initialize an attribute from outside a class is
              horrible programming practice. However, this was a
              requirement for greater readability in this method. If the
              pbest attributes were stored in a data structure in the
              COMB_Swarm class, the for loop in execute_search would read as
              "for i in range(self.npart):" instead of "for p in self.swarm",
              causing all references to the COMB_Particle to be "swarm[i]"
              instead of "p", greatly reducing readability of the main
              algorithm of this class. Thus, the author decided to store
              each pbest attribute inside its respective COMB_Particle.
              Because the eval_fitness method is in the COMB_Swarm class,
              this means that the COMB_Particle cannot initialize its own
              pbest attribute. Thus, it is done here.

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
            f, f_score = self.eval_fitness(self.swarm[i].b)
            # NOTE REFERENCE : See docstring above.
            self.swarm[i].p_fitness = f
            self.swarm[i].p_score = f_score
            if i == 0:
                self.gbest = self.swarm[i].x.copy()
                self.gbinary = self.swarm[i].b.copy()
                self.g_fitness = f
                self.g_score = f_score
            elif f > self.g_fitness:
                self.gbest = self.swarm[i].x.copy()
                self.gbinary = self.swarm[i].b.copy()
                self.g_fitness = f
                self.g_score = f_score
        self.abest = self.gbest.copy()
        self.abinary = self.gbinary.copy()
        self.a_fitness = self.g_fitness
        self.a_score = self.g_score
        self.var_by_time['num_features'][0] = np.count_nonzero(self.abinary)
        self.var_by_time['g_fitness'][0] = self.g_fitness
        self.var_by_time['a_fitness'][0] = self.a_fitness
        self.var_by_time['a_score'][0] = self.a_score

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
        None

        abinary : 1-Dimensional ndarray, size ndim; Holds the binary archived
                  best position. 

        Raises
        ------
        None
        """
        for i in range(1, self.t_bounds[1]):
            self.t = i
            for p in self.swarm:
                p.update_inertia(self.gbinary)
                p.update_velocity(self.gbest, self.abest)
                p.update_position()
                p.update_binary_position()
                f, f_score = self.eval_fitness(p.b)
                if f > p.p_fitness:
                    p.pbest = p.x.copy()
                    p.pbinary = p.b.copy()
                    p.p_fitness = f
                    p.p_score = f_score
                if f > self.g_fitness:
                    # -1 because the counter should be 0 during the
                    # next comparison to updated positions. The other
                    # 2 resets (shuffle_gbest and init) happen after
                    # the "self.gbest_counter += 1" line. This one
                    # happens before.
                    self.gbest_counter = -1
                    self.gbest = p.x.copy()
                    self.gbinary = p.b.copy()
                    self.g_fitness = f
                    self.g_score = f_score
            if self.g_fitness > self.a_fitness:
                self.abest = self.gbest.copy()
                self.abinary = self.gbinary.copy()
                self.a_fitness = self.g_fitness
                self.a_score = self.g_score
            self.gbest_counter += 1
            if self.gbest_counter >= 3:
                self.shuffle_gbest()
            self.var_by_time['num_features'][i] = np.count_nonzero(self.abinary)
            self.var_by_time['g_fitness'][i] = self.g_fitness
            self.var_by_time['a_fitness'][i] = self.a_fitness
            self.var_by_time['a_score'][i] = self.a_score
    
    def shuffle_gbest(self):
        """Randomize gbest after stagnation.

        Randomly reassigns gbest and saves the old gbest in abest, unless
        the new gbest is better than the old gbest. This method implements
        the "shuffle and archive" functionality described in the COMB-PSO
        algorithm. This method should only be used when gbest has been
        updated 3 times and has not changed.

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
        assert(self.gbest_counter == 3)
        self.gbest = np.random.uniform(low=self.x_bounds[0],
                                       high=self.x_bounds[1],
                                       size=self.ndim)
        self.gbinary = self.convert_pos_to_binary(self.gbest)
        self.g_fitness, self.g_score = self.eval_fitness(self.gbinary)
        if self.g_fitness > self.a_fitness:
            self.abest = self.gbest.copy()
            self.abinary = self.gbinary.copy()
            self.a_fitness = self.g_fitness
            self.a_score = self.g_score
        self.gbest_counter = 0

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

    def test_classify(self, b):
        """Return a classification performance for a binary position vector.

        Runs a classification with the clf attribute as a classifier and
        X_train and y_train as the feature data/correct classifications.
        Can be modified to classify any way, as long as it returns a metric
        of performance.

        Currently runs a 10-fold cross validation and returns the mean
        classification accuracy.

        Parameters
        ----------
        b : 1-Dimensional ndarray, size ndim; Holds a binary position vector
            where each value ϵ {0, 1} and represents respectively the
            exclusion or inclusion of that feature in the subset used for
            training.
        
        Returns
        -------
        scores.mean() : The mean of the classification accuracies returned
                        by the 10-fold cross validation.

        Raises
        ------
        None
        """
        scores = cross_val_score(self.clf, self.X_train[:, b],
                                 y=self.y_train, cv=10)
        return scores.mean()
        
    def eval_fitness(self, b):
        """Evaluate the fitness of the current position vector.

        Evaluates the fitness of the current binary position vector or feature
        subset by using the current binary position vector to train a
        classifier and test predictive performance.

        Fitness Function:

        f(X, y, b, alpha) = alpha * Pb + (1-alpha) * (|X|-|b|)/|X|

        where X is the set of all features,
              y is the set of class label values,
              b is the subset of features selected
              alpha is a weight factor that denotes importance to size of the
                    subset and accuracy of the classifier
              Pb is the classification score given only b (decoded by y
                 and the position vector)
              |X| is the number of features (size of position vector)
              |b| is the number of features in b

        Parameters
        ----------
        b : 1-Dimensional ndarray, size ndim; Holds a binary position vector
            where each value ϵ {0, 1} and represents respectively the
            exclusion or inclusion of that feature in the subset used for
            training.

        Returns
        -------
        f : float tuple, size 2; f[0] is the value of the fitness function,
                                 f[1] is the classification score.

        Raises
        ------
        None
        """
        if np.count_nonzero(b) == 0:
            self.all_false += 1
            return 0.0, 0.0 
        # clf_perf is the same as Pb in the above equation
        clf_perf = self.test_classify(b)
        f = ((self.alpha*clf_perf)
             + (   (1-self.alpha)
                 * ((self.ndim-np.count_nonzero(b)) / self.ndim)
               )
            )
        # print(str(f)+'\n'+str(clf_perf))
        # sys.exit(0)
        return f, clf_perf

    def final_eval(self):
        """Perform final evaluation of best position.

        Assigns the results from a final cross-validation on data that
        was reserved during creation of the dataset to final_scores.
        This data was not used at all during the entire COMB-PSO
        algorithm and is used to test the actual accuracy of the
        feature subset resulting from the algorithm.

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
        self.clf.fit(self.X_train[:, self.abinary], self.y_train)
        self.final_scores = cross_val_score(self.clf,
                                            self.X_test[:, self.abinary],
                                            self.y_test, cv=10)
