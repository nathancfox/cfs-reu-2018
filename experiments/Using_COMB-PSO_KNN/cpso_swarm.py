# Title: COMB-PSO Swarm Class
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 6, 2018
# Date Modified: July 2, 2018, by Nathan Fox <nathanfox@miami.edu>
#
#-----------------------------------------------------------------------------+

# Evolutionary Functionality: 
#
#   The program should have options on what data to store because
#   i/o affects runtime so much. For example: if the entire path of each
#   particle is not needed, it shouldn't be recorded to file.

import numpy as np
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from cpso_particle import COMB_Particle

import pprint as pp

class COMB_Swarm:

    """COMB-Particle Swarm Optimization (COMB-PSO) Swarm.

    COMB_Swarm is an implementation of a swarm object in
    Combined-Particle Swarm Optimization. It contains a group of
    COMB_Particle objects and maintains group attributes.

    Attributes
    ----------
    npart : integer; number of particles in the swarm.
    
    c1 : float; acceleration constant 1, for the cognitive component

    c2 : float; acceleration constant 2, for the social component

    c3 : float; acceleration constant 3, for the diversity component

    ndim : integer; number of dimensions in the search space; also the size
           of the position and velocity vectors.

    terms : dict; key is a string indicating a certain term of the
            fitness function. value is a float indicating the actual
            weight of that term.

    x_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the position vectors.

    v_bounds : float tuple, size 2; Holds lower/upper bounds for elements
               in the velocity vectors.

    w_bounds : float tuple, size 2; Holds lower/upper bounds for the
               inertia coefficient.

    t_bounds : integer tuple, size 2; Holds beginning and end times for
               the algorithm.
    
    t : integer; current time.

    all_false_counter : integer, > 0; Counter for the number of times that
                        a binary position of all False (0 feature subset)
                        was passed to the fitness function.

    gbest : 1-Dimensional ndarray, size ndim; Holds the global best position
            found by any particle in the swarm. Subject to reshuffling based
            on the "shuffle and archive" behavior in the COMB-PSO algorithm.

    gbest_counter : integer; stagnation counter for gbest. If it reaches 3,
                    shuffle_gbest is called.

    stagn_limit : int, > 0; number of times that gbest can stagnate
                  before shuffle_gbest() is called. Default 3.

    gbinary : 1-Dimensional ndarray, size ndim; Holds the binary global
              best position as a list of booleans.

    g_fitness : float; the fitness value returned by eval_fitness for the
                current gbest.

    g_score : float tuple, size 3, elements ϵ [0.0, 1.0]; holds the
              accuracy, sensitivity, and specificity for the classifier's
              performance using gbinary.

    abest : 1-Dimensional ndarray, size ndim; Holds the archived best
            position found by any particle in the swarm, even if the global
            best has been shuffled.

    abinary : 1-Dimensional ndarray, size ndim; Holds the binary archived
              best position as a list of booleans.

    a_fitness : float; the fitness value returned by eval_fitness for the
                current abest.

    a_score : float tuple, size 3, elements ϵ [0.0, 1.0]; holds the
              accuracy, sensitivity, and specificity for the classifier's
              performance using abinary.

    swarm : list, size npart; holds the swarm of COMB_Particle objects.

    p_fitness : float ndarray, size npart; holds the current fitness for
                each particle in the swarm.

    p_scores : float ndarray, size (npart, 3); holds the (accuracy,
               sensitivity, specificity) tuple for each particle in the
               swarm.

    clf : sklearn classifier object, currently a KNeighborsClassifier; 
          To be used in evaluating the fitness of a position
          vector.

    data : 2-Dimensional ndarray, size (number_of_data_points, ndim); holds the
           feature data, each row is a data point and each column is a feature.

    target : 1-Dimensional ndarray, size number_of_data_points; holds the
             correct classifications for the data points in data.

    var_by_time : dict; key is a string indicating a certain variable, value is
                  an ndarray holding the value of that variable for each time
                  during the run of execute_search().

    init_flag : boolean; True when self is fully initialized (both __init__()
                and initialize_particles() have been called).

    Functions
    ---------
    __init__ : Initializes a COMB_Swarm object and assigns all attributes.

    initialize_particles : Fills the swarm with initialized COMB_Particles. 

    execute_search : Execute one full run of the COMB-PSO algorithm.

    shuffle_gbest : "Shuffle and Archive", randomizes gbest after stagnation.

    convert_pos_to_binary : Converts a position vector to a binary one.

    test_classify : Returns classification performance for a given position.

    eval_fitness : Evaluates the fitness function for a position vector.
    """
    
    def __init__(self, npart, c1, c2, c3, ndim, terms, stagn_limit,
                 x_bounds, v_bounds, w_bounds, t_bounds,
                 data_path, target_path, init_particles):
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

        terms : dict; key is a string indicating a certain term of the
                fitness function. value is a float indicating the actual
                weight of that term. Example: terms['accuracy'] = 0.3.
                sum(terms.values()) should equal 1.0.

        stagn_limit : int, > 0; number of times that gbest can stagnate
                      before shuffle_gbest() is called. Default 3.

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

        init_particles : boolean; if true, automatically initialize the
                         particles in the swarm, if false,
                         initialize_particles() must be called manually
                         by the wrapper program.

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
        self.terms = terms
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        self.t_bounds = t_bounds

        self.t = 0
        self.all_false_counter = 0
        # Using -1 as placeholders to throw errors if they are not updated.
        self.gbest = np.zeros(self.ndim) - 1
        self.gbest_counter = 0
        self.stagn_limit = stagn_limit
        self.gbinary = np.zeros(self.ndim) - 1
        self.g_fitness = -1.0
        self.g_score = (-1.0, -1.0, -1.0)
        self.abest = np.zeros(self.ndim) - 1
        self.abinary = np.zeros(self.ndim) - 1
        self.a_fitness = -1.0
        self.a_score = (-1.0, -1.0, -1.0)
        self.swarm = []
        self.p_fitness = np.zeros(self.npart) - 1
        self.p_scores = np.zeros((self.npart, 3)) - 1

        self.clf = KNeighborsClassifier()
        self.data = np.loadtxt(data_path, dtype=np.float64, delimiter=',')
        self.target = np.loadtxt(target_path, dtype=np.int8, delimiter=',')
        self.var_by_time = {
                            'num_features': np.zeros(t_bounds[1]).astype(int),
                            'g_fitness': np.zeros(t_bounds[1]),
                            'g_score' : np.zeros((t_bounds[1], 3)),
                            'a_fitness': np.zeros(t_bounds[1]),
                            'a_score': np.zeros((t_bounds[1], 3)),
                           }
        self.init_flag = False
        if init_particles:
            self.initialize_particles()

    def initialize_particles(self):
        """Initialize the particles that comprise the swarm.

        Actually fills the empty list, swarm, with COMB_Particle objects.
        Particle initialization is separated from the __init__ function
        because of a comparatively high computational cost. Wrapper scripts
        using the COMB_Swarm class may manually call this function if they set
        the init_particles parameter to False in __init__().

        NOTE: The inertia coefficient, w, for each particle is not initialized
              here because it is updated based on gbinary as the very first
              action in the actual run of the algorithm in execute_search(). It
              is only initialized as a placeholder, 0.0, in the particle
              function, __init__(). If you use these classes out of context, be
              sure to know that the w attribute may not be what you expect.

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
            f, f_score = self.eval_fitness(self.swarm[i].b, 10, self.terms)
            self.p_fitness[i] = f
            self.p_scores[i] = f_score
            if f > self.g_fitness:
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
        self.var_by_time['g_score'][0] = self.g_score
        self.var_by_time['a_fitness'][0] = self.a_fitness
        self.var_by_time['a_score'][0] = self.a_score
        self.init_flag = True

    def execute_search(self):
        """Execute a full run of the COMB-PSO Algorithm.

        Completes one full run of the COMB-PSO Algorithm using an internal
        classifier object and a swarm of COMB_Particle objects, returning
        a 1-Dimensional ndarray containing the best position found
        by the algorithm.

        NOTE: initialize_particles MUST be called before this method
              will run correctly.

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
        if not self.init_flag:
            print('ERROR: initialize_particles() must be called '
                + 'before execute_search()')
            return
        for i in range(1, self.t_bounds[1]):
            self.t = i
            counter = 0
            for p in self.swarm:
                p.update_inertia(self.gbinary)
                p.update_velocity(self.gbest, self.abest)
                p.update_position()
                p.update_binary_position()
                f, f_score = self.eval_fitness(p.b, 10, self.terms)
                if f > self.p_fitness[counter]:
                    p.pbest = p.x.copy()
                    p.pbinary = p.b.copy()
                    self.p_fitness[counter] = f
                    self.p_scores[counter] = f_score
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
                counter += 1
            if self.g_fitness > self.a_fitness:
                self.abest = self.gbest.copy()
                self.abinary = self.gbinary.copy()
                self.a_fitness = self.g_fitness
                self.a_score = self.g_score
            self.gbest_counter += 1
            if self.gbest_counter >= self.stagn_limit:
                self.shuffle_gbest()
            self.var_by_time['num_features'][i] = np.count_nonzero(self.abinary)
            self.var_by_time['g_fitness'][i] = self.g_fitness
            self.var_by_time['g_score'][i] = self.g_score
            self.var_by_time['a_fitness'][i] = self.a_fitness
            self.var_by_time['a_score'][i] = self.a_score
    
    def shuffle_gbest(self):
        """Randomize gbest after stagnation.

        Randomly reassigns gbest and saves the old gbest in abest, unless
        the new gbest is better than the old gbest. This method implements
        the "shuffle and archive" functionality described in the COMB-PSO
        algorithm. This method should only be used when gbest has been
        updated stagn_limit times and has not changed.

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
        self.g_fitness, self.g_score = self.eval_fitness(self.gbinary, 10,
                                                         self.terms)
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
        b : ndarray, size ndim; boolean ndarray that holds the
            binary version of x, a continuous position vector.

        Raises
        ------
        None
        """
        return (np.random.uniform(size=x.size) < expit(x))

    def test_classify(self, b, k):
        """Return a classification performance for a binary position vector.

        Runs a classification test with the clf attribute as a classifier and
        self.data and self.target as the feature data/correct classifications.
        Can be modified to classify any way, as long as it returns a metric
        of performance.

        Currently runs a k-fold cross validation and returns the confusion
        matrices for all k iterations. Accuracy, sensitivity, and specificity
        can all be calculated from this confusion matrix. Wrapper functions
        should calculate and average these values independently if they wish.

        Parameters
        ----------
        b : 1-Dimensional ndarray, size ndim; Holds a binary position vector
            where each value ϵ {0, 1} and represents respectively the
            exclusion or inclusion of that feature in the subset used for
            training.

        k : integer, k > 1; number of folds in the k-fold cross validation.
        
        Returns
        -------
        conf_matrices : ndarray, size = (k, 2, 2); array of confusion matrices
                        where each confusion matrix reports the scoring results
                        from one iteration of the k-fold cross validation.

        Raises
        ------
        None
        """
        conf_matrices = np.zeros((k, 2, 2))
        cnt = 0
        kf = StratifiedKFold(n_splits=k)
        for train_index, test_index in kf.split(self.data, self.target):
            self.clf.fit(self.data[train_index][:, b], self.target[train_index])
            y_pred = self.clf.predict(self.data[test_index][:, b])
            conf_matrices[cnt] = confusion_matrix(self.target[test_index],
                                                  y_pred)
            cnt += 1
        return conf_matrices
        
    def eval_fitness(self, b, k, terms):
        """Evaluate the fitness of the given position vector.

        Evaluates the fitness of the given binary position vector or feature
        subset by using that vector to subset the data to train a
        classifier and test predictive performance.

        The fitness function contains multiple terms, all optional (there must
        be at least one). The terms to be included are indicated by passing a
        dict of weight factors. The general form of the fitness function is:

            fitness(X, y, b, T, w) = w_0*T_0 + w_1*T_1 + ... + w_n*T_n

        where X is the feature data,
              y is the target label data,
              b is the subset of features, represented by a boolean vector
              T is the vector of terms included in the function
              w is the vector of weight factors; size = number of terms

        Available Term Options:
        
            accuracy: The mean accuracy from the k-fold cross validation
                      executed by test_classify().

            sensitivity: The mean sensitivity from the k-fold cross validation
                         executed by test_classify(). NOTE: Should only be used
                         in binary classification.

            specificity: The mean specificity from the k-fold cross validation
                         executed by test_classify(). NOTE: Should only be used
                         in binary classification.

            low_number: The number of features in the subset. Fewer is better.

            overfitting: The degree of overfitting. Less is better.

        Example terms parameter:

            terms = {
                      'accuracy': 0.4
                      'sensitivity': 0.4
                      'low_number': 0.2
                    }

        Parameters
        ----------
        b : 1-Dimensional ndarray, size ndim; Holds a binary position vector
            where each value ϵ {0, 1} and represents respectively the
            exclusion or inclusion of that feature in the subset used for
            training.

        k : integer, k > 1; number of folds in the k-fold cross validation.

        terms : dict; key must be a string that is a term name, see above
                documentation. value is a float in interval (0.0, 1.0]. Values
                must sum to 1.0.

        Returns
        -------
        fitness : float; value returned by the fitness function.

        scores : float tuple, size 3; accuracy, sensitivity, and specificity

        Raises
        ------
        None
        """
        if np.count_nonzero(b) == 0:
            self.all_false_counter += 1
            return 0.0, (0.0, 0.0, 0.0)
        if not terms:
            print('Error - COMB_Swarm.eval_fitness: '
                + 'You must include at least one term.')
        # Weights should sum to 1, but floating point arithmetic is tricky
        if (sum(terms.values()) > 1.0001
          or sum(terms.values()) < 0.9999):
            print('Error - COMB_Swarm.eval_fitness: '
                + 'Weight factors must sum to 1.0')
            return
        valid_terms = {'accuracy', 'sensitivity', 'specificity',
                       'low_number', 'overfitting'}
        for key, value in terms.items():
            if key not in valid_terms:
                print('Error - COMB_Swarm.eval_fitness: '
                    + '{} is not a valid term'.format(k))
                return
            if value <= 0.0 or value > 1.0:
                print('Error - COMB_Swarm.eval_fitness: '
                    + '{} is not a valid weight factor. '.format(v)
                    + 'Weights must be in the interval (0.0, 1.0]')
                return
        cm = self.test_classify(b, k)
        accs = np.zeros(k)
        sens = np.zeros(k)
        spec = np.zeros(k)
        cnt = 0
        for m in cm:
            accs[cnt] = (m[0,0]+m[1,1])/m.sum()
            sens[cnt] = m[1,1]/(m[1,1]+m[1,0])
            spec[cnt] = m[0,0]/(m[0,0]+m[0,1])
            cnt += 1
        accuracy = accs.mean()
        sensitivity = sens.mean()
        specificity = spec.mean()
        fitness = 0.0
        if 'accuracy' in terms.keys():
            fitness += accuracy * terms['accuracy']
        if 'sensitivity' in terms.keys():
            fitness += sensitivity * terms['sensitivity']
        if 'specificity' in terms.keys():
            fitness += specificity * terms['specificity']
        if 'low_number' in terms.keys():
            fitness += (((self.ndim-np.count_nonzero(b)) / self.ndim)
                     * terms['low_number'])
        if 'overfitting' in terms.keys():
            # COST
            kf = StratifiedKFold(n_splits=5)
            scores = []
            for train_index, test_index in kf.split(self.data, self.target):
                self.clf.fit(self.data[train_index][:, b], self.target[train_index])
                y_pred = self.clf.predict(self.data[test_index][:, b])
                scores.append(mean_squared_error(self.target[test_index],
                                                 y_pred))
            scores = np.array(scores)
            # Binary {0, 1} classification so the MSE will never be outside
            # the interval [0.0, 1^2] = [0.0, 1.0] which is what I need
            # for this fitness function.
            fitness -= scores.mean()
        # Fixes bug where cost terms outweigh other terms and returns a negative
        # fitness, causing failure to update gbest, gbinary, abest, and abinary
        # from their placeholder values.
        if fitness < 0:
            fitness = 0.0
        return fitness, (accuracy, sensitivity, specificity)
