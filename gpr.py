import numpy as np
import pyopencl as cl

class GPR():
    """
    The Radial Basis function kernel:

    k(x_i,x_j) = z_0^2 * exp(-0.5 |x_i-x_j|^2 / z_1^2 ) + delta_i^j z_2^2

    is determined by three kernel hyper parameters (z_0,z_1,z_2), which are 
    considered as latent variables. The posterior distribution of latent 
    variables

    p(z|X,t) is given by the mean-field approximation:
    
    p(z|X,t) ~ \prod_i q(z_i | mu_i, var_i) = N(z_i | mu_i , var_i),

    where each variational distribution is a normal determined by 2 distribution
    parameters.
    """

    def __init__(self,Nsample=100,maxiter=100,lr=1e-3,decay_rate=1e-2):
        _ = [getattr(self,"set_{}".format(_k))({"Nsample":Nsample,\
                "maxiter":maxiter,"lr":lr,"decayrate":decay_rate}[_k]) \
                for _k in ["Nsample","maxiter","lr","decayrate"]]

        # precision
        self.dtype = np.float64

    def set_decayrate(self,decay_rate):
        self.decay_rate = decay_rate

    def set_lr(self,lr):
        """ 
        Learning rate
        
        lr_t = lr_0 / (1+t)
        """
        self.lr0 = lr
        self.lr = self.lr0

    def set_Nsample(self,Nsample):
        """
        Set number of stochastic samples to estimae log likelihood derivative
        """
        if Nsample<0:
            raise Exception("Must be a positive number")
        
        self.Nsample = int(Nsample)

    def set_maxiter(self,maxiter):
        if maxiter<0:
            raise Exception("Must be a positive number")

        self.maxiter = int(maxiter)
 
    def set_X(self,X):
        """
        Set observed variables X and reshape to rank 2 matrix if necessary.
        
        Arguments
        ---------
        X : np.ndarray
            - The Npoints observed variables of shape [Npoints][dimX] and 
            dimension dimX.
        """
        if len(X.shape)==1:
            X = np.reshape(X,(-1,1))
        
        self.X = X
    
        # store order of covariance matrix
        self.N = self.X.shape[0]

    def set_t(self,t):
        """
        Set the observed labels and check is a 1-d array since multivariate 
        regression is not currently supported.

        Arguments
        ---------
        t : np.ndarray, shape=[N,]
            - The observed labels.
        """
        if not len(t.shape)==1:
            raise Exception("Multivariate regression not currently supported")

        self.t = t

    def fit(self,X,t):
        # sanity check supplied data
        _ = [getattr(self,"set_{}".format(_k))({"X":X,"t":t}[_k]) for _k in ["X","t"]] 

        # initial values for latent variable dist. params
        self._init_latent_variables()

        # no need to recalculate l2-norm
        self._init_redundancy_arrays()

        self.loss = np.zeros(self.maxiter)

        for self.iter in range(self.maxiter):
            # stochastic gradient descent
            self.update_distribution_params()

            # approximate Lower BOund
            self.loss[self.iter] = self.calculate_LBO()

    def _init_redundancy_arrays(self):
        """
        Store redundant calculations
        """
        self.l2_norm = np.asarray([np.sum(np.square(np.tile(_x,(self.N,1))-self.X),axis=1) \
                for _x in self.X])

    def update_distribution_params(self):
        """
        x_t+1 = x_t - d objective / d x_t * learning_rate
        """  
        # (mu1,sig1,mu2,..,sig3)      
        jac = self.calculate_LBO_jacobian()
        
        for ii in range(3):
            self.latent_mu[ii] = self.latent_mu[ii] + jac[ii*2]*self.lr
            self.latent_sig[ii] = self.enforce_positive(self.latent_sig[ii] + jac[ii*2+1]*self.lr)
            
        # update learning rate
        self.lr = self.lr0/(1.0+self.decay_rate*self.iter)

    def enforce_positive(self,value):
        """
        if value<=0, return 1e-9
        """
        if value<0.0:
            res = 1e-9
        else:
            res = value

        return res

    def _init_latent_variables(self):
        """
        Each of the three RBF hyper parameters (pre-factor, unit scale, noise 
        level) are given normal prior and factored posterior distributions.
        """        
        # normal mean
        self.latent_mu = np.asarray([1.,1.,1.],dtype=np.float64)
        
        # normal standard deviation
        self.latent_sig = np.asarray([1e-1,1e-1,1e-1])

        # z_i = mu_i + sig_i*epsilon : epsolon ~ N(0,1)
        self.hyperparam = [np.random.normal(loc=self.latent_mu[ii],scale=self.latent_sig[ii],size=1)\
                for ii in range(3)]

    def calculate_LBO_jacobian(self):
        """
        Calculate the derivative of \int q(z) dz [ ln p(x,z) - ln q(z) ]

        with respect to the variational distribution parameters.
        """
        # (mu1,sig1,mu2,...,sig3)
        return self.likelihood_derivative() + self.variational_KL_derivative() 

    def likelihood_derivative(self):
        """ 
        d ln p(x|z_k) / d lambda_k
        """

        stochastic_samples = np.zeros((self.Nsample,3,2))
   
        for ss in range(self.Nsample):

            # draw sample
            epsilon = np.random.normal(loc=0,scale=1,size=3)

            # update latent variables, covariance and inverse covariance
            self.update_all_sampled_dependent_variables(epsilon=epsilon)

            # [Nsamples][3][mu,sigma]
            stochastic_samples[ss] = [self.log_likelihood_derivative(latent_idx=ii)\
                    *self.latent_variable_derivative(latent_idx=ii,epsilon=epsilon[ii])\
                    for ii in range(3)]

        # [mu_1,std_1,mu_2,...,std_3]
        return np.mean(np.asarray(stochastic_samples),axis=0).flatten()
        

    def update_all_sampled_dependent_variables(self,epsilon):
        # z_i = mu_i + sigma_i * epsilon_i
        self.update_hyperparams(epsilon=epsilon)

        # hyperparameters MUST be updated before this method call
        self.update_covariance_matrix()

    def update_covariance_matrix(self):
        # exp. kernel with diagonal correction 
        self.covariance = self.hyperparam[0]**2 \
                * np.exp(-0.5*self.l2_norm/(self.hyperparam[1]**2)) \
                + np.eye(self.N)*self.hyperparam[2]**2 + np.eye(self.N)*1e-2

        # psuedo-inv. if necessary
        self.covariance_inverse = np.linalg.inv(self.covariance)

    def update_hyperparams(self,epsilon):
        """
        z_i = mu_i + sigma_i * epsilon_i
        """
        self.hyperparam = [self.latent_mu[ii]+self.latent_sig[ii]*epsilon[ii] \
                for ii in range(3)]

      

    def variational_KL(self,latent_idx):
        """
        - KL(q||p) = \int q(z_i | mu_i,sigma_i) ln p(z_i)/q(z_i |mu_i,sigma_i) dz_i

        In the restricted case where:

        1. p(z_i) = N(0,1)
        2. q(z_i) = N(mu_i,\sigma_i^2),

        then 

        - KL(q||p) = 0.5 * (1 + 2*ln(sigma_i) - sigma_i^2 - mu_i^2)
        """
        res = 0.5*(1.0 + 2.0*np.log(self.latent_sig[latent_idx]) \
                - self.latent_sig[latent_idx]**2 - self.latent_mu[latent_idx]**2)

        return res

    
    def variational_KL_derivative(self):
        """
        Latent variable distribution hyper parameters are fixed during 
        a single jacobian calculation. 
        """
        return np.asarray([self.variational_KL_derivative__single(ii) \
                for ii in range(3)]).flatten()

    def variational_KL_derivative__single(self,latent_idx):
        """
        derivative of -KL(q||p) wrt. mu_i,sigma_i
        
        d -KL(q||p) / d mu_i = -mu_i
        d -KL(q||p) / d sigma_i = 1/sigma_i - sigma_i
        """
        dmu = -self.latent_mu[latent_idx]
        dsig = 1.0/self.latent_sig[latent_idx] - self.latent_sig[latent_idx]
        
        return [dmu,dsig]
        

    def latent_variable_derivative(self,latent_idx,epsilon):
        """
        Derivate of 

        z_i = mu_i + sigma_i * epsilon

        with respect to variational distribution parameters (mu_i,sigma_i).
        """
        return np.asarray([1.0,epsilon],dtype=np.float64)

    def calculate_LBO(self):
        """
        LBO = E_q(z)[ ln p(t|z) ] + sum_i E_q(z_i) [ ln p(z_i) - ln q(z_i) ]
        """

        # sum_i E_q(z_i) [ ln p(z_i) - ln q(z_i) ]
        KL_contribution = np.sum([self.variational_KL(latent_idx=ii) \
                for ii in range(3)])

        return self.log_likelihood_expectation() + KL_contribution

    def log_likelihood_expectation(self):
        """
        Expcted value of log likelihood sampled over self.Nsample samples
        of latent variables in the kernel.
        """

        reduction = np.zeros(self.Nsample)

        for ss in range(self.Nsample):
            epsilon = np.random.normal(loc=0,scale=1,size=3)

            # upate covariance matrices and inverse        
            self.update_all_sampled_dependent_variables(epsilon=epsilon)

            reduction[ss] = self.log_likelihood()
            
        return np.mean(reduction)

    def log_likelihood(self):
        """
        Log likelihood ln p(t|z)
        """
        # ln det(C) can overflow for very small or large det
        [sign,lndet] = np.linalg.slogdet(self.covariance)

        # -0.5 ln det(C)
        det_term = -0.5*sign*lndet

        # -0.5 t^T C^-1 t
        label_term = - 0.5*np.dot(self.t.T,np.dot(self.covariance_inverse,self.t))

        return det_term + label_term - 0.5*self.N*np.log(2.0*np.pi)

    def likelihood(self):
        """
        p(t|z) = N(t|0,C)
        """
        # normalizing constant
        prefactor = np.power(np.pi*2.0,-self.N*0.5)*np.power(np.linalg.det(self.covariance),-0.5)

        exp_term = np.exp(- 0.5*np.dot(self.t.T,np.dot(self.covariance_inverse,self.t)) )

        return prefactor*exp_term

    def log_likelihood_derivative(self,latent_idx):
        """
        Derivative of log-likelihood with respect to latent variable index.

        Arguments
        ---------
        latent_idx : int
            - 0 offset index identify latent variable of choice

        +-------+---------------------------------------+
        | index | latent variable                       |
        +-------+---------------------------------------+
        | 0     | exponential pre-factor                |
        +-------+---------------------------------------+
        | 1     | unit scale (variance in exponential)  |
        +-------+---------------------------------------+
        | 2     | diagonal correction                   |
        +-------+---------------------------------------+
        """
    
        # cov matrix derivative wrt. latent variable
        cov_jac = self.covariance_matrix_derivative(latent_idx=latent_idx)

        # -0.5 Tr( C^-1 dC/dlv )
        trace_term = -0.5*np.einsum("ij,ji->",self.covariance_inverse,cov_jac)

        # 0.5 t^T C^-1 dC/dlv C^-1 t
        label_term = 0.5*np.dot(self.t.T,np.dot(self.covariance_inverse,np.dot(cov_jac,\
                np.dot(self.covariance_inverse,self.t))))
       
        return trace_term + label_term

    def covariance_matrix_derivative(self,latent_idx): 
        """
        Derivative of covariance matrix with respect to latent variable index.

        Arguments
        ---------
        latent_idx : int
            - 0 offset index identify latent variable of choice

        +-------+---------------------------------------+
        | index | latent variable                       |
        +-------+---------------------------------------+
        | 0     | exponential pre-factor                |
        +-------+---------------------------------------+
        | 1     | unit scale (variance in exponential)  |
        +-------+---------------------------------------+
        | 2     | diagonal correction                   |
        +-------+---------------------------------------+
        """
        if latent_idx == 0:
            res = self.covariance_matrix_derivative__prefactor()
        elif latent_idx == 1:
            # variance in exponential
            res = self.covariance_matrix_derivative__scale()
        elif latent_idx == 2:
            # diagonal correction derivative
            res = self.covariance_matrix_derivative__diagonal()
        else: raise Exception("Implementation Error")

        return res

    def covariance_matrix_derivative__scale(self):
        """
        d K_ij / d z_1 = d z_0^2 * exp(-0.5 |x_i-x_j|^2 / z_1^2) / d z_1

                       = -|x_i-x_j|^2 z_2^{-3} K_ij
        """
        res = (self.hyperparam[0]**2)*(self.l2_norm/self.hyperparam[1]**3)\
                *np.exp(-0.5*self.l2_norm/self.hyperparam[1]**2)
        
        return res

    def covariance_matrix_derivative__prefactor(self):
        res = 2.0/self.hyperparam[0]*(self.covariance-self.hyperparam[2]**2 *np.eye(self.N))
        return res

    def covariance_matrix_derivative__diagonal(self):
        return 2.0*self.hyperparam[2]*np.eye(self.N)

    def predict(self,X):
        """
        Use mode of latent variable distribution parameters to make point
        estimate.
        """
        # update covariance, inv covartiance to mode of lv. distributions
        self.update_all_sampled_dependent_variables(epsilon=np.zeros(3))

        # [Ntest,Ntrain] kernels
        K = np.asarray([np.sum(np.square(np.tile(_x,(self.N,1))-self.X),axis=1) \
                for _x in X])
        K = self.hyperparam[0]**2 * np.exp(-0.5*K/self.hyperparam[1]**2) 

        # [Ntest,] array of expected values
        mean = np.einsum("ab,bc,c->a",K,self.covariance_inverse,self.t) 

        # k(x,x)
        k0 = self.hyperparam[0]**2 + self.hyperparam[2]**2

        std = np.sqrt(k0 - np.asarray([np.dot(_k.T,\
                np.dot(self.covariance_inverse,_k)) for _k in K]))

        return mean,std
