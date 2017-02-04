# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_weights(d1,d2, std=0.1):
    ''' Create weight matrix of shape (d1, d2) '''
    return tf.Variable(tf.truncated_normal([d1,d2],stddev = std))
    
    
def create_bias(d2,const=0.1):
    ''' Create bias term of shape (d2,1) '''
    return tf.Variable(tf.constant(const,tf.float32,shape=[d2]))
    
    
def create_layer(d1, d2, std = 0.1, const=0.1):
    weights = create_weights(d1,d2,std)
    bias    = create_bias(d2,const)
    return weights, bias
    


class BaseVAE(object):
    '''
    Base class for Variational Autoencoders with different likelihood functions.
    
    It is assumed that latent variable is distributed as standard normal and 
    posterior of latent variable is also approximated with normal distribution. 
    '''
    def __init__(self, network_architecture, batch_size,activation,learning_rate):
                     
        # encoder architecture & latent var & decoder 
        self.d_in,self.ed_h1,self.ed_h2,self.d_lat,self.dd_h1,self.dd_h2 = network_architecture
        
        # activation function
        if activation == None:
            self.activation = tf.nn.relu
            
        self.learning_rate = learning_rate            
        self.batch_size = batch_size
        self.x = tf.placeholder(shape=[None,self.d_in],dtype=tf.float32)
        self._create_network()
        self._create_loss()
        self._create_optimizer()

    
    
    def _create_network(self):
        
        # -------------  Encoder network
        
        # encoder hidden layer 1
        ew_h1, ew_b1 = create_layer(self.d_in, self.ed_h1)
        eh1          = self.activation(tf.matmul(self.x,ew_h1) + ew_b1)
        
        # encoder hidden layer 2
        ew_h2,eb_h2 = create_layer(self.ed_h1,self.ed_h2)
        eh2         = self.activation(tf.matmul(eh1,ew_h2) + eb_h2)
        
        # ------------  Latent variable 
        
        # model mean of latent variable (mean of q(z|x) )
        w_z_mu, b_z_mu = create_layer(self.ed_h2,self.d_lat)
        self.z_mu      = tf.matmul(eh2,w_z_mu) + b_z_mu
        
        # model diagonal covariance ( covariance matrix of q(z|x) )
        w_z_logstd, b_z_logstd  = create_layer(self.ed_h2,self.d_lat)
        self.z_log_std          = tf.matmul(eh2,w_z_logstd) + b_z_logstd
        
        eps = tf.random_normal([self.batch_size,self.d_lat])
        z   = self.z_mu + tf.mul(tf.sqrt(tf.exp(self.z_log_std)),eps)
        
        # ------------  Decoder network
        
        # decoder hidden layer 1
        dw_h1, db_h1 = create_layer(self.d_lat,self.dd_h1)
        dh1          = tf.nn.elu(tf.matmul(z,dw_h1) + db_h1)
        
        # decoder hidden layer 2
        dw_h2, db_h2 = create_layer(self.dd_h1,self.dd_h2)
        self.dh2          = tf.matmul(dh1,dw_h2) + db_h2
        self._reconstruction()

 
    def _reconstruction(self,dh2):
        ''' Reconstruct original input using output of hidden layer'''
        raise NotImplementedError()
            
        
    def _kl(self):
        ''' 
        Compute KL divergence between standard normal and approximating distribution.
        Here we assume approximatiing distribution is also normal distribution
        with diagonal covariance.
        '''
        self.kl = -0.5 * tf.reduce_sum( 1 + self.z_log_std - tf.square(self.z_mu) - tf.exp(self.z_log_std),1)
        
        
    def _neg_ll(self):
        ''' 
        Expectation of negative log-likelihood with respect to apprioximating
        distribution (q(z|x))
        '''
        raise NotImplementedError()
        
    
    def _create_loss(self):
        ''' Computes loss function for VAE '''
        self._neg_ll() # expected log likelihood (expectation is taken under approx. dist)
        self._kl()
        self.loss = tf.reduce_mean( self.neg_ll + self.kl )
        
        
    def _create_optimizer(self):
        ''' Creates '''
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
    
    
    def partial_fit(self,X):
        self.sess.run([self.optimizer], feed_dict={self.x: X})
        
        
    def reconstruct(self,X):
        x_hat = self.sess.run([self.reconstruction], feed_dict = {self.x: X})
        return x_hat
        
    


class GaussianVAE(BaseVAE):
    ''' 
    Variational Autoencoder with Gaussain Likelihood.
    '''    
    
    def __init__(self,network_architecture = [784,400,200,10,200,400], batch_size=100,
                 activation = None,learning_rate = 1e-3):
                     
        super(GaussianVAE,self).__init__(network_architecture, batch_size, activation, learning_rate)
        init      = tf.initialize_all_variables() 
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _reconstruction(self):
        wmu, bmu            = create_layer(self.dd_h2,self.d_in)
        wsd, bsd            = create_layer(self.dd_h2, self.d_in)
        self.x_log_std      = tf.matmul(self.dh2,wsd) + bsd
        self.reconstruction = tf.matmul(self.dh2,wmu) + bmu # mean of gaussian
    
    def _neg_ll(self):
        neg_ll = self.x_log_std + 0.5*tf.square(self.x - self.reconstruction) / tf.exp(self.x_log_std)
        self.neg_ll = tf.reduce_sum( neg_ll, 1)


class BernoulliVAE(BaseVAE):
    ''' 
    Variational Autoencoder with Bernoulli Likelihood. 
    '''

    def _reconstruction(self, dh2):
        ''' Reconstructs original data '''
        w_out,b_out  = create_layer(self.dd_h2,self.d_in) 
        self.reconstruction = tf.nn.sigmoid(tf.matmul(self.dh2,w_out) + b_out)
        
        
    def _neg_ll(self):
        self.neg_ll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.reconstruction,self.x),1)
        


if __name__=="__main__":
    
    def next_batch(batch_size, non_crossing=True):
        z_true = np.random.uniform(0,1,batch_size)
        r = np.power(z_true, 0.5)
        phi = 0.25 * np.pi * z_true
        x1 = r*np.cos(phi)
        x2 = r*np.sin(phi)
    
        # Sampling form a Gaussian
        x1 = np.random.normal(x1, 0.10* np.power(z_true,2), batch_size)
        x2 = np.random.normal(x2, 0.10* np.power(z_true,2), batch_size)
            
        # Bringing data in the right form
        X = np.transpose(np.reshape((x1,x2), (2, batch_size)))
        X = np.asarray(X, dtype='float32')
        return X
        
    net          =  [2,5,5,1,5,5]
    batch_size   =  100
    # initialise Gaussian VAE
    gvae = GaussianVAE(network_architecture = net, batch_size = batch_size,
                       activation = tf.nn.softplus )  
    
    print("Successfully initialised")
    test_batch = next_batch(100)      
        
    for epoch in range(600):
        x_batch = next_batch(100)
        gvae.partial_fit(x_batch)
        print("Iteration {0}".format(epoch))
        
    
    test_x_hat = gvae.reconstruct(test_batch)
        
    