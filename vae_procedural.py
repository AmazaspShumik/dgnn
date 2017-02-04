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
    posterior of latent variable is also approximated with normal distribution 
    '''
    def __init__(self, network_architecture, batch_size = 100, activation = None,
                 learning_rate = 1e-3):
        # encoder architecture & latent var & decoder 
        self.d_in,self.ed_h1,self.ed_h2,self.d_lat,self.dd_h1,self.dd_h2 = network_architecture
        
        # activation function
        if activation == None:
            self.activation = tf.nn.relu
            
        self.learning_rate = learning_rate            
        self.x    = tf.placeholder(shape=[None,self.d_in],dtype=tf.float32)
        init      = tf.initialize_all_tables() 
        self.sess = tf.Session()
        self.sess.run(init)
    
    
    def _create_network(self):
        
        # -------------  Encoder network
        
        # encoder hidden layer 1
        ew_h1, ew_b1 = create_layer(self.d_in, self.d_h1)
        eh1          = self.activation(tf.matmul(self.x,ew_h1) + ew_b1)
        
        # encoder hidden layer 2
        ew_h2,eb_h1 = create_layer(self.ed_h1,self.ed_h2)
        eh2         = self.activation(tf.matmul(eh1,ew_h1) + eb_h1)
        
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
        dh2          = tf.matmul(dh1,dw_h2) + db_h2
        self._reconstruction(dh2)

 
    def _reconstruction(self,dh2):
        ''' Reconstruct original input using output of hidden layer'''
        raise NotImplementedError()
            
        
    def _kl(self):
        ''' 
        Compute KL divergence between standard normal and approximating distribution.
        Here we assume approximatiing distribution is also normal distribution
        with diagonal covariance.
        '''
        self.kl = -0.5 * tf.reduce_sum( 1 + self.z_log_std - tf.square(self.z_mean) - tf.exp(self.log_std))
        
        
    def _expected_likelihood(self):
        ''' Expectation of data log ikelihood with respect to approx. distribution'''
        raise NotImplementedError()
        
    
    def _create_loss(self):
        self.loss = tf.add( self._expected_likelihood(), self._kl)
        
        
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize()
    
    
    def partial_fit(self,X):
        '''
        Fit batch of data
        
        Parameters
        ----------
        
        '''
        self.sess.run([self.optimizer], feed_dict={self.x:X})
        
        
#    def generate_data(selfl)
        


class GaussianVAE(BaseVAE):
    pass


class BernoulliVAE(BaseVAE):

    def _reconstruction(self, dh2):
        ''' Reconstructs original data '''
        w_out,b_out  = create_layer(self.dd_h2,self.d_in) 
        self.reconstruction = tf.nn.sigmoid(tf.matmul(dh2,w_out) + b_out)


class MultinoulliVAE(BaseVAE):
    pass
    


if __name__ == '__main__':
    
    
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
        
    # load and preprocess data
    
    D_in, D_h1, D_h2, D_lat, D_h3, D_h4   = 784,200,101,10,101,200
    
    batch_size   = 100
    
    
    #==========================  Encoder Network ================================
    
    x = tf.placeholder(tf.float32,[None,D_in])
    
    # first hidden layer    
    W_fc1 = tf.Variable(tf.truncated_normal([D_in,D_h1],stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[D_h1]))
    h_1   = tf.nn.relu( tf.matmul(x,W_fc1) + b_fc1 ) 
    
    # second hidden layer 
    W_fc2 = tf.Variable(tf.truncated_normal([D_h1,D_h2],stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[D_h2]))
    h_2   = tf.nn.relu( tf.matmul(h_1, W_fc2) + b_fc2 )
    
    # latent variable layer
    W_z_mean   = tf.Variable(tf.truncated_normal([D_h2,D_lat],stddev=0.1))
    b_z_mean   = tf.Variable(tf.constant(0.1, shape=[D_lat]))
    W_z_logstd = tf.Variable(tf.truncated_normal([D_h2,D_lat], stddev=0.1))
    b_z_logstd = tf.Variable(tf.constant(0.1, shape=[D_lat]))
    
    # model mean & log of standard deviation of latent variable
    z_mean     = tf.matmul(h_2,W_z_mean) + b_z_mean
    z_log_std  = tf.matmul(h_2,W_z_logstd) + b_z_logstd 
    
    #========================= Decoder Network ==================================
    
    eps = tf.random_normal((batch_size, D_lat), 0, 1, dtype=tf.float32) # Adding a random number
    z = z_mean + tf.mul(tf.sqrt(tf.exp(z_log_std)), eps)  # The sampled z
    
    W_fc3 = tf.Variable(tf.truncated_normal([D_lat, D_h3], stddev=0.1))
    b_fc3 = tf.Variable(tf.constant(0.1,shape=[D_h3]))
    h_3   = tf.nn.relu( tf.matmul(z,W_fc3) + b_fc3 )

    W_fc4 = tf.Variable( tf.truncated_normal([D_h3,D_h4], stddev=0.1) )
    b_fc4 = tf.Variable( tf.constant(0.1,shape=[D_h4]))  
    h_4   = tf.nn.relu( tf.matmul(h_3,W_fc4) + b_fc4)

    W_out = tf.Variable( tf.truncated_normal([D_h4,D_in], stddev=0.1) )
    b_out = tf.Variable( tf.constant(0.1,shape=[D_in]))
    logit = tf.matmul(h_4,W_out) + b_out
    y_hat = tf.nn.sigmoid(logit)
    
    #======================= Loss ===============================================
    
    # KL divergence 
    kl   = -0.5*tf.reduce_sum( 1 - tf.square(z_mean) - tf.exp(z_log_std) + z_log_std,1)
    
    # Expectattion of likelihood
    like = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logit,x),1)
    
    # Loss function 
    loss = tf.reduce_mean(kl + like)
    
    
    #====================== Optimizing===========================================
    
    optimizer =  tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)
    
    
    #==================== Run ==========================================
    
    ver = tf.__version__
    print("Tensor Flow version {}".format(ver))
    # Implementation of Variational Autoencoder
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
    print("Number of samples {} Shape of y[{}] Shape of X[{}]"
          .format(n_samples, mnist.train.labels.shape, mnist.train.images.shape))
    plt.imshow(np.reshape(-mnist.train.images[4242], (28, 28)), interpolation='none',cmap=plt.get_cmap('gray'))

    runs = 600 #Set to 0, for no training
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        batch_xs, _ = mnist.train.next_batch(batch_size)
        batch_xs[batch_xs>0.5] = 1
        batch_xs[batch_xs<0.5] = 0
        print(batch_xs.shape)
        dd = sess.run([loss], feed_dict={x: batch_xs})
        print('Test run after starting {}'.format(dd))
    
        for epoch in range(runs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                _,d = sess.run((optimizer, loss), feed_dict={x: batch_xs})
                avg_cost += d / n_samples * batch_size
    
            # Display logs per epoch step
            if epoch % 10 == 0:
                #save_path = saver.save(sess, "model/model.ckpt") #Saves the weights (not the graph)
                #print("Model saved in file: {}".format(save_path))
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
                
                # plot original & reconstruction
                x_sample = mnist.test.next_batch(batch_size)[0]
                x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val = sess.run((y_hat,z, z_mean, z_log_std), feed_dict={x: x_sample})
                
                plt.figure(figsize=(8, 12))
                for i in range(1):
                    plt.subplot(5, 3, 3*i + 1)
                    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
                    plt.title("Test input")
                    
#                    #plt.colorbar()
#                    plt.subplot(5, 3, 3*i + 2)
#                    plt.scatter(z_vals[:,0],z_vals[:,1], c='gray', alpha=0.5)
#                    plt.scatter(z_mean_val[i,0],z_mean_val[i,1], c='green', s=64, alpha=0.5)
#                    plt.scatter(z_vals[i,0],z_vals[i,1], c='blue', s=16, alpha=0.5)
#                   
#                    plt.xlim((-3,3))
#                    plt.ylim((-3,3))
#                    plt.title("Latent Space")
                    
                    plt.subplot(5, 3, 3*i + 3)
                    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
                    plt.title("Reconstruction")
                    #plt.colorbar()
                plt.tight_layout()
                
                