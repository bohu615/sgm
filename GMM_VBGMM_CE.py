##### DATASET

import numpy as np
import matplotlib.pyplot as plt

def construct_contour_gauss(centers, weights, learned_variance, interp=100):
    QMI_TRUE_LIST = []
    interp = 100
    delta = (max-min)/interp

    x_axis = np.linspace(min, max, interp)
    y_axis = np.linspace(min, max, interp)
    xv, yv = np.meshgrid(x_axis,y_axis)

    input = np.array((xv, yv)).reshape(2, -1).T

    gaussian_plot_joint_ = []
    gaussian_plot_split_x_ = []
    gaussian_plot_split_y_ = []

    #centers = np.concatenate((center_x, center_y), 1)
    #difference = input.reshape(-1, 2) - centers.reshape(-1, 1, 2)

    for i in range(0, centers.shape[0]):
        gaussian_plot_joint_.append(weights[i]*gaussian_nd(input - centers[i], 0, learned_variance[i]))
    gaussian_plot_joint = np.mean(np.array(gaussian_plot_joint_), 0)*delta*delta
    
    return gaussian_plot_joint.reshape(interp, interp)

def gaussian_nd_numpy(MEAN, VARIANCE):
    bs = VARIANCE.shape[0]
    dim = VARIANCE.shape[1]

    det = np.linalg.det(VARIANCE)
    inv = np.linalg.pinv(VARIANCE)
            
    product = np.sum((MEAN.reshape(-1, 1, dim)@inv).reshape(-1, dim)*MEAN.reshape(-1, dim), 1)
    
    return ((2*np.pi)**(-dim/2))*det**(-1/2)*np.exp(-(1/2)*product)

def compute_TRUE_ENTROPY(MEAN_matrix, COV_matrix, weights_matrix):
    K = MEAN_matrix.shape[0]
    dim = MEAN_matrix.shape[1]
    
    MEAN_DIFF = MEAN_matrix.reshape(K, 1, dim) - MEAN_matrix.reshape(1, K, dim)
    COV_DIFF = COV_matrix.reshape(K, 1, dim, dim) + COV_matrix.reshape(1, K, dim, dim)
    WEIGHT_DIFF = weights_matrix.reshape(K, 1)*weights_matrix.reshape(1, K)
    
    return np.sqrt(np.sum(WEIGHT_DIFF.reshape(-1)*gaussian_nd_numpy(MEAN_DIFF.reshape(-1, dim), COV_DIFF.reshape(-1, dim, dim))))

def gaussian_nd(input, m, sigma):
    k = sigma.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.pinv(sigma)
    
    return ((2*np.pi)**(-k/2))*det**(-1/2)*np.exp(-(1/2)*np.sum((input-m)@inv*(input-m), 1))

#### CONSTRUCT A MORE RESONABLE MKM

min = 0
max = 1

def makediag3d(a):
    a = np.asarray(a)
    depth, size = a.shape
    x = np.zeros((depth,size,size))
    for i in range(depth):
        x[i].flat[slice(0,None,1+size)] = a[i]
    return x

def create_Gaussian_mixture_MC():
    np.random.seed(4)
    
    center_x = np.linspace(0.2, 0.8, num)
    center_y = np.linspace(0.2, 0.8, num)
    
    np.random.shuffle(center_x)
    np.random.shuffle(center_y)

    xv, yv = np.meshgrid(center_x, center_y)

    MEAN_MATRIX = np.array((xv, yv)).reshape(2, -1).T
    COV = makediag3d(np.random.uniform(0.0001, 0.0004, MEAN_MATRIX.shape[0]*2).reshape(num*num, 2))        
    weights = np.ones((num, num))*(0.5/(num-1))+np.eye(num)*(0.5-0.5/(num-1))
    
    return MEAN_MATRIX, COV, weights, center_x, center_y

def gaussian_1d(input, m, sigma):
    det = sigma
    inv = 1/sigma
        
    input = input.reshape(-1, 1)
    m = m.reshape(1, -1)

    return ((2*np.pi)**(-1/2))*det**(-1/2)*np.exp(-(1/2)*((input-m)**2*inv))

def generate_gauss_samples_various(MEAN, COV_matrix, samples_per_class=3000000):
    
    num_class = MEAN.shape[0]
    component_ = []
    for i in range(0, num_class):
        COV = COV_matrix[i]
        samples = np.random.normal(MEAN[i], np.sqrt(COV), int(samples_per_class))
        component_.append(samples)
    component_ = np.array(component_)
    
    return component_
    
MEAN_SAVE = []
COV_SAVE = []
WEIGHT_SAVE = []
center_x_save = []
center_y_save = []

for num in [100]:
    
    np.random.seed(4)

    MEAN_matrix, COV_matrix, weights_matrix, center_x, center_y = create_Gaussian_mixture_MC()
    COV_matrix = COV_matrix*3
    weights_matrix = weights_matrix.reshape(-1)

    pdf = construct_contour_gauss(MEAN_matrix, weights_matrix.reshape(-1), COV_matrix)
    normalize = pdf/np.sum(pdf,1).reshape(-1, 1)

    # plt.imshow(pdf, origin='lower', extent=[min, max, min, max])
    # plt.show()

#     plt.contour(normalize, origin='lower', extent=[min, max, min, max])
#     plt.show()
    
    MEAN_SAVE.append(MEAN_matrix)
    COV_SAVE.append(COV_matrix)
    WEIGHT_SAVE.append(weights_matrix)
    center_x_save.append(center_x)
    center_y_save.append(center_y)
    
plt.title('Conditional pdf demo')
plt.contour(normalize, origin='lower', extent=[min, max, min, max], levels=30)
plt.show()

### GENERATING DATASET

x_list = []
y_list = []

next_chosen = []

print('Generating dataset...')

for j in range(0, 1):

  MEAN_matrix = MEAN_SAVE[0]
  COV_matrix = COV_SAVE[0]
  weights_matrix = WEIGHT_SAVE[0]

  np.random.seed(j)
    
  ### CHANGE TRIALS:
  inter = 10

  x_0 = np.linspace(0.1, 0.9, inter)
  #x_0 = np.array([0.5])
  next_samples = np.copy(x_0)
  density = []
  stored_samples = generate_gauss_samples_various(MEAN_matrix[:, 1], COV_matrix[:, 1, 1], samples_per_class=20000)

  ### CHANGE SIGNAL LENGTH:
  iter = 10

  current_ = np.zeros((iter, inter))
  next_ = np.zeros((iter, inter))
  num_samples = np.zeros((weights_matrix.shape[0]), dtype=int)

  for i in range(0, iter):
      current_[i] = np.copy(next_samples)
      weights_x0 = weights_matrix*gaussian_1d(next_samples, MEAN_matrix[:, 0], (COV_matrix[:, 0, 0]))
      weights_x0 = weights_x0/np.sum(weights_x0, 1).reshape(-1, 1)

      next_chosen = np.array([np.random.choice(weights_.shape[0], 1, p=weights_)[0] for weights_ in weights_x0])

      #print(next_chosen)
      next_samples = []
      for j in next_chosen:
          next_samples.append(stored_samples[j, num_samples[j]])
          num_samples[j]+=1
      next_samples = np.array(next_samples)
      density.append(np.histogram(next_samples, bins=100)[0])

      next_[i] = np.copy(next_samples)

  x_list.append(current_)
  y_list.append(next_)

  print('done')

def gaussian_nd(input, m, sigma):
    k = sigma.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.pinv(sigma)
    
    return ((2*np.pi)**(-k/2))*det**(-1/2)*np.exp(-(1/2)*np.sum((input-m)@inv*(input-m), 1))

# def gaussian_nd_save_some(input, m, sigma):
#     det = (sigma[:, 0]*sigma[:, 1])
#     inv = 1/sigma
    
#     return ((2*np.pi)**(-2/2))*det**(-1/2)*np.exp(-(1/2)*np.sum((input-m)*inv*(input-m), 1))

def gaussian_nd_save_some(MEAN, VARIANCE):
    bs = VARIANCE.shape[0]
    dim = VARIANCE.shape[1]

    det = VARIANCE[:, 0]
    for i in range(1, dim):
        det = det*VARIANCE[:, i]    
        
    product = np.sum(((MEAN.reshape(-1, dim)*(1/VARIANCE).reshape(-1, dim))*MEAN.reshape(-1, dim)), 1)
    
    return ((2*np.pi)**(-dim/2))*det**(-1/2)*np.exp(-(1/2)*product)

def gaussian_1d(input, m, sigma):
    det = sigma
    inv = 1/sigma
        
    input = input.reshape(-1, 1)
    m = m.reshape(1, -1)

    return ((2*np.pi)**(-1/2))*det**(-1/2)*np.exp(-(1/2)*((input-m)**2*inv))

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

x = x_list[0].reshape(-1)
y = y_list[0].reshape(-1)
joint = np.array((x, y)).T

### CHANGE COMPONENTS HERE:
gm_joint = BayesianGaussianMixture(n_components=10, random_state=0, covariance_type='diag', max_iter=100).fit(joint)

def compute_conditionalCE(joint, gm_joint):

  gmmmm = gm_joint.means_
  gmvvv = gm_joint.covariances_
  gmwww = gm_joint.weights_

  zeros = np.zeros((joint.shape))
  margin_x = gaussian_1d(joint[:, 0], gmmmm[:, 0], gmvvv[:, 0]).reshape(joint.shape[0], gmmmm.shape[0])
  responsibility = gmwww.reshape(1, -1)*margin_x/(np.sum(gmwww.reshape(1, -1)*margin_x, 1).reshape(-1, 1))
  responsibility = responsibility

  margin_y = gaussian_1d(joint[:, 1], gmmmm[:, 1], gmvvv[:, 1]).reshape(joint.shape[0], gmmmm.shape[0])
  numerator = np.mean(responsibility*margin_y)

  meandiff = (gmmmm[:, 1].reshape(-1, 1) - gmmmm[:, 1].reshape(1, -1)).reshape(-1, 1)
  vardiff = (gmvvv[:, 1].reshape(-1, 1) + gmvvv[:, 1].reshape(1, -1)).reshape(-1, 1)

  pair_wise_y = gaussian_nd_save_some(meandiff, vardiff)

  denominator = 0

  for i in range(0, joint.shape[0]):
    pair_wise_weight = responsibility[i].reshape(-1, 1)*responsibility[i].reshape(1, -1)
    denominator += np.mean(pair_wise_weight*pair_wise_y.reshape(gmmmm.shape[0], gmmmm.shape[0]))

  denominator = denominator/joint.shape[0]

  print('Estimated CE:', numerator/np.sqrt(denominator))
  return numerator/np.sqrt(denominator)

#### TO USE THIS FUNCTION, gm_joint can be from any GMM trained by scipy
compute_conditionalCE(joint, gm_joint)