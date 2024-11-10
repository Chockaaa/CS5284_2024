Lecture : Introduction to Graph Science
=======================================

Lab 01 : Generate artificial LFR social networks – Solution
-----------------------------------------------------------

Xavier Bresson, Nian Liu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/02_Graph_Science'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    %matplotlib inline
    #%matplotlib notebook 
    import matplotlib.pyplot as plt
    import subprocess # print output of LFR code
    import scipy.sparse # sparse matrix
    import scipy.sparse.linalg
    import pylab # 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot
    from lib import *
    import warnings; warnings.filterwarnings("ignore")
    import platform


Question 1 : Define a LFR graph by selecting the hyper-parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp

**Hyperparameters to select:** \* N : Number of nodes \* k : Number of
communities \* maxc : Maximum community size \* minc : Minimum community
size \* mu : Mixing parameter between communities (lower values indicate
less mixing between communities)

.. code:: ipython3

    # LFR parameters
    N = 1000
    k = maxk = 10
    
    # different community sizes
    alpha = 0.5
    minc = round((1-alpha)*N/k)
    maxc = round((1+alpha)*N/k)
    
    # mixing parameter
    mu = 0.1
    
    print('N, k, minc, maxc, mu :', N, k, minc, maxc, mu)


.. code:: ipython3

    print('os is :', platform.system())
    if platform.system()!='Windows': # os is not Windows
        # Compile LFR code
        !{'cd LFR; make; cd ..'}
        # Run LFR code
        cmd = './LFR/benchmark -N ' + str(N) + ' -k ' + str(k) + ' -maxk ' + str(maxk) + \
        ' -mu ' + str(mu) + ' -minc ' + str(minc) + ' -maxc ' + str(maxc)
        print(cmd)
        !{cmd}
        # Get path to generated dataset
        path_dataset = './'
    else: # os is Windows
        print('LFR cannot be compiled under Windows -- Using saved dataset instead')
        path_dataset = 'datasets/'


.. code:: ipython3

    # Read LFR data:
    #     'community.dat' contains the ground truth of communities from 1 to K
    #     'network.dat' contains the edges of the LFR network
    
    community = np.loadtxt(path_dataset + 'community.dat')
    community = community[:,1]
    print('nb of nodes=',community.shape[0])
    
    network = np.loadtxt(path_dataset + 'network.dat') 
    network -= 1 # index starts at 0 with python
    print('nb of edges=',network.shape[0])


Question 2 : Construct a sparse graph using the scipy library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://docs.scipy.org/doc/scipy/reference/sparse.html

Compute a sparse matrix W that represents the adjacency matrix of an LFR
network.

For example, you can create
``W = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape)``
using:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

**Parameters:** \* row_ind : A vector representing the starting node
index of the edge i, i.e. the row index of W. \* col_ind : A vector
representing the ending node index of the edge j, i.e., the column index
of W. \* data : The value associated with edge ij.

Finally, print the shape and type of the spare matrix.

.. code:: ipython3

    # Create LFR adjacency matrix W
    nv = community.shape[0] # nb of vertices
    ne = network.shape[0]
    #print(nv,ne)
    
    row = network[:,0]
    col = network[:,1]
    data = np.ones([ne])
    #print(row.shape,col.shape,data.shape)
    
    W = scipy.sparse.csr_matrix((data, (row, col)), shape=(nv, nv))
    print(W.shape,type(W))


Question 3 : Visualize the sparse adjacency matrix W.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``spy()`` function from Matplotlib :

https://matplotlib.org/stable/api/\_as_gen/matplotlib.pyplot.spy.html

After plotting, analyze the matrix. Does it reveal any interesting
structures or patterns?

.. code:: ipython3

    # Plot adjacency matrix
    plt.figure(1)
    plt.spy(W,precision=0.01, markersize=1)
    plt.show()


.. code:: ipython3

    ######################################
    # Function that reindexes W according to communities/classes
    ######################################
    
    ######################################
    # Usage: 
    #   [reindexed_W,reindexed_C] = reindex_W_with_C(W,C)
    #
    # Notations:
    #   n = nb_data
    #   nc = nb_communities
    #
    # Input variables:
    #   W = Adjacency matrix. Size = n x n.
    #   C = Classes used for reindexing W. Size = n x 1. Values in [0,1,...,nc-1].
    #
    # Output variables:
    #   reindexed_W = reindexed adjacency matrix. Size = n x n.
    #   reindexed_C = reindexed classes C. Size = n x 1. Values in [0,1,...,nc-1].
    ######################################
    
    def reindex_W_with_classes(W,C):
        n = C.shape[0] # nb of vertices
        nc = len(np.unique(C)) # nb of communities
        reindexing_mapping = np.zeros([n]) # mapping for reindexing W
        reindexed_C = np.zeros([n]) # reindexed C
        tot = 0
        for k in range(nc):
            cluster = (np.where(C==k))[0]
            length_cluster = len(cluster)
            x = np.array(range(tot,tot+length_cluster))
            reindexing_mapping[cluster] = x
            reindexed_C[x] = k
            tot += length_cluster
            
        idx_row,idx_col,val = scipy.sparse.find(W)
        idx_row = reindexing_mapping[idx_row]
        idx_col = reindexing_mapping[idx_col]
        reindexed_W = scipy.sparse.csr_matrix((val, (idx_row, idx_col)), shape=(n, n))
    
        return reindexed_W,reindexed_C
        

.. code:: ipython3

    C = community - 1
    [W,C] = reindex_W_with_classes(W,C)


Question 4 : Visualize the re-indexed adjacency matrix W based on ground truth communities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Does this visualization reveal any notable structures or patterns?

.. code:: ipython3

    # Plot same W but according to communities
    plt.figure(2)
    plt.spy(W,precision=0.01, markersize=1)
    plt.show()


Comment: The remainder of the notebook focuses on visualizing the graph in 2D and 3D spaces.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will explore graph visualization techniques in more detail later on.

.. code:: ipython3

    ######################################
    # Graph Laplacian Operator
    ######################################
    
    ######################################
    # Usages: 
    #   L = compute_graph_laplacian(W); # compute normalized graph Laplacian
    #   L = compute_graph_laplacian(W,False); # compute UNnormalized graph Laplacian
    #
    # Notations:
    #   n = nb_data
    #
    # Input variables:
    #   W = Adjacency matrix. Size = n x n.
    #
    # Output variables:
    #   L = Graph Laplacian. Size = n x n.
    ######################################
    
    def graph_laplacian(W, normalized=True):
        
        # Degree vector
        d = W.sum(axis=0)
    
        # Laplacian matrix
        if not normalized:
            D = scipy.sparse.diags(d.A.squeeze(), 0)
            L = D - W
        else:
            d += np.spacing(np.array(0, W.dtype)) # d += epsilon
            d = 1.0 / np.sqrt(d)
            D = scipy.sparse.diags(d.A.squeeze(), 0)
            I = scipy.sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
        return L
        

.. code:: ipython3

    ######################################
    # Visualization technique:
    #   Belkin-Niyogi, Laplacian eigenmaps for dimensionality reduction and data representation, 2003
    ######################################
    
    ######################################
    # Usage: 
    #   [X,Y,Z] = compute_non_linear_dim_reduction(W)
    #
    # Notations:
    #   n = nb_data
    #
    # Input variables:
    #   W = Adjacency matrix. Size = n x n.
    #
    # Output variables:
    #   X = 1st data coordinates in low-dim manifold. Size n x 1.
    #   Y = 2nd data coordinates in low-dim manifold. Size n x 1.
    #   Z = 3rd data coordinates in low-dim manifold. Size n x 1.
    ######################################
    
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:,idx]
    
    def compute_non_linear_dim_reduction(W):
        
        # Compute normalized graph Laplacian
        L = graph_laplacian(W)
        
        # Regularization for ill-posed graphs
        L = L + 1e-6* scipy.sparse.identity(L.shape[0], dtype=W.dtype)
    
        # Compute the first three Laplacian Eigenmaps
        lamb, U = scipy.sparse.linalg.eigsh(L, k=4, which='SM')
        
        # Sort eigenvalue from smallest to largest values
        lamb, U = sort(lamb, U)
        
        # Coordinates of graph vertices in the low-dim embedding manifold
        X = U[:,1]
        Y = U[:,2]
        Z = U[:,3]
    
        return X,Y,Z
    
    [X,Y,Z] = compute_non_linear_dim_reduction(W)
    #print(X.shape)


.. code:: ipython3

    # Visualize the social network in 2D
    plt.figure(3)
    plt.scatter(X, Y, c=C, s=3, color=pyplot.jet())
    plt.show()


.. code:: ipython3

    # 3D Visualization
    import plotly.graph_objects as go
    data = go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=3, color=C, colorscale='jet', opacity=1)) # data as points
    # data = go.Scatter3d(x=Xvis, y=Yvis, z=Zvis, mode='markers', marker=dict(size=1, color=C, colorscale='jet', opacity=1, showscale=True)) # w/ colobar 
    fig = go.Figure(data=[data]) 
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0)) # tight layout but t=25 required for showing title 
    fig.update_layout(autosize=False, width=600, height=600, title_text="3D visualization of LFR graph") # figure size and title
    # fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis = dict(visible=False))) # no grid, no axis 
    # fig.update_layout(scene = dict(xaxis_title = ' ', yaxis_title = ' ', zaxis_title = ' ')) # no axis name 
    fig.update_layout(scene = dict(zaxis = dict(showgrid = True, showticklabels = False), zaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(yaxis = dict(showgrid = True, showticklabels = False), yaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(xaxis = dict(showgrid = True, showticklabels = False), xaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
    fig.show()



Lecture : Introduction to Graph Science
=======================================

Lab 02 : Modes of variations of a graph system – Solution
---------------------------------------------------------

Xavier Bresson, Nian Liu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/02_Graph_Science'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    %matplotlib inline
    #%matplotlib notebook 
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    import scipy.io # Import data
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")


.. code:: ipython3

    # Load disco-boy network
    mat = scipy.io.loadmat('datasets/discoboy_network.mat')
    X = mat['X']
    W = mat['W']
    print(X.shape,W.shape)


.. code:: ipython3

    # Visualize the dataset in 2D
    plt.figure(1)
    size_vertex_plot = 20.
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(X.shape[0]), color=pyplot.jet())
    plt.show()


.. code:: ipython3

    # Compute graph Laplacian
    L = graph_laplacian(W)
    
    # Compute modes of variations of graph system = Fourier functions
    lamb, U = scipy.sparse.linalg.eigsh(L, k=9, which='SM')


Question 1 : Plot the Fourier functions of the human silhouette, encoded by a k-NN graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fourier functions are given by the columns of the eigenvector matrix
U, which can be computed using the ``scipy`` library:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

| Hint: You can use the ``scatter()`` function to plot the data:
| https://matplotlib.org/stable/api/\_as_gen/matplotlib.pyplot.scatter.html

Examine the Fourier functions, particularly focusing on their
oscillatory characteristics.

Can you identify any trends or properties in the oscillations?

.. code:: ipython3

    # Plot mode of variations
    for i in range(1,10):
        plt.figure(str(10+i))
        plt.scatter(X[:,0], X[:,1], c=U[:,i-1], s=size_vertex_plot*np.ones(X.shape[0]), color=pyplot.jet())
        plt.colorbar()
        plt.show()
        

Question 2 : Plot the Fourier functions of a standard grid, represented by a k-NN graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you plot the Fourier functions for a standard grid, consider if they
look familiar.

What are these functions commonly called?

.. code:: ipython3

    # Load grid network (for images)
    mat = scipy.io.loadmat('datasets/grid_network.mat')
    X = mat['X']
    W = mat['W']
    print(X.shape,W.shape)
    
    # Visualize the network in 2D
    plt.figure(20)
    size_vertex_plot = 20.
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(X.shape[0]), color=pyplot.jet())
    plt.show()


.. code:: ipython3

    # Compute graph Laplacian
    L = graph_laplacian(W)
    
    # Compute modes of variations of graph system = Fourier functions
    lamb, U = scipy.sparse.linalg.eigsh(L, k=9, which='SM')
    
    # Plot mode of variations
    for i in range(1,10):
        plt.figure(str(20+i))
        plt.scatter(X[:,0], X[:,1], c=U[:,i-1], s=size_vertex_plot*np.ones(X.shape[0]), color=pyplot.jet())
        plt.show()
        

Lecture : Introduction to Graph Science
=======================================

Lab 03 : Graph construction with pre-processing – Solution
----------------------------------------------------------

Xavier Bresson, Nian Liu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/02_Graph_Science'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    %matplotlib inline
    #%matplotlib notebook 
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import scipy.sparse # sparse matrix
    import scipy.sparse.linalg
    from matplotlib import pyplot
    import scipy.io # import data
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import compute_ncut
    import sklearn.metrics.pairwise # distance function
    import warnings; warnings.filterwarnings("ignore")


.. code:: ipython3

    # Load two-moon datasets
    mat = scipy.io.loadmat('datasets/two_moon_100D.mat'); dim = 100
    #mat = scipy.io.loadmat('datasets/two_moon_2D.mat'); dim = 2
    X = mat['X']
    n = X.shape[0]; C = np.zeros([n]); C[-int(n/2):] = 1
    print(X.shape,C.shape)
    
    # Visualize in 2D
    plt.figure(1)
    size_vertex_plot = 20.
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C, color=pyplot.jet())
    plt.title('Visualization of two-moon datase (with ground truth), DIMENTIONALITY= ' + str(dim))
    plt.show()


Question 1 : Center the dataset X and plot the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a dataset :math:`X\in\mathbb{R}^{n\times d}`, centering it to have
zero mean involves subtracting the mean of the dataset from each data
point:

:math:`x_i \ \leftarrow x_i - \textrm{mean}(\{x_i\}_{i=1}^n)\in\mathbb{R}^d`

.. code:: ipython3

    # size(X) = nb_data x dim
    Xzc = X - np.mean(X,axis=0)
    print(Xzc.shape)
    
    plt.figure(2)
    plt.scatter(Xzc[:,0], Xzc[:,1], s=size_vertex_plot*np.ones(n), c=C, color=pyplot.jet())
    plt.title('Center the data')
    plt.show()


Question 2 : Normalize the variance of the dataset X and plot the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To normalize the variance of the dataset
:math:`X\in\mathbb{R}^{n\times d}`, divide each data point by the
standard deviation of the dataset:

:math:`x_i \ \leftarrow x_i / \textrm{std}(\{x_i\}_{i=1}^n)\in\mathbb{R}^d`

.. code:: ipython3

    # size(X) = nb_data x dim
    Xnvar = Xzc/ np.sqrt(np.sum(Xzc**2,axis=0)+1e-10)
    print(Xnvar.shape)
    
    plt.figure(3)
    plt.scatter(Xnvar[:,0], Xnvar[:,1], s=size_vertex_plot*np.ones(n), c=C, color=pyplot.jet())
    plt.title('Normalize the variance')
    plt.show()


Question 3 : Project the dataset X onto a unit sphere and plot the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To project a dataset :math:`X\in\mathbb{R}^{n\times d}` onto a unit
sphere, normalize each data point by its Euclidean norm:

:math:`x_i \ \leftarrow x_i /||x_i||_2 \in\mathbb{R}^d`

.. code:: ipython3

    # size(X) = nb_data x dim
    Xl2proj = ( Xzc.T / np.sqrt(np.sum(Xzc**2,axis=1)+1e-10) ).T
    print(Xl2proj.shape)
    
    plt.figure(4)
    plt.scatter(Xl2proj[:,0], Xl2proj[:,1], s=size_vertex_plot*np.ones(n), c=C, color=pyplot.jet())
    plt.title('Projection on the L2-ball')
    plt.show()


Question 4 : Construct a k-NN graph using L2/Euclidean distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Center the Dataset:** Begin by centering the dataset X to have a
   mean of zero.

2. **Compute Pairwise Distances:** Calculate the pairwise Euclidean
   distances D between all data points. You can use the
   ``pairwise_distances()`` function from the ``sklearn`` library:
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html.

3. **Sort Distance Matrix:** Organize the distance matrix D in ascending
   order, from the smallest to the largest distances.

4. **Extract k-NN:** Identify the k-nearest neighbors for each data
   point based on the sorted distance matrix.

5. **Compute Adjacency Matrix:** Create the adjacency matrix W using
   Gaussian weights. This involves applying the Gaussian function to the
   distances.

6. **Make Adjacency Matrix Sparse:** Convert the adjacency matrix W into
   a sparse matrix to optimize storage and computation.

7. **Symmetrize Matrix:** Ensure the adjacency matrix W is symmetric by
   averaging W with its transpose.

.. code:: ipython3

    ######################################
    # Construct a k-NN graph with L2/Euclidean distance
    ######################################
    
    # Compute L2/Euclidean distance between all pairs of points
    Xzc = X - np.mean(X,axis=0) # zero-centered data
    D = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean', n_jobs=1)
    print(D.shape)
    
    # Sort distance matrix
    k = 10 # number of nearest neighbors
    idx = np.argsort(D)[:,:k] # indices of k nearest neighbors
    Dnot_sorted = np.copy(D)
    D.sort() # sort D from smallest to largest values
    Dsorted = np.copy(D)
    print(D.shape)
    D = D[:,:k]
    print(D.shape)
    
    # Compute weight matrix
    sigma2 = np.mean(D[:,-1])**2 # graph scale
    W = np.exp(- D**2 / sigma2)
    #print(W.shape)
    
    # Make W sparse
    n = X.shape[0]
    row = np.arange(0, n).repeat(k)
    col = idx.reshape(n*k)
    data = W.reshape(n*k)
    W = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    
    # Make W is symmetric
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    
    # No self-connections
    #W.setdiag(0)
    
    print(W.shape)
    print(W.nnz)


.. code:: ipython3

    # Visualize distances
    fig, (ax1, ax2) = plt.subplots(1,2)
    #fig.suptitle('Title of figure 2', fontsize=15)
    
    ax1.set_title('Euclidean distances for all data points')
    im1 = ax1.imshow(Dnot_sorted, interpolation='nearest')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.1)
    ax1.get_figure().colorbar(im1, cax=cax1)
    
    ax2.set_title('Sorted distances')
    im2 = ax2.imshow(Dsorted, interpolation='nearest')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.1)
    ax2.get_figure().colorbar(im2, cax=cax2)
    
    fig.tight_layout()
    
    fig.show()


.. code:: ipython3

    plt.figure(5)
    plt.spy(W,precision=0.01, markersize=1, color=pyplot.jet())
    plt.show()


Comment
~~~~~~~

To assess the quality of the constructed adjacency matrix, one effective
approach is to compute the classification error relative to some ground
truth communities.

In this notebook, we evaluate the adjacency matrix by comparing the
ground truth communities C with the communities estimated using the NCut
graph partitioning algorithm: http://www.cis.upenn.edu/~jshi/software.

.. code:: ipython3

    Cncut, acc = compute_ncut(W, C, 2)
    print(acc)


.. code:: ipython3

    plt.figure(6)
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cncut, color=pyplot.jet())
    plt.title('Clustering result with EUCLIDEAN distance, ACCURACY= '+ str(acc))
    plt.show()


Question 5 : Construct a k-NN graph using cosine distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Construct k-NN Graph:** Build a k-NN graph using cosine distance as
   the metric. The cosine distance may be computed using the
   ``arccos()`` function from NumPy:
   https://numpy.org/doc/stable/reference/generated/numpy.arccos.html.

2. **Evaluate Quality:** Assess the quality of the adjacency matrix
   construction by comparing it with the ground truth communities. Use
   the NCut graph partitioning algorithm to estimate communities and
   evaluate the performance.

.. code:: ipython3

    ######################################
    # Construct k-NN graph with Cosine distance
    ######################################
    
    # Compute Cosine distance between all pairs of points
    Xzc = X - np.mean(X,axis=0) # zero-centered data
    Xl2proj = ( Xzc.T / np.sqrt(np.sum(Xzc**2,axis=1)+1e-10) ).T # Projection on the sphere, i.e. ||x_i||_2 = 1
    D = Xl2proj.dot(Xl2proj.T)
    #print(D.shape)
    
    # Sort D according in descending order
    k = 10 # number of nearest neighbors
    idx = np.argsort(D)[:,::-1][:,:k] # indices of k nearest neighbors
    Dnot_sorted = np.copy(D)
    D.sort(axis=1)
    D[:] = D[:,::-1]
    Dsorted = np.copy(D)
    
    # Cosine distance
    Dcos = np.abs(np.arccos(D))
    D = Dcos
    D = D[:,:k]
    print(D.shape)
    
    # Compute Weight matrix
    sigma2 = np.mean(D[:,-1])**2 # graph scale
    W = np.exp(- D**2 / sigma2)
    #print(W.shape)
    
    # Make W sparse
    n = X.shape[0]
    row = np.arange(0, n).repeat(k)
    col = idx.reshape(n*k)
    data = W.reshape(n*k)
    W = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    
    # Make W is symmetric
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    
    # No self-connections
    #W.setdiag(0)
    
    print(W.shape)
    print(W.nnz)


.. code:: ipython3

    # Visualize distances
    fig, (ax1, ax2) = plt.subplots(1,2)
    #fig.suptitle('Title of figure 2', fontsize=15)
    
    ax1.set_title('Euclidean distances for all data points')
    im1 = ax1.imshow(Dnot_sorted, interpolation='nearest')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.1)
    ax1.get_figure().colorbar(im1, cax=cax1)
    
    ax2.set_title('Sorted distances')
    im2 = ax2.imshow(Dsorted, interpolation='nearest')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.1)
    ax2.get_figure().colorbar(im2, cax=cax2)
    
    fig.tight_layout()
    
    fig.show()


.. code:: ipython3

    plt.figure(7)
    plt.spy(W,precision=0.01, markersize=1, color=pyplot.jet())
    plt.show()


.. code:: ipython3

    Cncut, acc = compute_ncut(W, C, 2)
    print(acc)


.. code:: ipython3

    plt.figure(8)
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cncut, color=pyplot.jet())
    plt.title('Clustering result with EUCLIDEAN distance, ACCURACY= '+ str(acc))
    plt.show()



Lecture : Introduction to Graph Science
=======================================

Lab 04 : Construct a network of text documents – Solution
---------------------------------------------------------

Xavier Bresson, Nian Liu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/02_Graph_Science'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    %matplotlib inline
    #%matplotlib notebook 
    import matplotlib.pyplot as plt
    import pylab # 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot
    import scipy.io # import data
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import compute_ncut
    from lib.utils import reindex_W_with_classes
    from lib.utils import nldr_visualization
    from lib.utils import construct_knn_graph
    import warnings; warnings.filterwarnings("ignore")


Question
~~~~~~~~

In the previous notebook, we examined a graph of images and noted that
the quality of the adjacency matrix construction remained relatively
fine whether using Euclidean distance or cosine distance.

In this notebook, we will analyze a graph of text documents, where each
document is represented by a histogram of words.

1. **Characteristics of Text Document Histograms:** What is the key
   feature of text documents represented by histograms of words?

2. **Compare Graph Construction Quality:** Run and compare the quality
   of graph construction both visually and quantitatively using the NCut
   graph partitioning algorithm, given the ground truth communities.

3. **Best Graph Construction Approach:** Determine which graph
   construction method — Euclidean distance or cosine distance — is more
   effective for this type of graph.

.. code:: ipython3

    # Load 10 classes of 4,000 text documents
    mat = scipy.io.loadmat('datasets/20news_5classes_raw_data.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,d,nc)


.. code:: ipython3

    # Compute the k-NN graph with L2/Euclidean distance
    W_euclidean = construct_knn_graph(X, 10, 'euclidean')
    #print(W_euclidean)


.. code:: ipython3

    Cncut,acc = compute_ncut(W_euclidean, Cgt, nc)
    print(acc)


.. code:: ipython3

    [reindexed_W_gt,reindexed_C_gt] = reindex_W_with_classes(W_euclidean,Cgt)
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W_euclidean,Cncut)


.. code:: ipython3

    plt.figure(1)
    plt.spy(W_euclidean,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A')
    plt.show()
    
    plt.figure(2)
    plt.spy(reindexed_W_gt,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to GROUND TRUTH communities')
    plt.show()
    
    plt.figure(3)
    plt.spy(reindexed_W_ncut,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to GROUND TRUTH communities')
    plt.show()


.. code:: ipython3

    # Visualization
    [X,Y,Z] = nldr_visualization(W_euclidean)
    
    plt.figure(4)
    size_vertex_plot = 10
    plt.scatter(X, Y, s=size_vertex_plot*np.ones(n), c=Cncut, color=pyplot.jet())
    plt.title('Clustering result with EUCLIDEAN distance, ACCURACY= '+ str(acc))
    plt.show()
    
    # 3D Visualization
    fig = pylab.figure(5)
    ax = Axes3D(fig)
    ax.scatter(X, Y, Z, c=Cncut, color=pyplot.jet())
    pyplot.show()


.. code:: ipython3

    # Compute the k-NN graph with Cosine distance
    X = mat['X']
    W_cosine = construct_knn_graph(X,10,'cosine')
    
    Cncut, acc = compute_ncut(W_cosine, Cgt, nc)
    print(acc)


.. code:: ipython3

    [reindexed_W_gt,reindexed_C_gt] = reindex_W_with_classes(W_cosine,Cgt)
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W_cosine,Cncut)


.. code:: ipython3

    plt.figure(6)
    plt.spy(W_cosine,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A')
    plt.show()
    
    plt.figure(7)
    plt.spy(reindexed_W_gt,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to GROUND TRUTH communities')
    plt.show()
    
    plt.figure(8)
    plt.spy(reindexed_W_ncut,precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to NCUT communities')
    plt.show()


.. code:: ipython3

    # Visualization
    [X,Y,Z] = nldr_visualization(W_cosine)
    
    plt.figure(9)
    size_vertex_plot = 1
    plt.scatter(X, Y, s=size_vertex_plot*np.ones(n), c=Cncut, color=pyplot.jet())
    plt.title('Clustering result with EUCLIDEAN distance, ACCURACY= '+ str(acc))
    plt.show()
    


.. code:: ipython3

    # 3D Visualization
    import plotly.graph_objects as go
    data = go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=2, color=Cncut, colorscale='jet', opacity=1)) # data as points
    # data = go.Scatter3d(x=Xvis, y=Yvis, z=Zvis, mode='markers', marker=dict(size=1, color=C, colorscale='jet', opacity=1, showscale=True)) # w/ colobar 
    fig = go.Figure(data=[data]) 
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0)) # tight layout but t=25 required for showing title 
    fig.update_layout(autosize=False, width=600, height=600, title_text="3D visualization of 20newsgroups graph") # figure size and title
    # fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis = dict(visible=False))) # no grid, no axis 
    # fig.update_layout(scene = dict(xaxis_title = ' ', yaxis_title = ' ', zaxis_title = ' ')) # no axis name 
    fig.update_layout(scene = dict(zaxis = dict(showgrid = True, showticklabels = False), zaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(yaxis = dict(showgrid = True, showticklabels = False), yaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(xaxis = dict(showgrid = True, showticklabels = False), xaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
    fig.show()


Lecture : Graph Clustering
==========================

Lab 01 : Standard k-means – Solution
------------------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    #%matplotlib notebook 
    %matplotlib inline 
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    import time
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import construct_kernel
    from lib.utils import compute_kernel_kmeans_EM
    from lib.utils import compute_kernel_kmeans_spectral
    from lib.utils import compute_purity
    from lib.utils import construct_knn_graph
    from lib.utils import compute_ncut
    from lib.utils import compute_pcut
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")


Gaussian Mixture Model (GMM)
============================

.. code:: ipython3

    # Load raw data images
    mat = scipy.io.loadmat('datasets/GMM.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt'] - 1; Cgt = Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,d,nc)


.. code:: ipython3

    plt.figure(1)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, cmap='jet')
    plt.title('Gaussian Mixture Model (GMM) -- Linearly separable data points')
    plt.show()


Question 1: Evaluating the impact of different initializations on k-Means performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Initialization Methods:** \* **Constant Function:** You can use
``numpy.ones()`` for this initialization. \* **Random Function:**
Consider ``numpy.random.randint()`` for random initialization.

Discuss how these initialization methods affect the clustering results
on the distribution of a Gaussian Mixture Model.

.. code:: ipython3

    # Initialization
    n = X.shape[0]
    C_kmeans = np.ones(shape=n) # constant initialization
    C_kmeans = np.random.randint(nc,size=n) # random initialization
    
    # Linear Kernel for standard K-Means
    Ker = X.dot(X.T)
    print(Ker.shape)
    
    # Loop
    Cold = np.ones([n])
    diffC = 1e10
    Theta = np.ones(n) # Same weight for each data
    Theta = np.diag(Theta)
    Ones = np.ones((1,n))
    En_iters = []
    Clusters_iters = []; Clusters_iters.append(C_kmeans)
    k = 0
    while (k<50) & (diffC>1e-2):
        
        # Update iteration
        k += 1
        #print(k)
        
        # Distance Matrix D
        row = np.array(range(n))
        col = C_kmeans
        data = np.ones(n)
        F = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, nc)).todense()
        O = np.diag( np.array( 1./ (Ones.dot(Theta).dot(F) + 1e-6) ).squeeze() )
        T = Ker.dot(Theta.dot(F.dot(O)))
        D = - 2* T + np.repeat( np.diag(O.dot((F.T).dot(Theta.dot(T))))[None,:] ,n,axis=0)
        #print(D.shape)
        
        # Extract clusters
        C_kmeans = np.array(np.argmin(D,1)).squeeze()
        Clusters_iters.append(C_kmeans)
                    
        # L2 difference between two successive cluster configurations
        diffC = np.linalg.norm(C_kmeans-Cold)/np.linalg.norm(Cold)
        Cold = C_kmeans
            
        # K-Means energy
        En = np.multiply( (np.repeat(np.diag(Ker)[:,None],nc,axis=1) + D) , F)
        En_kmeans = np.sum(En)/n
        En_iters.append(En_kmeans)
        
    print(k)


.. code:: ipython3

    # Visualize k-means iterations
    fig, ax = plt.subplots()
    for k,C in enumerate(Clusters_iters):
        plt.scatter(X[:,0], X[:,1], s=10*np.ones(n), c=C, cmap='jet')
        plt.title('k-means clusters at iteration = ' + str(k+1) )
        display(fig)
        clear_output(wait=True)


.. code:: ipython3

    # Visualize loss vs iteration
    plt.figure(3)
    plt.plot(En_iters)
    plt.title('loss vs iteration')
    plt.show()


Two concentric circles
======================

.. code:: ipython3

    # Load raw data images
    mat = scipy.io.loadmat('datasets/two_circles.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt'] - 1; Cgt = Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,d,nc)
    
    plt.figure(10)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, cmap='jet')
    plt.title('Distribution of two circle distributions -- Non-linear data points')
    plt.show()


Question 2: Assessing k-Means performance with various initializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can you identify an initialization function that successfully separates
the two classes in this dataset?

Evaluate the effectiveness of k-means on this dataset.

.. code:: ipython3

    # Initialization
    n = X.shape[0]
    C_kmeans = np.ones(shape=n) # constant initialization
    C_kmeans = np.random.randint(nc,size=n) # random initialization
    
    # Linear Kernel for standard K-Means
    Ker = X.dot(X.T)
    print(Ker.shape)
    
    # Loop
    Cold = np.ones([n])
    diffC = 1e10
    Theta = np.ones(n) # Equal weight for each data
    Theta = np.diag(Theta)
    Ones = np.ones((1,n))
    En_iters = []
    Clusters_iters = []; Clusters_iters.append(C_kmeans)
    k = 0
    while (k<10) & (diffC>1e-2):
        
        # Update iteration
        k += 1
        #print(k)
        
        # Distance Matrix D
        row = np.array(range(n))
        col = C_kmeans
        data = np.ones(n)
        F = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, nc)).todense()
        O = np.diag( np.array( 1./ (Ones.dot(Theta).dot(F) + 1e-6) ).squeeze() )
        T = Ker.dot(Theta.dot(F.dot(O)))
        D = - 2* T + np.repeat( np.diag(O.dot((F.T).dot(Theta.dot(T))))[None,:] ,n,axis=0)
        #print(D.shape)
        
        # Extract clusters
        C_kmeans = np.array(np.argmin(D,1)).squeeze()
        Clusters_iters.append(C_kmeans)
                    
        # L2 difference between two successive cluster configurations
        diffC = np.linalg.norm(C_kmeans-Cold)/np.linalg.norm(Cold)
        Cold = C_kmeans
            
        # K-Means energy
        En = np.multiply( (np.repeat(np.diag(Ker)[:,None],nc,axis=1) + D) , F)
        En_kmeans = np.sum(En)/n
        En_iters.append(En_kmeans)
        
    print(k)
    
    # Visualize k-means iterations
    fig, ax = plt.subplots()
    for k,C in enumerate(Clusters_iters):
        plt.scatter(X[:,0], X[:,1], s=10*np.ones(n), c=C, cmap='jet')
        plt.title('k-means clusters at iteration = ' + str(k+1) )
        display(fig)
        clear_output(wait=True)
        
    # Visualize loss vs iteration
    plt.figure(12)
    plt.plot(En_iters)
    plt.title('loss vs iteration')
    plt.show()



Lecture : Graph Clustering
==========================

Lab 02 : Kernel k-means – Solution
----------------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline 
    import matplotlib.pyplot as plt
    import time
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import construct_kernel
    from lib.utils import compute_kernel_kmeans_EM
    from lib.utils import compute_kernel_kmeans_spectral
    from lib.utils import compute_purity
    from lib.utils import construct_knn_graph
    from lib.utils import compute_ncut
    from lib.utils import compute_pcut
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")


.. code:: ipython3

    # Load two-circle dataset
    mat = scipy.io.loadmat('datasets/two_circles.mat') 
    X = mat['X'] # (2000, 2), numpy.ndarray
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt']-1; Cgt = Cgt.squeeze() # (2000,)
    nc = len(np.unique(Cgt)) # 2
    print('n,d,nc:',n,d,nc)
    
    plt.figure(1)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, cmap='jet')
    plt.title('Distribution of two circle distributions -- Non-linear data points')
    plt.show()


Question 1: Evaluate the performance of *Linear* k-Means using Expectation-Maximization (EM) with multiple random initializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following code:
``compute_kernel_kmeans_EM(nc, Ker, Theta, n_trials)`` with the input
arguments:

-  nc : Number of clusters.
-  Ker : Kernel matrix of size n x n, where n is the number of data
   points.
-  Theta : Weight matrix of size n x n, typically a diagonal matrix with
   the weights of each data point.
-  n_trials : Number of runs for kernel k-means. The function returns
   the solution with the minimum final energy.

How many runs are necessary to obtain the correct solution?

.. code:: ipython3

    # Run standard/linear k-means
    Theta = np.ones(n) # Same weight for all data
    
    # Compute linear kernel for standard k-means
    Ker = construct_kernel(X, 'linear') # (2000, 2000)
    print(Ker.shape)
    
    # standard k-means
    n_trials = 10
    C_kmeans, En_kmeans = compute_kernel_kmeans_EM(nc, Ker, Theta, n_trials)
    
    # Plot
    plt.figure(2)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_kmeans, cmap='jet')
    plt.title('Standard k-means solution. Accuracy: ' + str(compute_purity(C_kmeans,Cgt,nc))[:5] +
             ', Energy: ' + str(En_kmeans)[:5])
    plt.show()


Question 2: Evaluate the performance of *Non-Linear* k-Means using EM with multiple random initializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How many runs are necessary to achieve the correct solution?

.. code:: ipython3

    # Run kernel/non-linear k-means with EM approach
     
    # Compute linear Kernel for standard k-means
    Ker = construct_kernel(X, 'kNN_gaussian', 100)
    print(Ker.shape)
    
    # Kernel k-means with EM approach
    n_trials = 10
    C_kmeans, En_kmeans = compute_kernel_kmeans_EM(nc, Ker, Theta, n_trials)
    
    # Plot
    plt.figure(3)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_kmeans, cmap='jet')
    plt.title('Kernel k-means solution with EM approach. Accuracy= ' + str(compute_purity(C_kmeans,Cgt,nc))[:5] +
             ', Energy= ' + str(En_kmeans)[:5])
    plt.show()


Question 3: Evaluate the performance of *Non-Linear* k-Means using the Spectral technique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the function ``compute_kernel_kmeans_spectral(nc, Ker, Theta)`` with
the following input arguments: \* nc : Number of clusters. \* Ker :
Kernel matrix of size n x n, where n is the number of data points. \*
Theta : Weight matrix of size n x n, a diagonal matrix containing the
weights of each data point.

Note that this function does not have an ``n_trials`` input argument.
Why do you think that is?

.. code:: ipython3

    # Run kernel/non-linear k-means with spectral approach
     
    # Compute linear kernel for standard k-means
    Ker = construct_kernel(X, 'kNN_gaussian', 100)
    print(Ker.shape)
    
    # Kernel k-means with spectral approach
    C_kmeans, En_kmeans = compute_kernel_kmeans_spectral(nc, Ker, Theta)
    
    # Plot
    plt.figure(4)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_kmeans, cmap='jet')
    plt.title('Kernel k-means solution with spectral approach. Accuracy= ' + 
              str(compute_purity(C_kmeans,Cgt,nc))[:5] + ' Energy= ' + str(En_kmeans)[:5])
    plt.show()



Lecture : Graph Clustering
==========================

Lab 03 : Metis – Solution
-------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pip install pymetis==2023.1 # install metis 
        !pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html # install dgl
        !pwd
        

.. code:: ipython3

    # Data conversion used in this notebook
    #  from DGL to networkx :                          G_nx = dgl.to_networkx(G_dgl)
    #  from scipy.sparse._csc.csc_matrix to DGL :      G_dgl = dgl.from_scipy(G_sp)
    #  from scipy.sparse._csc.csc_matrix to networkx : G_nx = nx.from_scipy_sparse_array(G_sp)
    #  from networkx to numpy :                        G_np = nx.to_numpy_array(G_nx)


.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    import time
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import compute_purity
    import warnings; warnings.filterwarnings("ignore")
    from lib.utils import nldr_visualization
    import os
    import torch
    import networkx as nx
    import time
    import dgl # DGL
    import pymetis # PyG Metis
    import platform


Artifical balanced graph
========================

Question 1: Construct a simple graph using the DGL library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference: https://www.dgl.ai

Create a graph with 9 nodes indexed from 0 to 8.

The set of edges is defined as E = { (0,2), (0,1), (1,2), (3,4), (4,5),
(6,7), (6,8), (7,8), (1,7), (2,3) }.

To construct this graph in DGL, you can use the ``dgl.graph()``
function: https://docs.dgl.ai/generated/dgl.graph.html#dgl-graph

For a simple example, see also:
https://docs.dgl.ai/en/0.2.x/tutorials/basics/1_first.html#step-1-creating-a-graph-in-dgl

Since the graph is undirected, DGL requires that you add both directions
for each edge, e.g. (0,2) and (2,0).

To automatically create an undirected graph from a directed one, you may
use the ``dgl.add_reverse_edges()`` function.

.. code:: ipython3

    # Build a simple artificail graph of 3 balanced communities with DGL
    r = torch.LongTensor([0, 0, 1, 3, 3, 4, 6, 6, 7, 1, 2])
    c = torch.LongTensor([2, 1, 2, 4, 5, 5, 7, 8, 8, 7, 3])
    n = 9
    G_dgl = dgl.graph((r,c), num_nodes=n)
    G_dgl = dgl.add_reverse_edges(G_dgl) # undirected graph
    print(G_dgl)
    print(G_dgl.nodes())
    print(G_dgl.edges())
    
    # Plot graph
    G_nx = dgl.to_networkx(G_dgl)
    plt.figure(figsize=[7,7])
    nx.draw_networkx(G_nx, with_labels=True)


Question 2: Partition the artificial graph using Metis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Metis is accessible through the PyMetis package:
| https://pypi.org/project/PyMetis

Experiment with different numbers of partitions to see how the graph is
divided.

.. code:: ipython3

    # Run Metis with PyMetis
    num_parts = 3
    G_nx = dgl.to_networkx(G_dgl)
    _, part_vert = pymetis.part_graph(num_parts, adjacency=G_nx)
    C_metis_pyg = torch.tensor(part_vert).long()
    print('C_metis_pyg',C_metis_pyg)
    plt.figure(figsize=[7,7])
    nx.draw_networkx(G_nx, with_labels=True, node_color=C_metis_pyg, cmap='jet')


.. code:: ipython3

    # Run Metis with DGL
    #  https://docs.dgl.ai/en/0.8.x/generated/dgl.dataloading.ClusterGCNSampler.html
    
    if platform.system()!='Windows': # os is not Windows
        try: os.remove("cluster_gcn.pkl") # remove any existing partition
        except: pass 
        num_parts = 3
        sampler = dgl.dataloading.ClusterGCNSampler(G_dgl, num_parts) 
        C_metis_dgl = torch.zeros(G_dgl.num_nodes()).long()
        for idx, (idx_start, idx_end) in enumerate(zip(sampler.partition_offset[:num_parts], sampler.partition_offset[1:num_parts+1])):
            C_metis_dgl[sampler.partition_node_ids[idx_start: idx_end]] = idx
        print('C_metis_dgl',C_metis_dgl)
        G_nx = dgl.to_networkx(G_dgl)
        plt.figure(figsize=[7,7])
        nx.draw_networkx(G_nx, with_labels=True, node_color=C_metis_dgl, cmap='jet')
    else:
        print('DGL has not yet implemented Metis under Windows.')
        

Real-world USPS image graph
===========================

.. code:: ipython3

    # Load USPS Music dataset
    mat = scipy.io.loadmat('datasets/USPS.mat')
    W = mat['W'] # scipy.sparse._csc.csc_matrix
    n = W.shape[0]
    Cgt = mat['Cgt']-1; Cgt = Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print('n,nc:',n,nc)


Question 3: Partition the real-world USPS image graph using Metis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://datasets.activeloop.ai/docs/ml/datasets/usps-dataset

After partitioning the graph using Metis, visualize it with clusters
represented by different colors.

Do the resulting clusters reveal any meaningful patterns?

.. code:: ipython3

    # Run Metis with PyMetis
    num_parts = nc
    G_nx = nx.from_scipy_sparse_array(W)
    start = time.time()
    _, part_vert = pymetis.part_graph(num_parts, adjacency=G_nx)
    print('Time(sec) : %.3f' % (time.time()-start) )
    C_metis_pyg = np.array(part_vert,dtype='int32')
    acc = compute_purity(C_metis_pyg, Cgt, nc)
    print('\nAccuracy Metis PyG :',acc)


.. code:: ipython3

    # Run Metis with DGL
    #  https://docs.dgl.ai/en/0.8.x/generated/dgl.dataloading.ClusterGCNSampler.html
    
    try: os.remove("cluster_gcn.pkl") # remove any existing partition
    except: pass 
    num_parts = nc
    G_dgl = dgl.from_scipy(W)
    start = time.time()
    sampler = dgl.dataloading.ClusterGCNSampler(G_dgl, num_parts) 
    print('Time(sec) : %.3f' % (time.time()-start) )
    C_metis_dgl = torch.zeros(G_dgl.num_nodes()).long()
    for idx, (idx_start, idx_end) in enumerate(zip(sampler.partition_offset[:num_parts], sampler.partition_offset[1:num_parts+1])):
        C_metis_dgl[sampler.partition_node_ids[idx_start: idx_end]] = idx
    print('C_metis_dgl',C_metis_dgl)
    C_metis_dgl = np.array(C_metis_dgl, dtype='int32')
    acc = compute_purity(C_metis_dgl, Cgt, nc)
    print('\nAccuracy Metis DGL :',acc)


.. code:: ipython3

    # Compute non-linear dim reduction
    start = time.time()
    [X,Y,Z] = nldr_visualization(nx.to_numpy_array(G_nx))
    print('Time(sec): %.3f' % (time.time()-start) )
    print(X.shape)
    
    # 2D Visualization
    plt.figure(3)
    plt.scatter(X, Y, c=C_metis_pyg, s=3, color=pyplot.jet())
    plt.show()


.. code:: ipython3

    # 3D Visualization
    import plotly.graph_objects as go
    data = go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1, color=C_metis_pyg, colorscale='jet', opacity=1)) # data as points
    # data = go.Scatter3d(x=Xvis, y=Yvis, z=Zvis, mode='markers', marker=dict(size=1, color=C, colorscale='jet', opacity=1, showscale=True)) # w/ colobar 
    fig = go.Figure(data=[data]) 
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0)) # tight layout but t=25 required for showing title 
    fig.update_layout(autosize=False, width=600, height=600, title_text="3D visualization of USPS image graph") # figure size and title
    # fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis = dict(visible=False))) # no grid, no axis 
    # fig.update_layout(scene = dict(xaxis_title = ' ', yaxis_title = ' ', zaxis_title = ' ')) # no axis name 
    fig.update_layout(scene = dict(zaxis = dict(showgrid = True, showticklabels = False), zaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(yaxis = dict(showgrid = True, showticklabels = False), yaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.update_layout(scene = dict(xaxis = dict(showgrid = True, showticklabels = False), xaxis_title = ' ') ) # no range values, no axis name, grid on
    fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
    fig.show()



Lecture : Graph Clustering
==========================

Lab 04 : NCut – Solution
------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    import time
    import sys
    sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import construct_kernel
    from lib.utils import compute_kernel_kmeans_EM
    from lib.utils import compute_kernel_kmeans_spectral
    from lib.utils import compute_purity
    from lib.utils import construct_knn_graph
    from lib.utils import compute_ncut
    from lib.utils import compute_pcut
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")


.. code:: ipython3

    # Load four-circle dataset
    mat = scipy.io.loadmat('datasets/four_circles.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print('(n,d,nc:',n,d,nc)
    
    plt.figure(1)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, color=pyplot.jet())
    plt.title('Ground truth communities of four concentric circles')
    plt.show()


.. code:: ipython3

    # Run standard/linear k-means with EM approach
    Theta = np.ones(n) # Same weight for each data
    # Compute linear Kernel for standard K-Means
    Ker = construct_kernel(X, 'linear')
    # Standard K-Means
    C_kmeans, En_kmeans = compute_kernel_kmeans_EM(nc, Ker, Theta, 10)
    # Plot
    plt.figure(2)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_kmeans)
    plt.title('Standard K-Means solution.\nAccuracy= ' + str(compute_purity(C_kmeans,Cgt,nc)) +
             ', Energy= ' + str(En_kmeans))
    plt.show()


.. code:: ipython3

    # Run kernel/non-linear k-means with spectral approach
    Ker = construct_kernel(X, 'kNN_gaussian', 100)
    # Kernel K-Means with Spectral approach
    C_kmeans, En_kmeans = compute_kernel_kmeans_spectral(nc, Ker, Theta)
    # Plot
    plt.figure(3)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_kmeans, color=pyplot.jet())
    plt.title('Kernel K-Means solution with Spectral.\nAccuracy= ' + 
              str(compute_purity(C_kmeans,Cgt,nc)) + ', Energy= ' + str(En_kmeans))
    plt.show()


Question 1: Apply the spectral NCut technique with different k values on the k-NN Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiment with the following values of k: {5, 10, 20, 40, 80}.

Observe and explain what happens when k is small, resulting in sparse
graphs, versus when k is large, leading to densely connected graphs.

.. code:: ipython3

    # Run NCut
    k = 40
    W = construct_knn_graph(X, k, 'euclidean_zelnik_perona')
    C_ncut, acc = compute_ncut(W, Cgt, nc)
    
    # Plot
    plt.figure(4)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_ncut, color=pyplot.jet())
    plt.title('NCut solution. Accuracy= ' + 
              str(compute_purity(C_ncut,Cgt,nc)) )
    plt.show()



Lecture : Graph Clustering
==========================

Lab 05 : PCut – Solution
------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook
    import matplotlib.pyplot as plt
    import time
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import construct_kernel
    from lib.utils import compute_kernel_kmeans_EM
    from lib.utils import compute_kernel_kmeans_spectral
    from lib.utils import compute_purity
    from lib.utils import construct_knn_graph
    from lib.utils import compute_ncut
    from lib.utils import compute_pcut
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")


Two-moon dataset
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Load raw data images
    mat = scipy.io.loadmat('datasets/two_moons.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,d,nc)
    
    # Plot
    plt.figure(1)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, cmap='jet')
    plt.title('Visualization of the two-moon datase with 2 classes, Data Dimentionality is 100')
    plt.show()


Question 1: Determine the optimal value of k for the k-NN graph in the spectral NCut technique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What value of k gives the best clustering results?

.. code:: ipython3

    # Run NCut 
    k = 10
    W = construct_knn_graph(X, k, 'euclidean')
    C_ncut, _ = compute_ncut(W, Cgt, nc)
    
    # Plot
    plt.figure(2)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_ncut, cmap='jet')
    plt.title('NCut solution. Accuracy= ' + 
              str(compute_purity(C_ncut, Cgt, nc))[:6] )
    plt.show()


Question 2: Evaluate the PCut technique with different values of k for the k-NN graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What value ofk produces the most effective clustering result?

Additionally, what is the range of k that provides optimal clustering
performance?

.. code:: ipython3

    # Run PCut
    k = 10
    W = construct_knn_graph(X, k, 'euclidean')
    C_pcut, _ = compute_pcut(W, Cgt, nc, 2, 200)
    
    # Plot
    plt.figure(3)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=C_pcut, cmap='jet')
    plt.title('PCut solution. Accuracy= ' + 
              str(compute_purity(C_pcut, Cgt, nc))[:6] )
    plt.show()


Comment: Compare the spectral NCut and PCut techniques on two real-world graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run both the Spectral NCut and PCut techniques on two real-world graphs
and compare their performance.

USPS image graph
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Load USPS dataset
    mat = scipy.io.loadmat('datasets/USPS.mat')
    W = mat['W'] # 'scipy.sparse._csc.csc_matrix'
    n = W.shape[0]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,nc)


.. code:: ipython3

    Cncut, acc = compute_ncut(W,Cgt,nc)
    print('Ncut accuracy =',acc)


.. code:: ipython3

    Cpcut, acc = compute_pcut(W,Cgt,nc,5,10)
    print('Pcut accuracy =',acc)


MIREX music graph
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Load USPS dataset
    mat = scipy.io.loadmat('datasets/MIREX.mat')
    W = mat['W'] # 'scipy.sparse._csc.csc_matrix'
    n = W.shape[0]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print(n,nc)


.. code:: ipython3

    Cncut, acc = compute_ncut(W,Cgt,nc)
    print('Ncut accuracy =',acc)


.. code:: ipython3

    Cpcut, acc = compute_pcut(W,Cgt,nc,0.5,400)
    print('Pcut accuracy =',acc)



Lecture : Graph Clustering
==========================

Lab 06 : Louvain Algorithm – Solution
-------------------------------------

Xavier Bresson, Jiaming Wang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/03_Graph_Clustering'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        !pip install python-louvain==0.15 # install louvain
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    import time
    import sys; sys.path.insert(0, 'lib/')
    %load_ext autoreload
    %autoreload 2
    from lib.utils import construct_kernel
    from lib.utils import compute_kernel_kmeans_EM
    from lib.utils import compute_kernel_kmeans_spectral
    from lib.utils import compute_purity
    from lib.utils import construct_knn_graph
    from lib.utils import compute_ncut
    from lib.utils import compute_pcut
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")
    import community # Louvain algorithm
    import networkx as nx


Two-moon dataset
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Load two-circle dataset
    mat = scipy.io.loadmat('datasets/two_circles.mat')
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print('n,d,nc:',n,d,nc)
    
    plt.figure(1)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Cgt, cmap='jet')
    plt.title('Distribution of two circle distributions -- Non-linear data points')
    plt.show()


Question 1: Evaluate the Louvain technique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How many “optimal” clusters does the Louvain method identify?

What is the clustering accuracy achieved by the Louvain solution?

Can you provide an explanation for the high accuracy?

.. code:: ipython3

    # Run Louvain algorithm
    W = construct_knn_graph(X, 50, 'euclidean_zelnik_perona')
    Wnx = nx.from_numpy_array(W)
    partition = community.best_partition(Wnx)
    nc_louvain = len(np.unique( [partition[nodes] for nodes in partition.keys()] ))
    n = len(Wnx.nodes())
    print('nb_data:', n , ', nb_clusters=', nc_louvain)
    
    # Extract clusters
    Clouv = np.zeros([n])
    clusters = []
    k = 0
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        Clouv[list_nodes] = k
        k += 1
        clusters.append(list_nodes)
        
    # Accuracy
    acc = compute_purity(Clouv,Cgt,nc_louvain)
    print('accuracy_louvain=',acc,' with nb_clusters=',nc_louvain)
    
    plt.figure(2)
    size_vertex_plot = 10
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n), c=Clouv, cmap='jet')
    plt.title('Louvain solution')
    plt.show()


USPS dataset
~~~~~~~~~~~~

Question 2: Compare the Louvain and spectral NCut solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the clustering results of the Louvain method with those of the
spectral NCut technique, using the same number of clusters.

How does the performance of the Spectral NCut technique change as the
number of clusters increases?

.. code:: ipython3

    # Load USPS dataset
    mat = scipy.io.loadmat('datasets/USPS.mat')
    W = mat['W']
    n = W.shape[0]
    Cgt = mat['Cgt']-1; Cgt=Cgt.squeeze()
    nc = len(np.unique(Cgt))
    print('n,nc:',n,nc)


.. code:: ipython3

    # Random partitionning
    Crand = np.random.randint(0,nc,[n])
    acc = compute_purity(Crand,Cgt,nc)
    print('Random solution:', str(acc)[:5])
    
    # Run NCut
    Cncut, acc = compute_ncut(W,Cgt,nc) 
    print('NCut solution:', str(acc)[:5])


.. code:: ipython3

    # Run Louvain
    Wnx = nx.from_numpy_array(W.toarray())
    partition = community.best_partition(Wnx)
    nc_louvain = len(np.unique( [partition[nodes] for nodes in partition.keys()] ))
    n = len(Wnx.nodes())
    print('nb_data:', n , ', nb_clusters=', nc_louvain)
    
    # Extract clusters
    Clouv = np.zeros([n])
    clusters = []
    k = 0
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        Clouv[list_nodes] = k
        k += 1
        clusters.append(list_nodes)
        
    # Accuracy
    acc = compute_purity(Clouv,Cgt,nc_louvain)
    print('Louvain solution ',str(acc)[:5],' with nb_clusters=',nc_louvain)


.. code:: ipython3

    # Run NCut with the number of clusters found by Louvain
    Cncut, acc = compute_ncut(W,Cgt,nc_louvain)
    print('NCut solution:',str(acc)[:5],' with nb_clusters=',nc_louvain)



Lecture : Graph SVM
===================

Lab 01 : Standard/Linear SVM – Solution
---------------------------------------

Xavier Bresson, Guoji Fu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/04_Graph_SVM'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    from IPython.display import display, clear_output
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import compute_purity
    import warnings; warnings.filterwarnings("ignore")


Linearly separable data points
==============================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_linearSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['Cgt_train'] - 1; Cgt_train = Cgt_train.squeeze()
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


.. code:: ipython3

    # Plot
    plt.figure(figsize=(8,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 100
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_train, color=pyplot.jet())
    plt.title('Training Data')
    p2 = plt.subplot(122)
    size_vertex_plot = 100
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    plt.tight_layout()
    plt.show()


Question 1: Implement the linear SVM on linear separable data using the primal-dual iterative algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Hint:* Follow Page 18-20, Lecture 4 Slides

**Step 1:** Compute the Linear Kernel :math:`Ker` and :math:`L, Q`
defined as - :math:`Ker= XX^\top`, - :math:`L = \text{diag}(l)`, -
:math:`Q = LKL`.

You may use function ``np.diag()``, the transpose operator ``.T``, and
the matrix-matrix multiplication operator ``.dot()``.

.. code:: ipython3

    # Compute linear kernel, L, Q
    
    l = l_train
    
    ############################################################################
    # Your code start
    ############################################################################
    
    Ker = Xtrain.dot(Xtrain.T)
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    ############################################################################
    # Your code end
    ############################################################################


**Step 2:** Initialize :math:`\alpha^{k=0} = \beta^{k=0} = 0_n`.

You may use function ``np.zeros()`` for initializing a zero vector.

.. code:: ipython3

    # Initialization
    ############################################################################
    # Your code start
    ############################################################################
    
    alpha = np.zeros([n])
    beta = np.zeros([n])
    
    ############################################################################
    # Your code end
    ############################################################################


**Step 3:** Choose the time steps :math:`\tau_\alpha, \tau_\beta` such
that :math:`\tau_\alpha\tau_\beta \leq \frac{1}{\|Q\| \cdot \|L\|}`.

Some feasible choices can be
:math:`\tau_\alpha = \frac{a}{\|Q\|}, \tau_\beta = \frac{b}{\|L\|}`,
where :math:`ab \leq 1`.

You may use ``np.linalg.norm()`` to compute the norm of a matrix.

Try to evaluate the performance of linear SVM with different choices of
time steps.

.. code:: ipython3

    # Time steps
    ############################################################################
    # Your code start
    ############################################################################
    
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    ############################################################################
    # Your code end
    ############################################################################


**Step 4:** Project alpha to :math:`[0, +\infty]` during the update of
alpha and beta with conjuguate gradient.

.. code:: ipython3

    # Run Linear SVM
    
    # Compute linear kernel, L, Q
    Ker = Xtrain.dot(Xtrain.T)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(Xtrain.dot(Xtest.T))
    
    # Initialization
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        
        # Projection of alpha on [0,+infty]
        ############################################################################
        # Your code start
        ############################################################################
    
        alpha[alpha<0.0] = 0
    
        ############################################################################
        # Your code here
        ############################################################################
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            plt.figure(figsize=(8,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^Tx+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^Tx+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            

Non-linearly separable data points
==================================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_twomoons_softSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['C_train_errors'] - 1; Cgt_train = Cgt_train.squeeze()
    Cgt_train[:250] = 0; Cgt_train[250:] = 1
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


.. code:: ipython3

    # Plot
    plt.figure(figsize=(10,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 33
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_train, color=pyplot.jet())
    plt.title('Training Data')
    p2 = plt.subplot(122)
    size_vertex_plot = 33
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    #plt.tight_layout()
    plt.show()


Question 2: Compute linear kernel, L, Q, time steps, initialization and projection of alpha as for Question 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the results with the linearly separable case and determine which
performs better.

Answer: Linear case is better

What strategy can be used to enhance the performance of SVM on
non-linearly separable data?

Answer: Kernel SVM, graph SVM

.. code:: ipython3

    # Run Linear SVM
    
    # Compute linear kernel, L, Q
    Ker = Xtrain.dot(Xtrain.T)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(Xtrain.dot(Xtest.T))
    
    # Initialization
    alpha = np.zeros([n])
    beta = 0.0
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^Tx+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^Tx+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            


Lecture : Graph SVM
===================

Lab 02 : Soft-Margin SVM – Solution
-----------------------------------

Xavier Bresson, Guoji Fu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/04_Graph_SVM'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd


.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    from IPython.display import display, clear_output
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import compute_purity
    from lib.utils import compute_SVM
    import warnings; warnings.filterwarnings("ignore")


Linearly separable data points
==============================

.. code:: ipython3

    # Data matrix X = linearly separable data points
    mat = scipy.io.loadmat('datasets/data_softSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['C_train_errors'] - 1; Cgt_train = Cgt_train.squeeze()
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


.. code:: ipython3

    # Plot
    plt.figure(figsize=(8,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 100
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_train, color=pyplot.jet())
    plt.title('Training Data with 25% ERRORS')
    p2 = plt.subplot(122)
    size_vertex_plot = 100
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    plt.tight_layout()
    plt.show()


Question 1: Evaluate the performance of soft-margin SVM on linearly separable data with different error parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How does the performance of a soft-margin SVM change on training and
testing data as the error parameter increases?

Answer: As the error parameter increases, the SVM becomes less tolerant
of misclassifications. A higher error parameter puts more emphasis on
minimizing the classification errors, meaning the SVM will try harder to
classify every point correctly, which could lead to overfitting. While
this may improve the performance on the training set, it can reduce the
model’s ability to generalize to unseen data, leading to worse
performance on the test set.

.. code:: ipython3

    # Error parameter
    ############################################################################
    # Your code start
    ############################################################################
    
    lamb = 0.01 # acc: 92.5%
    lamb = 0.1 # acc: 97.5% 
    
    ############################################################################
    # Your code end
    ############################################################################


Question 2: Project :math:`\alpha` to :math:`[0, \lambda]` during the update of alpha and beta with conjuguate gradient.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Run soft-margin SVM
    
    # Compute linear kernel, L, Q
    Ker = Xtrain.dot(Xtrain.T)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(Xtrain.dot(Xtest.T))
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.ones([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)  
        
        # Projection of alpha on [0, \lambda]
        ############################################################################
        # Your code start
        ############################################################################
    
        alpha[alpha<0.0] = 0 # Projection on [0, +infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        ############################################################################
        # Your code end
        ############################################################################
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            plt.figure(figsize=(8,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^Tx+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^Tx+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            

Non-linearly separable data points
==================================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_twomoons_softSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['C_train_errors'] - 1; Cgt_train = Cgt_train.squeeze()
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


.. code:: ipython3

    # Plot
    plt.figure(figsize=(10,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 33
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_train, color=pyplot.jet())
    plt.title('Training Data with 25% ERRORS')
    p2 = plt.subplot(122)
    size_vertex_plot = 33
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    #plt.tight_layout()
    plt.show()


Question 3: Evaluate the performance of soft-margin SVM on non-linearly separable data with different error parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the results with hard-margin Linear SVM, can significant
improvements in soft-margin linear SVM over hard-margin linear SVM be
achieved by tuning the error parameter on non-linearly separable data?

Answer: No. Although soft-margin SVM allows some misclassifications for
a more flexible decision boundary, it still produces a linear boundary.
Since both the training and testing data are non-linearly separable, a
non-linear decision boundary is needed to effectively separate the
classes.

.. code:: ipython3

    # Run soft-margin SVM
    
    # Compute linear kernel, L, Q
    Ker = Xtrain.dot(Xtrain.T)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(Xtrain.dot(Xtest.T))
    
    # Error parameter
    ############################################################################
    # Your code start
    ############################################################################
    
    lamb = 0.001 # acc: 80.4%
    lamb = 0.01 # acc: 81%
    lamb = 0.1 # acc: 81.6%
    lamb = 1 # acc: 82.6% 
    
    ############################################################################
    # Your code end
    ############################################################################
    
    # Initialization
    alpha = np.zeros([n])
    beta = 0.0
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^Tx+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^Tx+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            


Lecture : Graph SVM
===================

Lab 03 : Kernel/Non-Linear SVM – Solution
-----------------------------------------

Xavier Bresson, Guoji Fu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/04_Graph_SVM'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd


.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import compute_purity
    from lib.utils import compute_SVM
    import warnings; warnings.filterwarnings("ignore")
    import sklearn.metrics.pairwise


Non-linearly separable data points
==================================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_twomoons_kernelSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['C_train_errors'] - 1; Cgt_train = Cgt_train.squeeze()
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


.. code:: ipython3

    # Plot
    plt.figure(figsize=(10,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 33
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_train, color=pyplot.jet())
    plt.title('Training Data with 25% ERRORS')
    p2 = plt.subplot(122)
    size_vertex_plot = 33
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    #plt.tight_layout()
    plt.show()


Run soft-margin SVM
===================

.. code:: ipython3

    # Run soft-margin SVM
    
    # Compute linear kernel, L, Q
    Ker = Xtrain.dot(Xtrain.T)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(Xtrain.dot(Xtest.T))
    
    # Error parameter
    lamb = 0.1 # acc: 83%
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^Tx+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^Tx+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            


Run kernel SVM
==============

Question 1: Calculate the distance of each pair data points and compute the Gaussian kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gaussian kernel is defined as :
:math:`K_{i,j} = \exp({\frac{\|x_i - x_j\|^2}{2\sigma^2}})` between a
pair of data points :math:`(i,j)`.

You may use function
``sklearn.metrics.pairwise.pairwise_distances(X, Y, metric='euclidean', n_jobs=1)``
to compute the euclidean distance between all vector pairs
:math:`\|x_i - x_j\|^2`.

Hint: You may consider :math:`\sigma=0.5`.

.. code:: ipython3

    ############################################################################
    # Your code start
    ############################################################################
    
    # The Euclidean distance of pair training data points
    train_Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    
    # The Euclidean distance of pair data points between training data and testing data
    test_Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    
    # Compute Gaussian kernel
    sigma = 0.5;
    sigma2 = sigma**2
    Ker = np.exp(- train_Ddist**2 / sigma2)
    KXtest = np.exp(- test_Ddist**2 / sigma2)
    
    ############################################################################
    # Your code end
    ############################################################################


Question 2: Evaluate the performance of Kernel SVM on non-linearly separable data with different error parameters.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can kernel SVM outperform soft-margin linear SVM on non-linearly
separable data?

Answer: Yes, kernel SVM can outperform soft-margin linear SVM on
non-linearly separable data. While soft-margin linear SVM uses a linear
decision boundary, kernel SVM transforms the data into a
higher-dimensional space, allowing it to capture complex, non-linear
relationships between classes. This makes kernel SVM more effective for
datasets that are not linearly separable, often leading to better
performance.

.. code:: ipython3

    ############################################################################
    # Your code start
    ############################################################################
    
    # Error parameter
    lamb = 3 # acc: 95.4
    
    ############################################################################
    # Your code end
    ############################################################################


.. code:: ipython3

    # Run kernel SVM
    
    # Compute Gaussian kernel, L, Q
    sigma = 0.5; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(KXtest)
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^T\phi(x)+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^T\phi(x)+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            


Real-world graph of articles
============================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_20news_50labels.mat')
    Xtrain = mat['Xtrain']
    l_train = mat['l'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()


Run linear SVM
==============

Question 3: Run linear SVM on the real-world data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find the value ``lamb`` that maximizes the accuracy for the test set.

.. code:: ipython3

    ############################################################################
    # Your code start
    ############################################################################
    
    # Error parameter
    lamb = 3 # acc: 83.5
    
    ############################################################################
    # Your code end
    ############################################################################


.. code:: ipython3

    # Run linear SVM
    
    # Compute Gaussian kernel, L, Q
    Ker = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    KXtest = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(KXtest)
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Print
            print('Linear SVM, iter, diff_alpha, acc :',str(k),str(diff_alpha)[:7],str(accuracy_test)[:5])
            


Run kernel SVM
==============

Question 4: Evaluate the performance of kernel SVM on the real-world data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the results with soft-margin linear SVM.

What are the implications of kernel SVM outperforming soft-margin linear
SVM on real-world data?

Answer: It implies that the real-world data has complex, non-linear
patterns that a linear model cannot capture.

.. code:: ipython3

    ############################################################################
    # Your code start
    ############################################################################
    
    # Error parameter
    lamb = 3 # acc: 87.5
    
    ############################################################################
    # Your code end
    ############################################################################


.. code:: ipython3

    # Run kernel SVM
    
    # Compute Gaussian kernel, L, Q
    sigma = 0.5; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(KXtest)
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%10) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Print
            # print('iter, diff_alpha',str(k),str(diff_alpha)[:7])
            # print('acc',str(accuracy_test)[:5])
    
    print('Kernel SVM  iter, diff_alpha :',str(k),str(diff_alpha)[:7])
    print('            acc :',str(accuracy_test)[:5])



Lecture : Graph SVM
===================

Lab 04 : Graph SVM – Solution
-----------------------------

Xavier Bresson, Guoji Fu
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/04_Graph_SVM'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd


.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import compute_purity
    from lib.utils import compute_SVM
    from lib.utils import construct_knn_graph
    from lib.utils import graph_laplacian
    import warnings; warnings.filterwarnings("ignore")
    import sklearn.metrics.pairwise


.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_twomoons_graphSVM.mat')
    Xtrain = mat['Xtrain']
    Cgt_train = mat['Cgt_train'] - 1; Cgt_train = Cgt_train.squeeze()
    l_train = mat['l'].squeeze()
    nb_labeled_data_per_class = mat['nb_labeled_data_per_class'].squeeze()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    nc = len(np.unique(Cgt_train))
    print(n,d,nc)
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()
    print('l_train:',l_train)
    print('number of labeled data per class:',nb_labeled_data_per_class)
    print('number of unlabeled data:',n-2*nb_labeled_data_per_class)


.. code:: ipython3

    # Plot
    plt.figure(figsize=(12,4))
    p1 = plt.subplot(121)
    size_vertex_plot = 33
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=size_vertex_plot*np.ones(n), c=l_train, color=pyplot.jet())
    plt.title('Training Data: Labeled Data in red (first class)\n and blue (second class), \n and unlabeled Data in green (data geometry)')
    plt.colorbar()
    p2 = plt.subplot(122)
    size_vertex_plot = 33
    plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=Cgt_test, color=pyplot.jet())
    plt.title('Test Data')
    plt.colorbar()
    #plt.tight_layout()
    plt.show()


Run kernel SVM
==============

Observe the solution provided by kernel SVM with a limited number of 2 labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What was the accuracy when using a larger number of labels, i.e. code03?

Answer: 95.4%

.. code:: ipython3

    # Run kernel SVM
    
    # Compute Gaussian kernel, L, Q
    sigma = 0.5; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)
    l = l_train
    L = np.diag(l)
    Q = L.dot(Ker.dot(L))
    
    # Time steps
    tau_alpha = 10/ np.linalg.norm(Q,2)
    tau_beta = 0.1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Pre-compute J.K(Xtest) for test data
    LKXtest = L.dot(KXtest)
    
    # Error parameter
    lamb = 3 # acc: 73.8
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%100) or (diff_alpha<1e-3):
               
            # Indicator function of support vectors
            idx = np.where( np.abs(alpha)>0.25* np.max(np.abs(alpha)) )
            Isv = np.zeros([n]); Isv[idx] = 1
            nb_sv = len(Isv.nonzero()[0])
            
            # Offset
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(L.dot(alpha)) )/ nb_sv
            else:
                b = 0
                
            # Continuous score function
            f_test = alpha.T.dot(LKXtest) + b 
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^T\phi(x)+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^T\phi(x)+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            


Run Graph SVM
=============

.. code:: ipython3

    # Compute Gaussian kernel 
    sigma = 0.15; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)


Question 1 : Construct a KNN graph from training data and compute the Laplacian matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may use ``A = construct_knn_graph(data, k, dist_metric)`` to
generate the adjacency matrix of a KNN graph from a training dataset.

Usage: - ``data``: The features of the training data. - ``k``: The
number of nearest neighbors. - ``dist_metric``: The distance matric,
e.g., using the Euclidean distance ``'euclidean'``,
``'euclidean_zelnik_perona'`` or ``'cosine'``. - ``A``: The adjacency
matrix of the KNN graph.

You may consider ``Lap = graph_laplacian(A).todense()`` to obtain the
Laplacian matrix from an adjacency matrix ``A``.

.. code:: ipython3

    # Compute kNN graph
    ############################################################################
    # Your code start
    ############################################################################
    
    kNN = 10
    A = construct_knn_graph(Xtrain, kNN, 'euclidean')
    Lap = graph_laplacian(A).todense()
    
    ############################################################################
    # Your code end
    ############################################################################


Question 3: Compute the indicator function of labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Compute Indicator function of labels
    ############################################################################
    # Your code start
    ############################################################################
    
    H = np.zeros([n])
    H[np.abs(l_train)>0.0] = 1
    H = np.diag(H)
    
    ############################################################################
    # Your code end
    ############################################################################


Question 4: Compute L, Q for graph SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to Lecture 4 Page 50:

-  :math:`L = \text{Diag}(l)`
-  :math:`Q = LHK(I+\gamma \mathcal{L}K)^{-1}HL`

You may use these functions: \* ``np.diag()``: Diagonal matrix from a
vector. \* ``np.eye()``: Identity matrix. \* ``np.linalg.inv()``:
Inverse matrix.

.. code:: ipython3

    # Compute L, Q
    ############################################################################
    # Your code start
    ############################################################################
    
    gamma = 25 # weight of the graph loss
    l = l_train
    L = np.diag(l)
    T = np.eye(n) + gamma* Lap.dot(Ker)
    Tinv = np.linalg.inv(T)
    Q = L.dot(H.dot(Ker.dot(Tinv.dot(H.dot(L)))))
    
    ############################################################################
    # Your code end
    ############################################################################


.. code:: ipython3

    # Run Graph SVM
    
    # Time steps
    tau_alpha = 1/ np.linalg.norm(Q,2)
    tau_beta = 1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + np.eye(n)
    
    # Error parameter
    lamb = 1 # acc: 98.6
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 201
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%100) or (diff_alpha<1e-3):
            
            # xi vector
            xi = Tinv.dot(H.dot(L.dot(alpha)))
    
            # Offset
            idx_unlabeled_data = np.where( np.abs(l)<1./2 )
            alpha_labels = alpha; alpha_labels[idx_unlabeled_data] = 0
            idx = np.where( np.abs(alpha_labels)>0.25* np.max(np.abs(alpha_labels)) )
            Isv = np.zeros([n]); Isv[idx] = 1 # Indicator function of Support Vectors
            nb_sv = len(Isv.nonzero()[0])
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(np.squeeze(np.array(xi))) )/ nb_sv
            else:
                b = 0        
                
            # Continuous score function
            f_test = np.squeeze(np.array(xi.dot(KXtest) + b))
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            print('C_test',C_test.shape)
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Plot
            size_vertex_plot = 33
            plt.figure(figsize=(12,4))
            p1 = plt.subplot(121)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=f_test, color=pyplot.jet())
            plt.title('Score function $s(x)=w^T\phi(x)+b$ \n iter=' + str(k)+ ', diff_alpha=' + str(diff_alpha)[:7])
            plt.colorbar()
            p2 = plt.subplot(122)
            plt.scatter(Xtest[:,0], Xtest[:,1], s=size_vertex_plot*np.ones(n), c=C_test, color=pyplot.jet())
            plt.title('Classification function $f(x)=sign(w^T\phi(x)+b)$\n iter=' + str(k) + ', acc=' + str(accuracy_test)[:5])
            #plt.tight_layout()
            plt.colorbar()
            plt.show()
            if k<num_iter-1:
                clear_output(wait=True)   
            

Real-world graph of articles
============================

Dataset has 10 labeled data and 40 unlabeled data
=================================================

.. code:: ipython3

    # Dataset
    mat = scipy.io.loadmat('datasets/data_20news_10labels_40unlabels.mat')
    Xtrain = mat['Xtrain']
    n = Xtrain.shape[0]
    l_train = mat['l'].squeeze()
    d = Xtrain.shape[1]
    Xtest = mat['Xtest']
    Cgt_test = mat['Cgt_test'] - 1; Cgt_test = Cgt_test.squeeze()
    nc = len(np.unique(Cgt_test))
    print(n,d,nc)
    num_labels = np.sum(np.abs(l_train)>0.0)
    print('l_train:',l_train)
    print('number of labeled data per class:',num_labels//2)
    print('number of unlabeled data:',n-num_labels)


Run Kernel SVM (no graph information)
=====================================

.. code:: ipython3

    # Run Kernel SVM (no graph information)
    
    # Compute Gaussian kernel 
    sigma = 0.5; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)
    
    # Compute kNN graph
    kNN = 5
    gamma = 0 # <= no graph information
    A = construct_knn_graph(Xtrain, kNN, 'cosine')
    Lap = graph_laplacian(A).todense()
    
    # Compute Indicator function of labels
    H = np.zeros([n])
    H[np.abs(l_train)>0.0] = 1
    H = np.diag(H)
    
    # Compute L, Q
    L = np.diag(l_train)
    l = l_train
    T = np.eye(n)
    T += gamma* Lap.dot(Ker) 
    Tinv = np.linalg.inv(T)
    Q = L.dot(H.dot(Ker.dot(Tinv.dot(H.dot(L)))))
    
    # Time steps
    tau_alpha = 1/ np.linalg.norm(Q,2)
    tau_beta = 1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + 1* np.eye(n)
    
    # Error parameter
    lamb = 100 
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 1001
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%100) or (diff_alpha<1e-3):
            
            # xi vector
            xi = Tinv.dot(H.dot(L.dot(alpha)))
    
            # Offset
            idx_unlabeled_data = np.where( np.abs(l)<1./2 )
            alpha_labels = alpha; alpha_labels[idx_unlabeled_data] = 0
            idx = np.where( np.abs(alpha_labels)>0.25* np.max(np.abs(alpha_labels)) )
            Isv = np.zeros([n]); Isv[idx] = 1 # Indicator function of Support Vectors
            nb_sv = len(Isv.nonzero()[0])
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(np.squeeze(np.array(xi))) )/ nb_sv
            else:
                b = 0        
                
            # Continuous score function
            f_test = xi.dot(KXtest) + b
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Print
            # print('iter, diff_alpha',str(k),str(diff_alpha)[:7])
            # print('acc',str(accuracy_test)[:5])
    
    print('Kernel SVM  iter, diff_alpha :',str(k),str(diff_alpha)[:7])
    print('            acc :',str(accuracy_test)[:5])


Run Graph SVM
=============

Question 5: Compare the results with kernel SVM and deduce the implications of graph SVM outperforming kernel SVM on real-world data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Answer: It suggests that the data has inherent relationships better
captured by a graph-based approach. Graph SVM, which uses connectivity
information between data points, make it well suited for data with
network-like structures (e.g., social networks, molecular data). This
implies that graph-based models can outperform traditional kernel
methods by better capturing complex dependencies between data points.

.. code:: ipython3

    # Run Graph SVM
    
    # Compute Gaussian kernel 
    sigma = 0.5; sigma2 = sigma**2
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtrain, metric='euclidean', n_jobs=1)
    Ker = np.exp(- Ddist**2 / sigma2)
    Ddist = sklearn.metrics.pairwise.pairwise_distances(Xtrain, Xtest, metric='euclidean', n_jobs=1)
    KXtest = np.exp(- Ddist**2 / sigma2)
    
    # Compute kNN graph
    kNN = 8
    gamma = 100
    A = construct_knn_graph(Xtrain, kNN, 'cosine')
    Lap = graph_laplacian(A).todense()
    
    # Compute Indicator function of labels
    H = np.zeros([n])
    H[np.abs(l_train)>0.0] = 1
    H = np.diag(H)
    
    # Compute L, Q
    L = np.diag(l_train)
    l = l_train
    T = np.eye(n)
    T += gamma* Lap.dot(Ker) 
    Tinv = np.linalg.inv(T)
    Q = L.dot(H.dot(Ker.dot(Tinv.dot(H.dot(L)))))
    
    # Time steps
    tau_alpha = 1/ np.linalg.norm(Q,2)
    tau_beta = 1/ np.linalg.norm(L,2)
    
    # For conjuguate gradient
    Acg = tau_alpha* Q + 1* np.eye(n)
    
    # Error parameter
    lamb = 1
    
    # Initialization
    alpha = np.zeros([n])
    beta = np.zeros([n])
    alpha_old = alpha
    
    # Loop
    k = 0
    diff_alpha = 1e6
    num_iter = 1001
    while (diff_alpha>1e-3) & (k<num_iter):
        
        # Update iteration
        k += 1
        
        # Update alpha
        # Approximate solution with conjuguate gradient
        b0 = alpha + tau_alpha - tau_alpha* l* beta
        alpha, _ = scipy.sparse.linalg.cg(Acg, b0, x0=alpha, tol=1e-3, maxiter=50)   
        alpha[alpha<0.0] = 0 # Projection on [0,+infty]
        alpha[alpha>lamb] = lamb # Projection on [-infty,lamb]
    
        # Update beta
        beta = beta + tau_beta* l.T.dot(alpha)
        
        # Stopping condition
        diff_alpha = np.linalg.norm(alpha-alpha_old)
        alpha_old = alpha
        
        # Plot
        if not(k%100) or (diff_alpha<1e-3):
            
            # xi vector
            xi = Tinv.dot(H.dot(L.dot(alpha)))
    
            # Offset
            idx_unlabeled_data = np.where( np.abs(l)<1./2 )
            alpha_labels = alpha; alpha_labels[idx_unlabeled_data] = 0
            idx = np.where( np.abs(alpha_labels)>0.25* np.max(np.abs(alpha_labels)) )
            Isv = np.zeros([n]); Isv[idx] = 1 # Indicator function of Support Vectors
            nb_sv = len(Isv.nonzero()[0])
            if nb_sv > 1:
                b = (Isv.T).dot( l - Ker.dot(np.squeeze(np.array(xi))) )/ nb_sv
            else:
                b = 0        
                
            # Continuous score function
            f_test = xi.dot(KXtest) + b
    
            # Binary classification function
            C_test = np.sign(f_test) # decision function in {-1,1}
            accuracy_test = compute_purity(0.5*(1+C_test),Cgt_test,nc) # 0.5*(1+C_test) in {0,1}
    
            # Print
            # print('iter, diff_alpha',str(k),str(diff_alpha)[:7])
            # print('acc',str(accuracy_test)[:5])
    
    print('Graph SVM  iter, diff_alpha :',str(k),str(diff_alpha)[:7])
    print('           acc :',str(accuracy_test)[:5])


Plot graph of test data points
==============================

.. code:: ipython3

    # Plot graph of test data points
    kNN = 8 
    A = construct_knn_graph(Xtest, kNN, 'cosine')
    print(type(A),A.shape)
    
    import networkx as nx
    A.setdiag(0) 
    A.eliminate_zeros()
    G_nx = nx.from_scipy_sparse_array(A)
    plt.figure(figsize=[40,40])
    nx.draw_networkx(G_nx, with_labels=True, node_color=np.array(C_test), cmap='jet')



Lecture : Recommendation on Graphs
==================================

Lab 01 : PageRank
-----------------

Xavier Bresson
~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/05_Recommendation'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    import scipy.sparse.linalg
    import warnings; warnings.filterwarnings("ignore")


Synthetic small graph
=====================

.. code:: ipython3

    # Data matrix 
    mat = scipy.io.loadmat('datasets/pagerank_synthetic.mat')
    W = mat['W']
    W = scipy.sparse.csr_matrix(W)
    Wref = W
    X = mat['X']
    n = X.shape[0]
    d = X.shape[1]
    E = mat['E']
    XE = mat['X2']
    print('num_nodes:',n)


.. code:: ipython3

    plt.figure(1)
    size_vertex_plot = 100
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*np.ones(n))
    plt.quiver(XE[:,0], XE[:,1], E[:,0], E[:,1], scale=1., units='xy') 
    plt.title('Visualization of the artificial WWW')
    plt.axis('equal')
    plt.axis('off') 
    plt.show()

.. code:: ipython3

    # Solve eigenproblem
    
    # vector of 1's
    e = np.ones([n,1])/n 
    one = np.ones([n,1])
    
    # Dumpling vector
    D = np.array(W.sum(axis=1),dtype='float32').squeeze()
    a_idx = np.zeros([n],dtype='int32')
    a_idx[np.where(D<1./2)] = 1
    a = (1.0* a_idx)[:,None]
    
    # Compute P = W D^{-1}
    invD = 1./(D+1e-10)
    invD[a_idx==1] = 0
    invD = np.diag(invD)
    W = Wref.todense()
    P = invD.dot(W).T
    
    # EVD
    alpha = 0.85
    start = time.time()
    Phat = alpha* P +  alpha* e.dot(a.T) + (1.0-alpha)* e.dot(one.T)
    Phat = scipy.sparse.csr_matrix(Phat)
    lamb, U = scipy.sparse.linalg.eigs(Phat, k=1, which='LM') 
    x_pagerank = np.abs(U[:,0])/ np.sum(np.abs(U[:,0]))
    
    # Computational time
    print('Computational time for PageRank solution with EIGEN Method (sec):',time.time() - start)

.. code:: ipython3

    plt.figure(2)
    size_vertex_plot = 1e3*6
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*x_pagerank)
    plt.quiver(XE[:,0], XE[:,1], E[:,0], E[:,1], scale=1., units='xy') 
    plt.title('PageRank solution with EIGEN Method.')
    plt.axis('equal')
    plt.axis('off') 
    plt.show()

.. code:: ipython3

    # PageRank values
    x = x_pagerank
    val = np.sort(x)[::-1] 
    idx = np.argsort(x)[::-1]
    index = np.array(range(1,1+n))
    in_degree = np.array(W.sum(axis=0)).squeeze(axis=0)
    out_degree =  np.array(W.sum(axis=1)).squeeze(axis=1)
    index = index[idx]
    in_degree = in_degree[idx]
    out_degree = out_degree[idx]
    print('\n  ''Node'' | ''PageRank'' | ''In-degree'' | ''Out-degree'' ')
    for i in range(n):
        print('   ',index[i], '  ', round(val[i],3) ,'      ', in_degree[i],'      ', out_degree[i], end='\n')

.. code:: ipython3

    # Power Method
    
    # Initialization
    x = e
    diffx = 1e10
    k = 0
    
    # Iterative scheme
    start = time.time()
    alpha = 0.85
    while (k<1000) & (diffx>1e-6):
        
        # Update iteration
        k += 1
    
        # Update x
        xold = x
        x = alpha* P.dot(x) + e.dot( alpha* a.T.dot(x) + (1.0-alpha) )
        
        # Stopping condition
        diffx = np.linalg.norm(x-xold,1)
        
    x_pagerank_PM = np.array(x).squeeze(axis=1)
    
    # Computational time
    print('Computational time for PageRank solution with POWER Method (sec):',time.time() - start)
    
    plt.figure(3)
    size_vertex_plot = 1e3*6
    plt.scatter(X[:,0], X[:,1], s=size_vertex_plot*x_pagerank)
    plt.quiver(XE[:,0], XE[:,1], E[:,0], E[:,1], scale=1., units='xy') 
    plt.title('PageRank solution with POWER Method.')
    plt.axis('equal')
    plt.axis('off') 
    plt.show()

Real-world dataset CALIFORNIA
=============================

.. code:: ipython3

    ###########################
    # California graph
    #   http://vlado.fmf.uni-lj.si/pub/networks/data/mix/mixed.htm
    #   This graph was constructed by expanding a 200-page response set 
    #   to a search engine query 'California'.
    ###########################
    
    network = np.loadtxt('datasets/california.dat')
    row = network[:,0]-1
    col = network[:,1]-1
    n = int(np.max(network))+1 # nb of vertices
    ne = len(row)
    print('nb of nodes=',n)
    print('nb of edges=',ne)
    
    # Create Adjacency matrix W
    data = np.ones([ne])
    W = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    Wref = W
    print(W.shape)
    
    # Plot adjacency matrix
    plt.figure(4)
    plt.spy(W,precision=0.01, markersize=1)
    plt.show()

.. code:: ipython3

    # Solve eigenproblem
    
    # vector of 1's
    e = np.ones([n,1])/n 
    one = np.ones([n,1])
    
    # Dumpling vector
    D = np.array(W.sum(axis=1),dtype='float32').squeeze()
    a_idx = np.zeros([n],dtype='int32')
    a_idx[np.where(D<1./2)] = 1
    a = (1.0* a_idx)[:,None]
    
    # Compute P = W D^{-1}
    invD = 1./(D+1e-10)
    invD[a_idx==1] = 0
    invD = np.diag(invD)
    W = Wref.todense()
    P = invD.dot(W).T
    
    # EVD
    alpha = 0.85
    start = time.time()
    Phat = alpha* P +  alpha* e.dot(a.T) + (1.0-alpha)* e.dot(one.T)
    Phat = scipy.sparse.csr_matrix(Phat)
    lamb, U = scipy.sparse.linalg.eigs(Phat, k=1, which='LM') 
    x_pagerank = np.abs(U[:,0])/ np.sum(np.abs(U[:,0]))
    
    # Computational time
    print('Computational time for PageRank solution with EIGEN Method (sec):',time.time() - start)

.. code:: ipython3

    # Power Method
    
    # Initialization
    x = e
    diffx = 1e10
    k = 0
    
    # Iterative scheme
    start = time.time()
    alpha = 0.85
    while (k<1000) & (diffx>1e-6):
        
        # Update iteration
        k += 1
    
        # Update x
        xold = x
        x = alpha* P.dot(x) + e.dot( alpha* a.T.dot(x) + (1.0-alpha) )
        
        # Stopping condition
        diffx = np.linalg.norm(x-xold,1)
        
    x_pagerank_PM = np.array(x).squeeze(axis=1)
    
    # Computational time
    print('Computational time for PageRank solution with POWER Method (sec):',time.time() - start)


Lecture : Recommendation on Graphs
==================================

Lab 02 : Collaborative recommendation
-------------------------------------

Xavier Bresson
~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/05_Recommendation'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from IPython.display import display, clear_output
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import shrink
    import scipy.sparse.linalg
    import warnings; warnings.filterwarnings("ignore")


Synthetic dataset
=================

.. code:: ipython3

    # Load graphs of rows/users and columns/movies
    mat = scipy.io.loadmat('datasets/synthetic_netflix.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining) / (n*m)
    print('perc_obs_training=',perc_obs_training)


.. code:: ipython3

    # Viusalize the rating matrix
    plt.figure(1)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Low-rank Matrix M.\nNote: We NEVER observe it\n in real-world applications')
    plt.show()
    
    plt.figure(2)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M\n for TRAINING.\n Percentage=' + str(perc_obs_training))
    plt.show()


.. code:: ipython3

    # Collaborative filtering / low-rank approximation by nuclear norm
    
    # Norm of the operator
    OM = O*M
    normOM = np.linalg.norm(OM,2)
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario : very low number of ratings, 0.03%, error metric = 138.75
    lambdaNuc = normOM/4; lambdaDF = 1e3 * 1e-2
    
    
    # Indentify zero columns and zero rows in the data matrix X
    idx_zero_cols = np.where(np.sum(Otraining,axis=0)<1e-9)[0]
    idx_zero_rows = np.where(np.sum(Otraining,axis=1)<1e-9)[0]
    nb_zero_cols = len(idx_zero_cols)
    nb_zero_rows = len(idx_zero_rows)
       
    # Initialization
    X = M; Xb = X;
    Y = np.zeros([n,m])
    normA = 1.
    sigma = 1./normA
    tau = 1./normA
    diffX = 1e10
    min_nm = np.min([n,m])
    k = 0
    while (k<2000) & (diffX>1e-1):
        
        # Update iteration
        k += 1
            
        # Update dual variable y
        Y = Y + sigma* Xb
        U,S,V = np.linalg.svd(Y/sigma)
        Sdiag = shrink( S , lambdaNuc/ sigma )
        I = np.array(range(min_nm))
        Sshrink = np.zeros([n,m])
        Sshrink[I,I] = Sdiag
        Y = Y - sigma* U.dot(Sshrink.dot(V))    
        
        # Update primal variable x
        Xold = X
        X = X - tau* Y
        X = ( X + tau* lambdaDF* O* M)/ (1 + tau* lambdaDF* O)
        # Fix issue with no observations along some rows and columns
        r,c = np.where(X>0.0); median = np.median(X[r,c])
        if nb_zero_cols>0: X[:,idx_zero_cols] = median
        if nb_zero_rows>0: X[nb_zero_rows,:] = median
    
        # Update primal variable xb
        Xb = 2.* X - Xold
            
        # Difference between two iterations
        diffX = np.linalg.norm(X-Xold)
            
        # Reconstruction error
        err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
        
        # Plot
        if not k%50:
            clear_output(wait=True)
            plt.figure(1)
            plt.imshow(X, interpolation='nearest', cmap='jet')
            plt.title('Collaborative Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
            plt.show()        
            print('diffX',diffX)
    
    
    clear_output(wait=True) 
    print('Reconstruction Error: '+ str(round(err_test,5)))
    
    # Final plot
    plt.figure(2)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Ground truth low-rank matrix M')
    
    plt.figure(3)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M')
    
    plt.figure(4)
    plt.imshow(X, interpolation='nearest', cmap='jet')
    plt.title('Collaborative Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,2)))
    plt.show()
    



Real-world dataset SWEETRS
==========================

.. code:: ipython3

    # Load graphs of rows/users and columns/products
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario1.mat')
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario2.mat')
    # mat = scipy.io.loadmat('datasets/real_sweetrs_scenario3.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    print('M', M.shape)
    print('Otraining', Otraining.shape)
    print('Otest', Otest.shape)
    print('Wrow', Wrow.shape)
    print('Wcol', Wcol.shape)
    
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining) / (n*m)
    print('perc_obs_training=',perc_obs_training)
    perc_obs_test = np.sum(Otest) / (n*m)


.. code:: ipython3

    # Visualize the original rating matrix
    plt.figure(1,figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    
    # Visualize the observed rating matrix
    plt.figure(2, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()


.. code:: ipython3

    # Collaborative filtering / low-rank approximation by nuclear norm
    
    # Norm of the operator
    OM = O*M
    normOM = np.linalg.norm(OM,2)
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario 1 : low number of ratings, 1.3%, error metric = 744.10
    lambdaNuc = normOM/4; lambdaDF = 1e3 * 1e-2
    
    # scenario 2 : intermediate number of ratings, 13.1%, error metric = 412.01
    lambdaNuc = normOM/4 * 1e2; lambdaDF = 1e3 * 1e0
    
    # scenario 3 : large number of ratings, 52.7%, error metric = 698.97
    # lambdaNuc = normOM/4 * 1e2; lambdaDF = 1e3 
    
    
    # Indentify zero columns and zero rows in the data matrix X
    idx_zero_cols = np.where(np.sum(Otraining,axis=0)<1e-9)[0]
    idx_zero_rows = np.where(np.sum(Otraining,axis=1)<1e-9)[0]
    nb_zero_cols = len(idx_zero_cols)
    nb_zero_rows = len(idx_zero_rows)
       
    # Initialization
    X = M; Xb = X;
    Y = np.zeros([n,m])
    normA = 1.
    sigma = 1./normA
    tau = 1./normA
    diffX = 1e10
    min_nm = np.min([n,m])
    k = 0
    while (k<2000) & ( diffX>1e-1 or k<100 ) :
        
        # Update iteration
        k += 1
            
        # Update dual variable y
        Y = Y + sigma* Xb
        U,S,V = np.linalg.svd(Y/sigma)
        Sdiag = shrink( S , lambdaNuc/ sigma )
        I = np.array(range(min_nm))
        Sshrink = np.zeros([n,m])
        Sshrink[I,I] = Sdiag
        Y = Y - sigma* U.dot(Sshrink.dot(V))    
        
        # Update primal variable x
        Xold = X
        X = X - tau* Y
        X = ( X + tau* lambdaDF* O* M)/ (1 + tau* lambdaDF* O)
        # Fix issue with no observations along some rows and columns
        r,c = np.where(X>0.0); median = np.median(X[r,c])
        if nb_zero_cols>0: X[:,idx_zero_cols] = median
        if nb_zero_rows>0: X[nb_zero_rows,:] = median
    
        # Update primal variable xb
        Xb = 2.* X - Xold
            
        # Difference between two iterations
        diffX = np.linalg.norm(X-Xold)
            
        # Reconstruction error
        err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
        
        # Plot
        if not k%50:
            clear_output(wait=True)   
            plt.figure(figsize=(10,10))
            plt.imshow(X, interpolation='nearest', cmap='jet', aspect=0.1)
            plt.colorbar(shrink=0.65)
            plt.title('Collaborative Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
            plt.show()
            print('diffX',diffX)
    
    clear_output(wait=True) 
    print('Reconstruction Error: '+ str(round(err_test,5)))
      
    
    # Final plots
    plt.figure(2, figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    plt.show()
    
    plt.figure(3, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()
    
    plt.figure(4, figsize=(10,10))
    plt.imshow(X, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Collaborative Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
    plt.show()
    



Lecture : Recommendation on Graphs
==================================

Lab 03 : Content recommendation
-------------------------------

Xavier Bresson
~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/05_Recommendation'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import shrink
    from lib.utils import graph_laplacian
    import scipy.sparse.linalg
    import warnings; warnings.filterwarnings("ignore")
    from lib.utils import compute_ncut, reindex_W_with_classes, construct_knn_graph
    import torch
    import networkx as nx


Synthetic dataset
=================

.. code:: ipython3

    # Load graphs of rows/users and columns/movies
    mat = scipy.io.loadmat('datasets/synthetic_netflix.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining) / (n*m)
    print('perc_obs_training=',perc_obs_training)


.. code:: ipython3

    # Viusalize the rating matrix
    plt.figure(1)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Low-rank Matrix M.\nNote: We NEVER observe it\n in real-world applications')
    plt.show()
    
    plt.figure(2)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M\n for TRAINING.\n Percentage=' + str(perc_obs_training))
    plt.show()


.. code:: ipython3

    # Content Filtering / Graph Regularization by Dirichlet Energy
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario : very low number of ratings, 0.03%, error metric = 161.32
    lambdaDir = 1e-1; lambdaDF = 1e3; alpha = 0.02
    
    
    # Compute Graph Laplacians
    Lr = graph_laplacian(Wrow)
    Lc = graph_laplacian(Wcol)
    I = scipy.sparse.identity(m, dtype=Lr.dtype)
    Lr = scipy.sparse.kron( I, Lr )
    Lr = scipy.sparse.csr_matrix(Lr)
    I = scipy.sparse.identity(n, dtype=Lc.dtype)
    Lc = scipy.sparse.kron( Lc, I )
    Lc = scipy.sparse.csr_matrix(Lc)
    
    # Pre-processing
    L = alpha* Lc + (1.-alpha)* Lr 
    vecO = np.reshape(O.T,[-1]) 
    vecO = scipy.sparse.diags(vecO, 0, shape=(n*m, n*m) ,dtype=L.dtype)
    vecO = scipy.sparse.csr_matrix(vecO) 
    At = lambdaDir* L + lambdaDF* vecO 
    vecM = np.reshape(M.T,[-1])
    bt = lambdaDF* scipy.sparse.csr_matrix( vecM ).T
    bt = np.array(bt.todense()).squeeze()
    
    # Solve by linear system
    x,_ = scipy.sparse.linalg.cg(At, bt, x0=bt, tol=1e-9, maxiter=100)
    X = np.reshape(x,[m,n]).T
     
    # Reconstruction error
    err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
    print('Reconstruction Error: '+ str(round(err_test,5)))
    
    # Plot
    plt.figure(2)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Ground truth low-rank matrix M')
    
    plt.figure(3)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M')
    
    plt.figure(4)
    plt.imshow(X, interpolation='nearest', cmap='jet')
    plt.title('Content Filtering\nReconstruction Error= '+ str(round(err_test,5)))
    plt.show()



Real-world dataset SWEETRS
==========================

.. code:: ipython3

    # Load graphs of rows/users and columns/products
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario1.mat')
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario2.mat')
    # mat = scipy.io.loadmat('datasets/real_sweetrs_scenario3.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    print('M', M.shape)
    print('Otraining', Otraining.shape)
    print('Otest', Otest.shape)
    print('Wrow', Wrow.shape)
    print('Wcol', Wcol.shape)
    
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining)/(n*m)
    print('perc_obs_training=',perc_obs_training)
    perc_obs_test = np.sum(Otest) / (n*m)


.. code:: ipython3

    # Visualize the original rating matrix
    plt.figure(1,figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    
    # Visualize the observed rating matrix
    plt.figure(2, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()


.. code:: ipython3

    # Visualize graph of users and graph of products
     # Plot adjacency matrix w.r.t. NCut communities
    
    # plot graph of users
    W = Wrow
    nc = 10; Cncut, _ = compute_ncut(W, np.zeros(Mgt.shape[0]), nc)# compute NCut clustering 
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W,Cncut)
    plt.figure(1)
    plt.spy(reindexed_W_ncut, precision=0.01, markersize=1)
    plt.title('Adjacency matrix of users indexed \naccording to the NCut communities')
    plt.show()
    A = W.copy()
    A.setdiag(0) 
    A.eliminate_zeros()
    G_nx = nx.from_scipy_sparse_array(A)
    plt.figure(2,figsize=[30,30])
    nx.draw_networkx(G_nx, with_labels=True, node_color=np.array(Cncut), cmap='jet')
    
    # plot graph of products
    W = Wcol
    nc = 10; Cncut, _ = compute_ncut(W, np.zeros(Mgt.shape[1]), nc)# compute NCut clustering 
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W,Cncut)
    plt.figure(3)
    plt.spy(reindexed_W_ncut, precision=0.01, markersize=1)
    plt.title('Adjacency matrix of products indexed \naccording to the NCut communities')
    plt.show()
    A = W.copy()
    A.setdiag(0) 
    A.eliminate_zeros()
    G_nx = nx.from_scipy_sparse_array(A)
    plt.figure(4,figsize=[30,30])
    nx.draw_networkx(G_nx, with_labels=True, node_color=np.array(Cncut), cmap='jet')
    


.. code:: ipython3

    # Content Filtering / Graph Regularization by Dirichlet Energy
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario 1 : low number of ratings, e.g. 1.3%, error metric = 399.89
    lambdaDir = 1e-1; lambdaDF = 1e3; alpha = 0.02
    
    # scenario 2 : intermediate number of ratings, e.g. 13.1%, error metric = 411.24
    lambdaDir = 1e-1; lambdaDF = 1e3; alpha = 0.02
    
    # scenario 3 : large number of ratings, e.g. 52.7%, error metric = 748.52
    # lambdaDir = 1e-1; lambdaDF = 1e3; alpha = 0.02
    
    
    # Compute Graph Laplacians
    Lr = graph_laplacian(Wrow)
    Lc = graph_laplacian(Wcol)
    I = scipy.sparse.identity(m, dtype=Lr.dtype)
    Lr = scipy.sparse.kron( I, Lr )
    Lr = scipy.sparse.csr_matrix(Lr)
    I = scipy.sparse.identity(n, dtype=Lc.dtype)
    Lc = scipy.sparse.kron( Lc, I )
    Lc = scipy.sparse.csr_matrix(Lc)
    
    # Pre-processing
    L = alpha* Lc + (1.-alpha)* Lr 
    vecO = np.reshape(O.T,[-1]) 
    vecO = scipy.sparse.diags(vecO, 0, shape=(n*m, n*m) ,dtype=L.dtype)
    vecO = scipy.sparse.csr_matrix(vecO) 
    At = lambdaDir* L + lambdaDF* vecO 
    vecM = np.reshape(M.T,[-1])
    bt = lambdaDF* scipy.sparse.csr_matrix( vecM ).T
    bt = np.array(bt.todense()).squeeze()
    
    # Solve by linear system
    x,_ = scipy.sparse.linalg.cg(At, bt, x0=bt, tol=1e-9, maxiter=100)
    X = np.reshape(x,[m,n]).T
     
    # Reconstruction error
    err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
    print('Reconstruction Error: '+ str(round(err_test,5)))
    
    # Plots
    plt.figure(2, figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    plt.show()
    
    plt.figure(3, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()
    
    plt.figure(4, figsize=(10,10))
    plt.imshow(X, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Content Filtering\nReconstruction Error= '+ str(round(err_test,5)))
    plt.show()



Lecture : Recommendation on Graphs
==================================

Lab 04 : Hybrid recommendation
------------------------------

Xavier Bresson
~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/05_Recommendation'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    from IPython.display import display, clear_output
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import shrink
    from lib.utils import graph_laplacian
    import scipy.sparse.linalg
    import warnings; warnings.filterwarnings("ignore")


Synthetic dataset
=================

.. code:: ipython3

    # Load graphs of rows/users and columns/movies
    mat = scipy.io.loadmat('datasets/synthetic_netflix.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining) / (n*m)
    print('perc_obs_training=',perc_obs_training)


.. code:: ipython3

    # Viusalize the rating matrix
    plt.figure(1)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Low-rank Matrix M.\nNote: We NEVER observe it\n in real-world applications')
    plt.show()
    
    plt.figure(2)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M\n for TRAINING.\n Percentage=' + str(perc_obs_training))
    plt.show()


.. code:: ipython3

    # Hybrid system : Matrix Completion on graphs
    
    # Norm of the operator
    OM = O*M
    normOM = np.linalg.norm(OM,2)
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario : very low number of ratings, 0.03%, error metric = 112.68
    lambdaDir = 1e-1 * 1e0; lambdaDF = 1e3; lambdaNuc = normOM/4; alpha = 0.1
    
    #Compute Graph Laplacians
    Lr = graph_laplacian(Wrow)
    Lc = graph_laplacian(Wcol)
    I = scipy.sparse.identity(m, dtype=Lr.dtype)
    Lr = scipy.sparse.kron( I, Lr )
    Lr = scipy.sparse.csr_matrix(Lr)
    I = scipy.sparse.identity(n, dtype=Lc.dtype)
    Lc = scipy.sparse.kron( Lc, I )
    Lc = scipy.sparse.csr_matrix(Lc)
    
    # Indentify zero columns and zero rows in the data matrix X
    idx_zero_cols = np.where(np.sum(Otraining,axis=0)<1e-9)[0]
    idx_zero_rows = np.where(np.sum(Otraining,axis=1)<1e-9)[0]
    nb_zero_cols = len(idx_zero_cols)
    nb_zero_rows = len(idx_zero_rows) 
     
    # Pre-processing
    L = alpha* Lc + (1.-alpha)* Lr 
    vecO = np.reshape(O.T,[-1]) 
    vecO = scipy.sparse.diags(vecO, 0, shape=(n*m, n*m) ,dtype=L.dtype)
    vecO = scipy.sparse.csr_matrix(vecO) 
    At = lambdaDir* L + lambdaDF* vecO 
    vecM = np.reshape(M.T,[-1])
    bt = lambdaDF* scipy.sparse.csr_matrix( vecM ).T
    bt = np.array(bt.todense()).squeeze()
    Id = scipy.sparse.identity(n*m)
    Id = scipy.sparse.csr_matrix(Id) 
    
    # Initialization
    X = M; Xb = X;
    Y = np.zeros([n,m])
    normA = 1.
    sigma = 1./normA
    tau = 1./normA
    diffX = 1e10
    min_nm = np.min([n,m])
    k = 0
    while (k<2000) & (diffX>1e-1):
        
        # Update iteration
        k += 1
            
        # Update dual variable y
        Y = Y + sigma* Xb
        U,S,V = np.linalg.svd(Y/sigma) # % Y/sigma = U*S*V'
        Sdiag = shrink( S , lambdaNuc/ sigma )
        I = np.array(range(min_nm))
        Sshrink = np.zeros([n,m])
        Sshrink[I,I] = Sdiag
        Y = Y - sigma* U.dot(Sshrink.dot(V))    
        
        # Update primal variable x
        Xold = X
        X = X - tau* Y
        A = tau* At + Id
        vecX = np.reshape(X.T,[-1]) 
        vecX = scipy.sparse.csr_matrix(vecX) 
        b = tau* bt + vecX
        b = np.array(b).squeeze()    
        
        # Solve by linear system
        x,_ = scipy.sparse.linalg.cg(A, b, x0=b, tol=1e-6, maxiter=25)
        X = np.reshape(x,[m,n]).T
        # Fix issue with no observations along some rows and columns
        r,c = np.where(X>0.0); median = np.median(X[r,c])
        if nb_zero_cols>0: X[:,idx_zero_cols] = median
        if nb_zero_rows>0: X[nb_zero_rows,:] = median
            
        # Update primal variable xb
        Xb = 2.* X - Xold
            
        # Difference between two iterations
        diffX = np.linalg.norm(X-Xold)
            
        # Reconstruction error
        err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
        
        # Plot
        if not k%10:
            clear_output(wait=True)
            plt.figure(1)
            plt.imshow(X, interpolation='nearest', cmap='jet')
            plt.title('Hybrid Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
            plt.show()        
            print('diffX',diffX)
    
    
    clear_output(wait=True) 
    print('Reconstruction Error: '+ str(round(err_test,5)))
    
    # Final plot
    plt.figure(2)
    plt.imshow(Mgt, interpolation='nearest', cmap='jet')
    plt.title('Ground truth low-rank matrix M')
    
    plt.figure(3)
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet')
    plt.title('Observed values of M')
    
    plt.figure(4)
    plt.imshow(X, interpolation='nearest', cmap='jet')
    plt.title('Hybrid Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,2)))
    plt.show()
    



Real-world dataset SWEETRS
==========================

.. code:: ipython3

    # Load graphs of rows/users and columns/products
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario1.mat')
    mat = scipy.io.loadmat('datasets/real_sweetrs_scenario2.mat')
    # mat = scipy.io.loadmat('datasets/real_sweetrs_scenario3.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    print('M', M.shape)
    print('Otraining', Otraining.shape)
    print('Otest', Otest.shape)
    print('Wrow', Wrow.shape)
    print('Wcol', Wcol.shape)
    
    n,m = M.shape
    print('n,m=',n,m)
    
    Mgt = M # Ground truth
    O = Otraining
    M = O* Mgt
    perc_obs_training = np.sum(Otraining)/(n*m)
    print('perc_obs_training=',perc_obs_training)
    perc_obs_test = np.sum(Otest) / (n*m)


.. code:: ipython3

    # Visualize the original rating matrix
    plt.figure(1,figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    
    # Visualize the observed rating matrix
    plt.figure(2, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()


.. code:: ipython3

    # Hybrid system : Matrix Completion on graphs
    
    # Norm of the operator
    OM = O*M
    normOM = np.linalg.norm(OM,2)
    
    #######################################
    # Select the set of hyper-parameters
    #######################################
    
    # scenario 1 : low number of ratings, e.g. 1.3%, error metric = 402.71
    lambdaDir = 1e-1 * 1e3 * 0.5; lambdaDF = 1e3; lambdaNuc = normOM/4; alpha = 0.02 
    
    # # scenario 2 : intermediate number of ratings, e.g. 13.1%, error metric = 397.47
    lambdaDir = 1e-1; lambdaDF = 1e3 * 10; lambdaNuc = normOM/4 /10; alpha = 0.25 
    
    # # # scenario 3 : large number of ratings, e.g. 52.7%, error metric = 695.00
    # lambdaDir = 1e-1 * 1e1; lambdaDF = 1e3; lambdaNuc = normOM/4; alpha = 0.02
    
    
    #Compute Graph Laplacians
    Lr = graph_laplacian(Wrow)
    Lc = graph_laplacian(Wcol)
    I = scipy.sparse.identity(m, dtype=Lr.dtype)
    Lr = scipy.sparse.kron( I, Lr )
    Lr = scipy.sparse.csr_matrix(Lr)
    I = scipy.sparse.identity(n, dtype=Lc.dtype)
    Lc = scipy.sparse.kron( Lc, I )
    Lc = scipy.sparse.csr_matrix(Lc)
    
    # Indentify zero columns and zero rows in the data matrix X
    idx_zero_cols = np.where(np.sum(Otraining,axis=0)<1e-9)[0]
    idx_zero_rows = np.where(np.sum(Otraining,axis=1)<1e-9)[0]
    nb_zero_cols = len(idx_zero_cols)
    nb_zero_rows = len(idx_zero_rows)
    
    # Pre-processing
    L = alpha* Lc + (1.-alpha)* Lr 
    vecO = np.reshape(O.T,[-1]) 
    vecO = scipy.sparse.diags(vecO, 0, shape=(n*m, n*m) ,dtype=L.dtype)
    vecO = scipy.sparse.csr_matrix(vecO) 
    At = lambdaDir* L + lambdaDF* vecO 
    vecM = np.reshape(M.T,[-1])
    bt = lambdaDF* scipy.sparse.csr_matrix( vecM ).T
    bt = np.array(bt.todense()).squeeze()
    Id = scipy.sparse.identity(n*m)
    Id = scipy.sparse.csr_matrix(Id) 
    
    # Initialization
    X = M; Xb = X;
    Y = np.zeros([n,m])
    normA = 1.
    sigma = 1./normA
    tau = 1./normA
    diffX = 1e10
    min_nm = np.min([n,m])
    k = 0
    while (k<2000) & (diffX>1e-1):
        
        # Update iteration
        k += 1
            
        # Update dual variable y
        Y = Y + sigma* Xb
        U,S,V = np.linalg.svd(Y/sigma) # % Y/sigma = U*S*V'
        Sdiag = shrink( S , lambdaNuc/ sigma )
        I = np.array(range(min_nm))
        Sshrink = np.zeros([n,m])
        Sshrink[I,I] = Sdiag
        Y = Y - sigma* U.dot(Sshrink.dot(V))    
        
        # Update primal variable x
        Xold = X
        X = X - tau* Y
        A = tau* At + Id
        vecX = np.reshape(X.T,[-1]) 
        vecX = scipy.sparse.csr_matrix(vecX) 
        b = tau* bt + vecX
        b = np.array(b).squeeze()    
        
        # Solve by linear system
        x,_ = scipy.sparse.linalg.cg(A, b, x0=b, tol=1e-6, maxiter=25)
        X = np.reshape(x,[m,n]).T
        # Fix issue with no observations along some rows and columns
        r,c = np.where(X>0.0); median = np.median(X[r,c])
        if nb_zero_cols>0: X[:,idx_zero_cols] = median
        if nb_zero_rows>0: X[nb_zero_rows,:] = median
            
        # Update primal variable xb
        Xb = 2.* X - Xold
            
        # Difference between two iterations
        diffX = np.linalg.norm(X-Xold)
            
        # Reconstruction error
        err_test = np.sqrt(np.sum((Otest*(X-Mgt))**2)) / np.sum(Otest) * (n*m)
        
        # Plot
        if not k%10:
            clear_output(wait=True)   
            plt.figure(figsize=(10,10))
            plt.imshow(X, interpolation='nearest', cmap='jet', aspect=0.1)
            plt.colorbar(shrink=0.65)
            plt.title('Hybrib Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
            plt.show()
            print('diffX',diffX)
    
    clear_output(wait=True) 
    print('Reconstruction Error: '+ str(round(err_test,5)))
    
    # Final plots
    plt.figure(2, figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*np.sum(Mgt>0)/(n*m))[:5])
    plt.show()
    
    plt.figure(3, figsize=(10,10))
    plt.imshow(Otraining*Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Observed rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_training)[:5])
    plt.show()
    
    plt.figure(4, figsize=(10,10))
    plt.imshow(X, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Hybrib Filtering\nIteration='+ str(k)+'\nReconstruction Error= '+ str(round(err_test,5)))
    plt.show()
    



Lecture : Recommendation on Graphs
==================================

Lab 05 : Preparing the real-world dataset SWEETRS
-------------------------------------------------

Xavier Bresson
~~~~~~~~~~~~~~

.. code:: ipython3

    # For Google Colaboratory
    import sys, os
    if 'google.colab' in sys.modules:
        # mount google drive
        from google.colab import drive
        drive.mount('/content/gdrive')
        path_to_file = '/content/gdrive/My Drive/CS5284_2024_codes/codes/05_Recommendation'
        print(path_to_file)
        # change current path to the folder containing "path_to_file"
        os.chdir(path_to_file)
        !pwd
        

.. code:: ipython3

    # Load libraries
    import numpy as np
    import scipy.io
    %matplotlib inline
    #%matplotlib notebook 
    from IPython.display import display, clear_output
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import time
    import sys; sys.path.insert(0, 'lib/')
    from lib.utils import shrink
    import scipy.sparse.linalg
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from lib.utils import compute_ncut, reindex_W_with_classes, construct_knn_graph
    from scipy.io import savemat
    from pandas import read_csv
    import networkx as nx


.. code:: ipython3

    # Real-world dataset SWEETRS (Sweet Recommender System)
    # https://paperswithcode.com/dataset/sweetrs
    # https://github.com/kidzik/sweetrs-analysis
    # https://raw.githubusercontent.com/kidzik/sweetrs-analysis/master/ratings-final.csv
    # 1,476 users submitted 33,692 grades for 77 candies, 39.2% of available ratings
    
    # ratings-final.csv => ratings-sweetrs.csv  
      # "product","user","value"
      # "Raffaello",1476,5
      # "Toblerone white",1476,2
      # "Haribo",1476,2
      # etc
    
    data = read_csv('datasets/ratings-sweetrs.csv')
    # converting data columns to lists
    product = data['product'].tolist()
    user = data['user'].tolist()
    value = data['value'].tolist()
    num_ratings = len(product)
    
    # Print first 100
    print('product[:100]:',product[:100])
    print('user[:100]:',user[:100])
    print('value[:100]:',value[:100])
    print('num_ratings:',num_ratings)


.. code:: ipython3

    # make a dictionary of products
    dictionary_product = []
    num_products = 0
    for item in product:
        if item not in dictionary_product:
            dictionary_product.append(item); num_products += 1
    print('dictionary_product:',dictionary_product,'\n')
    print('num_products (unique):',num_products,'\n') # 1476
    print('max(dictionary_product):',max(dictionary_product))
    product2index = { product:index for index,product in enumerate(dictionary_product) }
    index2product = { index:product for index,product in enumerate(dictionary_product) }
    print('product2index:', product2index,'\n')
    print('index2product:', index2product,'\n')
    
    # make a dictionary of users
    dictionary_user = []
    num_users = 0
    for item in user:
        if item not in dictionary_user:
            dictionary_user.append(item); num_users += 1
    print('dictionary_user[:10]:',dictionary_user[:10],'\n')
    print('num_users (unique):',num_users,'\n') # 1476
    user2index = { user:index for index,user in enumerate(dictionary_user) }
    index2user = { index:user for index,user in enumerate(dictionary_user) }


.. code:: ipython3

    # Compute the original rating matrix
    #  user ratings are {0,1,2,3,4,5}
    #  when no rating given, the value will be -1
    print('num_users, num_products, num_ratings:',num_users, num_products, num_ratings)
    perc_obs_rating = num_ratings / (num_users * num_products)
    print('percentage available ratings:', perc_obs_rating)
    print(num_users, num_products)
    Morig = -np.ones((num_users,num_products)) # initialize all ratings at -1
    print(Morig.shape)
    for idx in range(num_ratings):
        idx_product = product2index[product[idx]]
        idx_user = user2index[user[idx]]
        rating_value = value[idx]
        Morig[idx_user,idx_product] = rating_value
    
    # Visualize the rating matrix
    plt.figure(figsize=(10,10))
    plt.imshow(Morig, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Original rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_rating)[:5])


.. code:: ipython3

    # Processing original rating matrix
     # Remove users with less than 50% rating
     # Randomly shuffle the order of users
    Mgt = []
    M = torch.tensor(Morig)
    for idx_user in range(num_users):
        num_ratings_user = (M[idx_user,:]>0).sum()
        if num_ratings_user > 0.50 * num_products:
            Mgt.append(M[idx_user,:])
    Mgt = torch.stack(Mgt)
    num_users_new = Mgt.size(0)
    idx_suffle = torch.randperm(num_users_new)
    Mgt = Mgt[idx_suffle,:]
    print(Mgt.size())
    
    num_ratings_new = torch.nonzero(Mgt).size(0)
    perc_obs_rating_new = num_ratings_new / (num_users_new * num_products)
    print('num_users_new, num_products, num_ratings_new:',num_users_new, num_products, num_ratings_new)
    print('perc_obs_rating_new:', perc_obs_rating_new)
    
    # Visualize the new rating matrix
    plt.figure(figsize=(10,10))
    plt.imshow(Mgt, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('New rating matrix\n Percentage observed ratings: ' + str(100*perc_obs_rating_new)[:5])
    


.. code:: ipython3

    # Compute mask of training data
    #         mask of test data
    
    idx_nzeros = torch.nonzero(Mgt>0)
    print('position of ratings:',idx_nzeros.size())
    num_nzero = idx_nzeros.size(0)
    print('number of ratings:',num_nzero)
    
    # select the percentage of observed ratings in the original matrix for training and testing
    perc_train = 0.02 # scenario 1 : 2% 
    perc_train = 0.2  # scenario 2 : 20% 
    # perc_train = 0.8  # scenario 3 : 80%
    
    num_train_data = int(perc_train * num_nzero)
    num_test_data = num_nzero - num_train_data
    print('num_train_data:',num_train_data)
    print('num_train_data+num_test_data:',num_train_data+num_test_data)
    idx_randperm = torch.randperm(num_nzero)
    idx_train_data = idx_nzeros[idx_randperm[:num_train_data]]
    print('num_train_data:',len(idx_train_data))
    idx_test_data = idx_nzeros[idx_randperm[num_train_data:]]
    print('num_test_data:',len(idx_test_data))
    
    num_users, num_products = Mgt.size()
    print('num_users, num_products:',num_users, num_products)
    Otraining = torch.zeros(num_users, num_products)
    for idx in idx_train_data:
        Otraining[idx[0],idx[1]] = 1
    print('number of train ratings:',torch.nonzero(Otraining).size(0))
    Otest = torch.zeros(num_users, num_products)
    for idx in idx_test_data:
        Otest[idx[0],idx[1]] = 1
    print('number of test ratings:',torch.nonzero(Otest).size(0))
    
    # Visualize the training mask
    plt.figure(figsize=(10,10))
    plt.imshow(Otraining, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Training mask\n Percentage training ratings: ' + str(100*torch.nonzero(Otraining).size(0)/num_nzero)[:5])
    
    # Visualize the test mask
    plt.figure(figsize=(10,10))
    plt.imshow(Otest, interpolation='nearest', cmap='jet', aspect=0.1)
    plt.colorbar(shrink=0.65)
    plt.title('Test mask\n Percentage training ratings: ' + str(100*torch.nonzero(Otest).size(0)/num_nzero)[:5])


.. code:: ipython3

    # Compute graph of users    / rows    
    #         graph of products / columns 
    
    # Graph of users 
    X = Mgt.numpy()
    
    # # Construct medium-quality graph with 50% of original ratings -- comment this part or not
    # # only select 50% of rating
    # idx_nzeros = torch.nonzero(Mgt>0)
    # num_nzero = idx_nzeros.size(0)
    # perc_rating = 0.5 
    # num_graph_data = int(perc_rating * num_nzero)
    # print('num_data, num_graph_data:', num_nzero, num_graph_data)
    # idx_randperm = torch.randperm(num_nzero)
    # idx_graph_data = idx_nzeros[idx_randperm[:num_graph_data]]
    # num_users, num_products = Mgt.size()
    # print('num_users, num_products:',num_users, num_products)
    # X = torch.zeros(num_users, num_products)
    # for idx in idx_graph_data:
    #     X[idx[0],idx[1]] = 1
    # print('num_graph_data:',len(idx_graph_data))
    # X = X.numpy()
    # # Compute k-NN graph with cosine distance
    # W = Wrow = construct_knn_graph(X, 10, 'cosine_binary') # best
    # # END Construct medium-quality graph with 50% of original ratings 
    
    # # Construct high-quality graph with 100% of original ratings -- comment this part or not
    # Compute k-NN graph with cosine distance
    W = Wrow = construct_knn_graph(X, 10, 'cosine') 
    # # END Construct high-quality graph with 100% of original ratings
    
    # Evaluate the graph construction by visualizing the adjacency matrix
    nc = 10
    Cncut, _ = compute_ncut(W, torch.zeros(Mgt.size(0)).numpy(), nc) # compute NCut clustering 
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W,Cncut)
    # plot adjacency matrix w.r.t. NCut communities
    plt.figure(1)
    plt.spy(reindexed_W_ncut, precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to the NCut communities')
    plt.show()
    # plot graph of users
    A = W.copy()
    A.setdiag(0) 
    A.eliminate_zeros()
    G_nx = nx.from_scipy_sparse_array(A)
    plt.figure(2,figsize=[30,30])
    nx.draw_networkx(G_nx, with_labels=True, node_color=np.array(Cncut), cmap='jet')
    
    # Graph of products 
    X = Mgt.transpose(1,0).numpy()
    # Compute k-NN graph with euclidean distance
    W = Wcol = construct_knn_graph(X, 15, 'cosine_binary')
    
    # Evaluate the graph construction by visualizing the adjacency matrix
    nc = 10
    Cncut, _ = compute_ncut(W, torch.zeros(Mgt.size(1)).numpy(), nc)
    [reindexed_W_ncut,reindexed_C_ncut] = reindex_W_with_classes(W,Cncut) # compute NCut clustering 
    # plot adjacency matrix w.r.t. NCut communities
    plt.figure(3)
    plt.spy(reindexed_W_ncut, precision=0.01, markersize=1)
    plt.title('Adjacency Matrix A indexed according to the NCut communities')
    plt.show()
    # plot graph of products
    A = W.copy()
    A.setdiag(0) 
    A.eliminate_zeros()
    G_nx = nx.from_scipy_sparse_array(A)
    plt.figure(4,figsize=[30,30])
    nx.draw_networkx(G_nx, with_labels=True, node_color=np.array(Cncut), cmap='jet')
    


.. code:: ipython3

    # save data
    M = Mgt.numpy()
    Otraining = Otraining.numpy()
    Otest = Otest.numpy()
    savemat('datasets/real_sweetrs.mat',{'M': M, 'Otraining': Otraining, 'Otest': Otest, 'Wrow': Wrow, 'Wcol': Wcol})
    
    # checking : load data
    mat = scipy.io.loadmat('datasets/real_sweetrs.mat')
    M = mat['M']
    Otraining = mat['Otraining']
    Otest = mat['Otest']
    Wrow = mat['Wrow']
    Wcol = mat['Wcol']
    print('M', M.shape)
    print('Otraining', Otraining.shape)
    print('Otest', Otest.shape)
    print('Wrow', Wrow.shape)
    print('Wcol', Wcol.shape)





