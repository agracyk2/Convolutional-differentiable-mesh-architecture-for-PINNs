Convolutional-differentiable mesh architecture for PINNs
This repository contains a convolutional architecture that 
- takes an input function
- transforms that function into an output over a mesh
- allows a single point along that mesh to be extracted
- allows differentiability along that point with respect to the domain (mesh) and time, thus suitable for physics-informed applications.

In particular, we learn the mapping


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Psi_{\theta_{\Psi}}(\psi,\Omega,(i_1,...,i_d),t)\rightarrow(\tilde{\psi}_{t},[\tilde{\psi}_{t}]_{i_1,...,i_d})," alt="Eqn1">
</p>

or equivalently,


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Psi:\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{N}^{d}\times(\mathbb{R}^&plus;\cap\{0\})\times\Theta_{\Psi}\rightarrow\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{R}," alt="Eqn1">
</p>


where $\Psi$ is the neural network, $\psi$ is the input function, $\Omega$ is a mesh, $(i_1,...,i_d)$ is a specified index, $t$ is time, and $\tilde{\psi}$ is the solution.
  
The architecture thus outputs both the entire mesh at once as well as a single, specified point. This architecture is notable because differentiation in this process is traditionally nontrivial (i.e. we can bypass the error "One of the differentiated tensors has not been used in the graph"), and it prevents the need for concatenation over the entire grid (i.e. it is already done for us).

In this repository, our code learns the mapping from initial data of CIFAR-10. This can be changed quite flexibly. A more typical example of when to use this architecture, since it is suitable for PINN-type algorithms, is to map a PDE initial condition $\psi$ to the solution PDE over a grid, where we train the PINN using an output $\tilde{\psi}_{t=0}$ of the same initial condition $\psi$ at $t=0$.
