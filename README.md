## Convolutional-differentiable mesh architecture for numerous PINNs
This repository contains a convolutional architecture that 
- takes an input function
- transforms that function into an output over a mesh
- allows a single point along that mesh to be extracted
- allows automatic differentiability along that point with respect to the domain (mesh) and time, thus suitable for physics-informed applications.

In particular, we learn the mapping


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Psi_{\theta_{\Psi}}(\psi,\Omega,u_{i_1,...i_d},t)\rightarrow(\tilde{\psi}_{t},\tilde{\psi}_{t}|_{u_{i_1,...,i_d}})," alt="Eqn1">
</p>

or equivalently,


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Psi:\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{R}^{d}\times(\mathbb{R}^&plus;\cap\{0\})\times\Theta_{\Psi}\rightarrow\mathbb{R}^{n_1\times...\times&space;n_d}\times\mathbb{R}," alt="Eqn1">
</p>


where $\Psi$ is the neural network, $\psi$ is the input function, $\Omega$ is a mesh, $(i_1,...,i_d)$ is a specified index, $t$ is time, and $\tilde{\psi}$ is the solution.


<div align="center">
<img src="https://github.com/user-attachments/assets/dfe4685e-e256-490b-858f-329fde45eec9" width="500">
</div>
  
The architecture thus outputs both the entire mesh at once as well as a single, specified point. This architecture is notable because differentiation in this process is traditionally nontrivial (i.e. we can bypass the error "One of the differentiated tensors has not been used in the graph"), and it prevents the need for concatenation over the entire grid (i.e. it is already done for us).

The reason differentiation with a similar architecture is nontrivial is we want the solution independent of the point to be differentiated (otherwise the learning task become significantly harder if there is a dependence), while simultaneously allowing differentiation via connecting the point through the computational graph, since there is a dependence on space and time input to allow automatic differentiation. This architecture conciliates this issue.

In this repository, our code learns the mapping from initial data of CIFAR-10. This can be changed quite flexibly. A more typical example of when to use this architecture, since it is suitable for PINN-type algorithms, is to map a PDE initial condition $\psi$ to the solution PDE over a grid, where we train the PINN using an output $\tilde{\psi}_{t=0}$ of the same initial condition $\psi$ at $t=0$.

We remark this architecture has close connections to physics-informed neural operators, as it has potential to map a discretized initial condition to a solution over numerous instances of data, and individual points can evaluate a PINN-type loss simultaneously; however, we do not learn an operator mapping here, and it is indeed between finite-dimensional vector spaces, thus it is not quite the same as a neural operator.
