# Theory

The basis of acoustic tomography explored here follows the formulation provided in \citet{Vecherin2006}. The theory builds from an assumption that the travel time ($t_i$) of any acoustic signal through the atmosphere depends only on the length of the path traversed along the $i$\textsuperscript{th} ray and the group velocity of the acoustic signal, $u_i$, which is the sum of the local speed of sound and the projection of the ambient convection velocity onto the unit vector aligned with the path $\mathbf{r}_i$.

The travel time, $t_i$, along any signal path, $L_i$, can be expressed.

$$
    t_i = \int_{L_i} \frac{1}{u_i (\mathbf{r}_i)} dl
$$
where $i=1,...,I$ represents the index of the path and $I$ is the total number of travel paths considered ($I=64$ for the current application).

The local speed of sound is typically taken as the Laplace adiabatic sound speed, $c_L$, which is related to the acoustic virtual temperature, $T_{av}$,
$$
    c_L = \sqrt{\gamma R_a T_\text{av}}
$$

where $\gamma \approx 1.41$ is the ratio of specific heats and $R_a=287.058$ is the universal gas constant for dry air.

Decomposing the instantaneous fields of the speed of sound, temperature, and ambient atmospheric horizontal velocity components ($c_L$, $T_{av}$, $\tilde{u}$, and $\tilde{v}$, respectively) into spatially averaged values (denoted with the 0 subscript) and spatially heterogeneous fluctuations ($c, T, u, v = f(\mathbf{r},t)$) yields

$$
\begin{aligned}
    c_L(\mathbf{r}, t) &= c_0(t) + c(\mathbf{r}, t) \nonumber\\
    T_\text{av}(\mathbf{r}, t) &= T_0(t) + T(\mathbf{r}, t) \nonumber\\
    \tilde{u}(\mathbf{r}, t) &= u_0(t) + u(\mathbf{r}, t) \nonumber\\
    \tilde{v}(\mathbf{r}, t) &= v_0(t) + v(\mathbf{r}, t).
\end{aligned}
$$

Here, $\tilde{u}(\mathbf{r}, t$) and $\tilde{v}(\mathbf{r}, t$) are the components of the two-dimensional velocity vector and $\mathbf{r}$ represents the Cartesian coordinates of the locations within the tomographic area.   

The linearized expression for travel time of an acoustic emission is

$$
    t_i = \epsilon_i(t_0) + \frac{L_i}{c_0(t)} \left( 1 - \frac{u_0(t) \text{cos}~\phi_i + v_0(t) \text{sin}~\phi_i}{c_0(t)} \right) - \frac{1}{c_0^2(t)} \int_{L_i} \frac{c_0(t)}{2T_0(t)} T(\mathbf{r}, t)+ u(\mathbf{r}, t) \text{cos}~\phi_i + v(\mathbf{r}, t)\text{sin}~\phi_i ~dl,
$$

where $\epsilon_i(t_0)$ is the noise in the travel time and $\phi_i$ is the angle between the $i^\text{th}$ ray and the positive $x$ axis.  

## Mean Field Estimation

Estimates of the mean (spatially averaged) temperature and in-plane components of velocity ($T_0$, $u_0$, and $v_0$) are achieved by assuming a spatially homogeneous field.
With this assumption, Equation~\eqref{eq:tt_exp} reduces to

$$
    t_i = \frac{L_i}{c_0(t)} \left( 1 - \frac{u_0(t) \text{cos}~\phi_i + v_0(t) \text{sin}~\phi_i}{c_0(t)} \right)% - \frac{1}{c_0^2(t)}
$$

Equation~\eqref{eq:meanfield} can then be rewritten in matrix form:

$$
    \mathbf{Gf} = \mathbf{b}
$$

where the elements of the column vector $\mathbf{b}$ are known and are given by $b_i=t_i(t)/L_i$. The vector of unknown values has three elements, $\mathbf{f} = [1/c_0(t), u_0(t)/c_0^2(t), v_0(t)/c_0^2(t)]$, and the matrix $\mathbf{G}$ is a function of the orientation of each path vector with respect to the positive $x$-axis, oriented east-west in the current AT array:

$$
    \mathbf{G} = \begin{bmatrix}
        1 & -\text{cos}~\phi_1 & -\text{sin}~\phi_1 \\
        \vdots & \vdots & \vdots \\
        1 & -\text{cos}~\phi_I & -\text{sin}~\phi_I
    \end{bmatrix}
$$

The observed travel times of acoustic chirps along the paths $r_i$ provide an overdetermined linear system for the spatially averaged speed of sound and convective velocity.
Least-squares estimation resolves the components of $\mathbf{f}$ as

$$
    \mathbf{f} = (\mathbf{G}^T\mathbf{G})^{-1} \mathbf{G}^T\mathbf{b}
$$

resulting in estimates of $c_0(t)$, $u_0(t)$, and $v_0(t)$ at the temporal resolution of frame collection by the AT system---in this case, 2 hertz (Hz).


## Time-Dependent Stochastic Inversion

Once the spatial mean values of temperature, adiabatic sound speed, and wind velocity fields within a tomographic area are known, the column vector of data $d(t)$ obtained at time $t$ with elements is introduced as

$$
    d_i(t) = L_i [ c_0(t) - u_0(t) \text{cos}~\phi_i - v_0(t) \text{sin}~\phi_i] - c_0^2(t)t_i(t) + \xi_i(t)
$$

where the noise term $\xi_i(t)$ includes errors in travel time measurement $\epsilon_i(t)$ and errors in estimation of the associated spatially averaged temperature and velocity.
Using Equation~\eqref{eq:tt_exp}, the data vector can be expressed as $\mathbf{d}(t) = \mathbf{d}_0(t) + \mathbf{\xi}(t)$, where $ \mathbf{d}_0(t)$ is a vector of noise-free data expressed as

$$
    d_{0i}(t) = \int_{L_i} \left( \frac{c_0(t)}{2T_0(t)}T(\mathbf{r},t) + u(\mathbf{r},t)\text{cos}~\phi_i + v(\mathbf{r},t)\text{sin}~\phi_i \right) dl.
$$

The central problem of time-dependent stochastic inversion is to reconstruct the fluctuating fields of temperature and velocity, $T(\mathbf{r},t_0)$, $u(\mathbf{r},t_0)$, and $v(\mathbf{r},t_0)$, at a time $t_0$ with knowledge of the acoustic travel time and the spatial average temperature and velocity.
This is undertaken through the introduction of a vector of models ($\mathbf{m}$) for temperature and velocity at all measurement times along all paths:

$$
    \mathbf{m}(t_0) = \left[ T(\mathbf{r}_1, t_0)~;~ \dots ~;~  T(\mathbf{r}_J, t_0)~;~ u(\mathbf{r}_1, t_0)~;~ \dots ~;~  u(\mathbf{r}_J, t_0)~;~ v(\mathbf{r}_1, t_0)~;~ \dots ~;~  v(\mathbf{r}_J, t_0) \right],
$$

where $J$ is the number of spatial locations within the domain to be resolved.
Assuming that one can collect data $N+1$ times during a time interval with duration $\tau$, denoted as $N_\tau$, by emitting and detecting chirps within the tomographic area, one can form the vector $\mathbf{d}$ for each frame

$$
    \mathbf{d} = \left[ \mathbf{d}(t - N \tau)~;~ \mathbf{d}(t - N \tau+\tau)~;~\dots ~;~ \mathbf{d}(t) \right].
$$

Here, $\mathbf{d}$ is a vector containing $(N+1)$ column vectors $\mathbf{d}(t-n\tau)$, each of length $I$, and $n=0,1...,N$. $\tau$ represents equal time intervals at which data is collected.

Using the data vectors from Equation~\eqref{eq:datavec}, it is necessary to find a linear estimate $\hat{\mathbf{m}}(t_0)$ of the unknown models $\mathbf{m}(t_0)$ at each time $t_0$. 
Models are formed using observed travel times ahead of ($t_0 > t$), concurrent with ($t_0 = t$), or following ($t_0 < t$) any time of interest, $t_0$.
In the current formulation, models are sought as linearizations of the observed data:

$$
    \hat{\mathbf{m}}(t_0) = \mathbf{Ad},
$$

where the elements of matrix $\mathbf{A}$ are unknown and must be estimated. Matrix $\mathbf{A}$ contains linear coefficients $a_{jk}$, where $j=1,2,\dots,3J$ and $k = 1,2,\dots,(N+1)I$.
Discrepancies $\mathbf{\varepsilon}$ between the true and reconstructed temperature and velocity fields at time $t_0$ are written as

$$
    \mathbf{\varepsilon}_j =  \hat{\mathbf{m}}_j(t_0) - \mathbf{m}_j(t_0).
$$

Elements of the coefficient matrix $\mathbf{A}$ are chosen to minimize discrepancies $\mathbf{\varepsilon}_j$ in a mean-squared sense.

The matrix $\mathbf{A}$ that solves the problem does not depend on a particular structure of the models $\mathbf{m}$ and data $\mathbf{d}$:

$$
    \mathbf{A} = \mathbf{R}_{\mathbf{md}} \mathbf{R}_{\mathbf{dd}}^{-1},
$$

where $\mathbf{R}_{\mathbf{md}}$  and $\mathbf{R}_{\mathbf{dd}}$ are model-data and data-data covariance matrices. 
Since data and models have been chosen, covariance matrices have the following structure:

$$
\begin{aligned}
    \mathbf{R}_{\mathbf{md}} &=&  \left[ \mathbf{B}_{\mathbf{m}\mathbf{d}}(t_0, t-N\tau) , ~ \mathbf{B}_{\mathbf{m}\mathbf{d}}(t_0, t-N\tau + \tau),~\dots, ~ \mathbf{B}_{\mathbf{m}\mathbf{d}}(t_0, t) \right]\\
    \mathbf{R}_{\mathbf{dd}} &=&  \begin{bmatrix}
        \mathbf{B}_{\mathbf{dd}}(t - N\tau, t-N\tau) & \mathbf{B}_{\mathbf{dd}}(t - N\tau, t-N\tau + \tau) & \dots & \mathbf{B}_{\mathbf{dd}}(t - N\tau, t) \\
        \mathbf{B}_{\mathbf{dd}}(t - N\tau+\tau, t-N\tau) & \mathbf{B}_{\mathbf{dd}}(t - N\tau+\tau, t-N\tau + \tau) & \dots & \mathbf{B}_{\mathbf{dd}}(t - N\tau+\tau, t) \\
        \vdots & \vdots & \ddots & \vdots \\    
        \mathbf{B}_{\mathbf{dd}}(t, t-N\tau) & \mathbf{B}_{\mathbf{dd}}(t, t-N\tau + \tau) & \dots & \mathbf{B}_{\mathbf{dd}}(t, t)
    \end{bmatrix}
\end{aligned}
$$

Here, $\mathbf{B}_{\mathbf{md}}(t_1, t_2) = \langle \mathbf{m}(t_1) \mathbf{d}^T(t_2)\rangle$ is a covariance matrix of size $[3J,I]$ between the models at time $t_1$ and the data at time $t_2$. 
Similarly, $\mathbf{B}_{\mathbf{dd}}(t_1, t_2) = \langle \mathbf{d}(t_1) \mathbf{d}^T(t_2)\rangle$ is the covariance matrix of size $[I,I]$ between the data at times $t_1$ and $t_2$. 
Note that the matrix $R_{\mathbf{dd}}$ is symmetric by definition.
It is further possible to quantify error in the field reconstruction process directly as

$$
    \mathbf{R_{\epsilon\epsilon}} = \mathbf{R}_{\mathbf{mm}} - \mathbf{R}_{\mathbf{md}}\mathbf{R}_{\mathbf{dd}}^{-1}\mathbf{R}_{\mathbf{md}}^T,
$$

where $\mathbf{R}_{\mathbf{mm}} = \langle \mathbf{m}(t_0) \mathbf{m}^T(t_0)\rangle$ represents the covariance of the models at time $t_0$.

Since the optimal stochastic inverse operator $\mathbf{A}$ given by
Equation~\eqref{eq:Amatrix} is determined in terms of the matrices $\mathbf{R}_{\mathbf{md}}$ and $\mathbf{R}_{\mathbf{dd}}$, the formulation of each covariance matrix determines the fidelity and accuracy of the reconstructed temperature and velocity fields.
In the case of travel-time tomography, one can use the linear relationship expressed in Equation~\eqref{eq:data} to derive expressions for the covariance matrix $\mathbf{B}_{\mathbf{m}\mathbf{d}_0}(t_1, t_2)$ between the models at time $t_1$ and the noise-free data at time $t_2$:

$$
\begin{aligned}    
    \mathbf{B}_{\mathbf{m}_j\mathbf{d}_{0i}}(t_1, t_2) &=& \langle m_j(t_1) d_{0i}(t_2) \rangle\\
    &=& \int_{L_i} \left( \frac{c_0(t_2)}{2T_0(t_2)} \langle m_j(t_1) T(\mathbf{r},t_2) \rangle + \langle m_j(t_1) u(\mathbf{r},t_2)\rangle \text{cos}~\phi_i + \langle m_j(t_1) v(\mathbf{r},t_2)\rangle \text{sin}~\phi_i \right) dl\\
    &=& \begin{cases}
        \int_{L_i} \left( \frac{c_0(t_2)}{2T_0(t_2)}B_{TT}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \right) dl, & \text{if } 1 \le j \le J \\
        \int_{L_i} \left(B_{uu}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{cos}~\phi_i + B_{uv}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{sin}~\phi_i \right) dl, & \text{if } J+1 \le j \le 2J \\
        \int_{L_i} \left(B_{vu}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{cos}~\phi_i + B_{vv}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{sin}~\phi_i \right) dl, & \text{if } 2J+1 \le j \le 3J
  \end{cases}
\end{aligned}
$$

Note that the index $j$ indicates which field (either temperature, $T$, $x$-velocity, $u$, or $y$-velocity, $v$) is being considered, and that there are $J$ instances of each respective model.
Indices $i = 1,\dots,I$ indicate the index of the path being considered and $\mathbf{r} \in L_i$ are the path vectors.
Spatial-temporal covariance functions of the corresponding fields are denoted $B_{TT}$, $B_{uu}$, $B_{uv}$, $B_{vu}$, and $B_{vv}$, and $\mathbf{r}_j$ are the chosen spatial points within the tomographic area for which the sought fields are reconstructed.

Similarly, the covariance matrix $\mathbf{B}_{\mathbf{d}_0\mathbf{d}_0}(t_1, t_2)$ between the noise-free data at two times is provided by the expression

$$
    \begin{aligned}
    \mathbf{B}_{\mathbf{d}_{0i}\mathbf{d}_{0p}}(t_1, t_2) = & \iint_{L_i, L_p} \frac{c_0(t_1)c_0(t_2)}{4T_0(t_1)T_0(t_2)} B_{TT}(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) + B_{uu}(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) \text{cos}~\phi_i ~ \text{cos}~\phi_p + \\
    & B_{vv}(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) \text{sin}~\phi_i ~ \text{sin}~\phi_p + 
    B_{uv}(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) \text{cos}~\phi_i ~ \text{sin}~\phi_p + 
    B_{vu}(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) \text{sin}~\phi_i ~ \text{cos}~\phi_p  ~ dl~dl^\prime,
    \end{aligned}
$$

where indices $i,p = 1,\dots,I$ both indicate which paths are being considered and $\mathbf{r} \in L_i, \mathbf{r}^\prime \in L_p$ are the path vectors. 
Note that it is assumed that the covariance functions between the temperature and either component of velocity $B_{Tu}=B_{Tv}=0$.

With a further assumption that the fluctuating temperature and velocity are stationary, the covariance fields lose their explicit dependence on either of the times being considered, and instead rely only on the time separation

$$
    B(\mathbf{r}, t_1; \mathbf{r}^\prime, t_2) \Rightarrow B(\mathbf{r}, \mathbf{r}^\prime, \Delta t), \hspace{1cm} \Delta t = t_2 - t_1.

$$

Employing Taylor's frozen field hypothesis, one can assume that the fluctuating field does not evolve in time but is simply convected along the main flow direction at some advection velocity $\mathbf{U}$. 
Thus the formulation of the covariance functions may be further simplified as
$$
    B(\mathbf{r}, \mathbf{r}^\prime, \Delta t) \Rightarrow B^S(\mathbf{r}, \mathbf{r}^\prime - \mathbf{U} \Delta t),
$$

where the superscript $S$ implies that the covariance function is now purely spatial, and all time dependence of the function has been removed.

Finally, to describe the spatial covariance of the temperature and wind velocity fields within the tomographic area, the following Gaussian covariance functions were used:

$$
\begin{aligned}
    B^S_{TT} &= \sigma_T^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l_T^2}\right)\\
    B^S_{uu} &= \sigma_u^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right) \left( 1 - \frac{(y - y^\prime)^2}{l^2} \right)\\
    B^S_{vv} &= \sigma_v^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right)\left( 1 - \frac{(x - x^\prime)^2}{l^2} \right)\\
    B^S_{uv} &= \sigma_u \sigma_v \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right) \left( \frac{(x - x^\prime)(y - y^\prime)}{l^2} \right)\\
\end{aligned}
$$

where $\sigma_T$, $\sigma_u$, and $\sigma_v$ are the standard deviations of the corresponding fields, $l_T$ and $l$ are correlation lengths of temperature and velocity, and the vectors $\mathbf{r} = (x,y)$ and $\mathbf{r}^\prime = (x^\prime,y^\prime)$.
With the correlation functions in Equations~\eqref{eq:corr1} and \eqref{eq:corr2}, one can estimate the covariance matrices $\mathbf{R}_{\mathbf{md}_0}$ and $\mathbf{R}_{\mathbf{d}_0\mathbf{d}_0}$, and finally the coefficient matrix $\mathbf{A}$ using Equation~\eqref{eq:Amatrix}.
In the described TDSI algorithm, five parameters must be chosen or deduced: $\sigma_T$, $\sigma_u$, $\sigma_v$, $l_T$, and $l$.
All parameters may be estimated from the sonic anemometer installed on Station 8 of the AT array.
Note that the formulation in Equations~\eqref{eq:corr1} and \eqref{eq:corr2} employ a Gaussian description of the covariance for temperature and velocity fields.
Other formulations may be more physically representative of the field variance in the atmospheric boundary layer but may come at increased computational costs.
