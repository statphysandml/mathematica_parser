Mathematica Parser: Translate mathematica expressions into any programming language
=================

The python library parses mathematical functions that are exported from mathematica. The mathematical operations are represented within the flow_equation_parser.py file as a tree. This tree can be used to generate code in any programming language to perform the same mathematical operation. We do this here on the example of a set of ordinary differential equations and convert the equation into Thrust code that can be run with CUDA on a GPU.

Note that based on the flow_equation_parser, the code can be straightforwardly extended to support other programming languages and types of equations.

Installation
-----------

So far, the library needs to be build locally. This can be done by running

```bash
cd path_to_mathematica_parser/
pip install --use-feature=in-tree-build .
```

Basic Example
-------------

We consider as and example a lorentz attractor:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align}&space;\dot{x}&=\sigma(y&space;-&space;x)&space;\nonumber\\&space;\dot{y}&=x(\rho-z)&space;-&space;y&space;\nonumber\\&space;\dot{z}&=xy&space;-&space;\beta&space;z\nonumber&space;\end{align}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;\dot{x}&=\sigma(y&space;-&space;x)&space;\nonumber\\&space;\dot{y}&=x(\rho-z)&space;-&space;y&space;\nonumber\\&space;\dot{z}&=xy&space;-&space;\beta&space;z\nonumber&space;\end{align}" title="\begin{align} \dot{x}&=\sigma(y - x) \nonumber\\ \dot{y}&=x(\rho-z) - y \nonumber\\ \dot{z}&=xy - \beta z\nonumber \end{align}" /></a>

with:

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma=10\text{,}\;\rho&space;=&space;28\text{,}\;\beta=8/3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma=10\text{,}\;\rho&space;=&space;28\text{,}\;\beta=8/3" title="\sigma=10\text{,}\;\rho = 28\text{,}\;\beta=8/3" /></a>

which results in a chaotic solution.

The equations are exported from the Mathematica file with the FullForm (see mathematica_notebooks/lorentz_attractor.nb). The resulting output is (see examples/flow_equations/lorentz_attractor):

```text
List[Equal[Derivative[1][x][k], Times[10.`, Plus[Times[-1, x[k]], y[k]]]], Equal[Derivative[1][y][k], Plus[Times[-1, y[k]], Times[x[k], Plus[28.`, Times[-1, z[k]]]]]], Equal[Derivative[1][z][k], Plus[Times[x[k], y[k]], Times[Rational[-8, 3], z[k]]]]]
```
Note that in this case k refers to the time.

By executing the following python code in the top-level directory of the repository:

```python
from mathematicaparser.odevisualization.generators import generate_equations
generate_equations("lorentz_attractor", "./examples/flow_equations/lorentz_attractor/")
```

the mathematicaparser.core.flow_euquation_parser.FlowEquationParser class generates a tree in a pandas dataframe out of the above expression (example for the first ordinary differential equation):

![alt text](doc/operation_tree.jpg)

Based on this tree, thrust cuda code is generated with the help of the modules of mathematicaparser.odevisualization

```c++
void LorentzAttractorFlowEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 10 * ((-1 * val1) + val2); });
}
```

The computation of the first ordinary differential equation can be recognized in the lambda expression of the flow equation of the code snippet.

Dependencies
------------

- numpy
- pandas

Projects using the pystatplottools library
------------------------------------------

- ODEVisualization (https://github.com/statphysandml/ODEVisualization)


Support and Development
----------------------

For bug reports/suggestions/complaints please file an issue on GitHub.

Or start a discussion on our mailing list: statphysandml@thphys.uni-heidelberg.de.
