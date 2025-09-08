[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/F2F01JB1KE)

# Hydraulic Modelling Toolkit for District Heating (Python)
An open-source Python package for steady-state hydraulic modelling of district heating (DH) networks. The toolkit builds incidence and loop matrices using graph theory and solves mass and energy balances through Newton–Raphson (NR) and Genetic Algorithm (GA) methods. It provides a transparent and reproducible framework for researchers and practitioners to analyse flow and pressure distributions in central hydronic networks of any scale.

## Methodology
District Heating networks are central hydronic systems that distribute thermal energy via supply and return pipelines. Hydraulic modelling is critical to ensure adequate pressure and flow at all substations, evaluate network extensions, integrate new heat sources, and support efficient operation. Key principles:

### Graph-Theoretical Formulation
A pipe-node list is used to construct incidence and loop matrices, providing a mathematical representation of the DH topology. Loop detection algorithms automatically generate the fundamental cycle basis.

### Mass and Energy Conservation
Flow distribution is solved by enforcing Kirchhoff-type mass balances at nodes and energy conservation in loops.

### Numerical Solvers
- Newton–Raphson (NR): A rapid iterative solver with proven convergence properties, suitable for large-scale networks.
- Genetic Algorithm (GA): A heuristic solver that explores the solution space, useful for non-linear or highly constrained cases.

> Scope & Aim: Provides steady-state flow and pressure distribution; as designed as a physical modelling tool (not a black-box), ensuring transparency and extensibility for further research.

## Requirements
Python ≥ 3.8 with (`pandas`, `numpy`, `scipy`, `geneticalgorithm`, `matplotlib`)

```
numpy>=1.24,<2.0
scipy>=1.10
pandas>=1.5
openpyxl>=3.1
geneticalgorithm==1.0.2
matplotlib>=3.7
```

## License
You are free to use, modify and distribute the code as long as **authorship is properly acknowledged**. Please reference this repository in derivative works.

## Citing
- Tol, Hİ. Development of a physical hydraulic modelling tool for District Heating systems. Energy and Buildings (2021) 253. https://doi.org/10.1016/j.enbuild.2021.111512

## Acknowledgements
Above all, I give thanks to **Allah, The Creator (C.C.)**, and honor His name **Al-‘Alīm (The All-Knowing)**.

This repository is lovingly dedicated to my parents who have passed away, in remembrance of their guidance and support.

I would also like to thank **ChatGPT (by OpenAI)** for providing valuable support in updating and improving the Python implementation.
