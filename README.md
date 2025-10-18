![Logo](images/CSS_logo.png)

# CrystalSubstructureSearcher
*CrystalSubstructureSearcher* is an open-source program for analyzing crystal structures and identifying low-periodic substructures (layers, chains, and molecular fragments) within them.

## Overview

The discovery of new two-dimensional materials is vital for advancing electronics and quantum technologies. As most 2D materials originate from layered bulk structures, identifying exfoliable crystals is a critical first step.

## Key Features

- **Automated substructure identification**: Identifies 2D layers, 1D chains, and 0D molecular fragments within crystal structures
- **Graph-based analysis**: Constructs structure graphs using Voronoi tessellation to represent atomic connectivity
- **Bond strength characterization**: Allows for selection of bond strength descriptors (distance, Voronoi face area, solid angle, bond valence) as proxies for bond strength
- **Fragment charge estimation**: Analyzes electronegativity differences and bond valences to estimate charge distribution
- **Bond valence sum descriptors**: Computes comprehensive descriptors characterizing bond strength distribution

## Input/Output

- **Input**: CIF files containing crystal structure data
- **Output**: Identified substructures with their periodicities, bond strength descriptors, and charge estimates

## Algorithm
The program uses an iterative edge-removal algorithm that progressively breaks the weakest bonds (based on user-selected bond strength descriptor) and monitors changes in structural periodicity.

![Workflow](images/algo_example.png)

**Figure 1.** *The first iterations of the Ca2Sb structure (ICSD refcode 154) analysis leading to identification of the 2-periodic substructure. The bond valences are selected as edge weight in this example. The edges of the structure graph broken at each iteration are highlighted in red, their characteristics are shown in the table below. The edges restored at the final step are shown in green. BVS total is the sum of the bond valences of the retained edges in the graph components.*

## Usage examples

1. simple run

```python CSS_run.py -f .\examples\154_Ca2Sb.cif```

2. search for 1-p (chains) substructure

```python CSS_run.py -f .\examples\124_CuSCN.cif --target-periodicity 1```

3. increasing default vacuum space in the CIF file with identified 2-p (layers) substructure

```python CSS_run.py -f .\examples\100042_MoS2.cif --vacuum-space 25.0```

4. save slab (not just single layer) of identified substructures with specified thickness

```python CSS_run.py -f .\examples\351_KCeS2.cif --save-slab --min-slab-thickness 25.0```

5. example of the dependence of the identified 2-p substructure on the bond strength criteria

```
python CSS_run.py -f .\examples\25626_AgCrSe2.cif --bond-property BV
python CSS_run.py -f .\examples\25626_AgCrSe2.cif --bond-property SA
python CSS_run.py -f .\examples\25626_AgCrSe2.cif --bond-property A
```

6. batch run with parameters specified in the params.yaml file

```python CSS_batch_run.py```

7. parallel run with parameters specified in the params.yaml file

```python CSS_parallel_run.py -N 2 --folder examples -p params.yaml```