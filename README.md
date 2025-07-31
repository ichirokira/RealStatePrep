<div align="center">
<img width="300" src="https://pbs.twimg.com/profile_images/807108296796557316/3UtQWApG_400x400.jpg">&nbsp;

<a target="_blank" href="http://facebook.com/utsfeit"><img src="https://img.shields.io/badge/style--5eba00.svg?label=Facebook&logo=facebook&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/UTS_QSI"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <a target="_blank" href="http://instagram.com/utsengineeringandit"><img src="https://img.shields.io/badge/style--5eba00.svg?label=Instagram&logo=instagram&style=social"></a>
</div>
<div align="center">
    <br>
    <h1>Real State Preparation Benchmarks</h1>
    <br>
    🔥&nbsp; Developed by Tuyen Quang Nguyen, Gabe Waite, Alan Robertson, ...🔥
</div>

<br>
<hr>

## Overview
<br>

- **💡 Implementation**: We implement some SOTA state preparation algorithms, focusing on 1D Gaussian state.
- **💻 Resource Estimation**: We provide resource estimation of the algorithms

## Installation
We require Python 3.11

```
pip install -r requirement.txt
```

## Folder Structures

Our main implementations are in `ops` folder, while the experiments are in `experiments` folder. To test the code, following the following bash

### Inequality-based State Preparation 
```
python experiments/ineq_qrom.py --system_nqubits 2 --data_nqubits 2 --sigma 1.0 
```

### QSP-based State Preparation 
```
python experiments/qsp_state_prep.py --system_nqubits 2 --error 1e-6 --sigma 1.0 
```

A figure to compare the outcome of quantum algorithms versus the target function will be generated in `results` folder.