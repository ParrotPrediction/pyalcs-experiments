# rMPX Hashing Benchmark
The following experiment is designed to measure the performance of ALCS algorithms with large real-valued multiplexer environments.

## Flow
- The rRMPX returns a vector of numbers (attributes) ranging from 0 to 1.
- To maintain fixed size each attribute is hashed using chosen algorithm - `HASH_FUNC`.
- The _modulo_ operation with constant value `MODULO` is applied to reduce the output space

Pseudo-code below:
```python
import hashlib

import numpy as np

HASH_FUNC = 'sha256'
MODULO = 64

if __name__ == '__main__':
    for i in np.arange(0, 1, .1):
        h = hashlib.new(HASH_FUNC)
        h.update(str(i).encode('utf-8'))
        hash = int(h.hexdigest(), 16)
        output = hash % MODULO
        print(f"{i}:\t{output}")

```

## Verification
To estimate the performance of the algorithm the experiment is executed in _explore-exploit_ mode.

The experiment collect the following performance metrics:
- population size
- information whether prediction was successful
- relative trial execution time

## Hypothesis
**H1**: There is a positive correlation between the percentage of correct predictions and the `MODULO` value. (_Increasing `MODULO` increases the accuracy_)

**H2**: The population size stabilizes faster for smaller `MODULO` parameter. (_Compare the population size and average trial execution time in the course of the experiment_)

**H3**: The selection of hashing function `HASH_FUNC` has no significant impact on population size, performance or execution time.

**H4**: There is a negative correlation between metrics and the `MODULO` parameter when increasing problem size. (_Increasing problem space results negatively in all metrics_)

## MLflow

Useful Docker commands for working with the container image with experiments:

```shell
make docker_build_image
```
    

```shell
make run_docker PARAMS="-P trials=1000 -P rmpx-size=3 -P hash=md5 -P agent=acs2 -P modulo=4"
```
