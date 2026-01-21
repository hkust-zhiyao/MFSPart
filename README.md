# MFSPart

## Prerequisites
* **Eigen** library

## Dataset
All datasets are obtained from the open-source links provided by the following paper:
*TopoPart: a Multi-level Topology-Driven Partitioning Framework for Multi-FPGA Systems* (ICCAD 2021).

## Compilation
Use the following command to compile the desired version (e.g., MFSPart_Fixed, MFSPart_Non-Fixed, etc.):

```bash
g++ -o <output_name> <source_file>.cpp -std=c++17 -I ~/research/partition/eigen-3.4.0/
```

## Execution
To run the framework:

```bash
./<output_name> <data_name>
```

## Acknowledgement
We gratefully acknowledge Dr. Benzheng Li for kindly answering our questions about his paper and providing valuable clarifications.
