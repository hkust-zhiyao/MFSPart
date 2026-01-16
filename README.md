# MFSPart

## Prerequisites
* **Eigen** library

## Dataset
All datasets are obtained from the open-source links provided by the following paper:
*TopoPart: a Multi-level Topology-Driven Partitioning Framework for Multi-FPGA Systems* (ICCAD 2021).

## Compilation
Execute the following commands to compile:

```bash
g++ -o MFSPart_Fixed MFSPart_Fixed.cpp -std=c++17 -I ~/research/partition/eigen-3.4.0/
```

## Executioin
To run the framework:

```bash
./MFSPart_Fixed <data_name>
```
