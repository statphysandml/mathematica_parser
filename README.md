Mathematica Parser: Translate mathematica into any programming language
=================

The python library parses mathematical functions that are exported from mathematica. The mathematical operations are represented within the flow_equation_parser.py file as a tree. This tree can be used to generate code in any programming language to perform the same mathematical operation. We do this here on the example of a set of ordinary differential equations and convert the equation into Thrust code that can be run with CUDA on a GPU.