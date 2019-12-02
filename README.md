# graph-cut-image-segmentation
Graph cuts for interactive image segementation. Assignment of the computer vision course of ENPC.

*Objective*
--- 
The aim of the assignment is to perform image segmentation, identifying foreground from background on an image with coins. 
This is done through the energy minimization of an MRF (Markov Random Field). 
This is performed with the use of a mincut solver. 

*Implementation*
---
The user marks the foreground and the background of the image with left and right clicks. From there, the graph is built and mincut computation performed with the use of maxflow library. 
