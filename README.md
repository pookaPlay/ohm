# OHM: Ordered Hypothesis Machines

A new approach to digital neuromorphic computing based on Bit-Serial OHM (Ordered Hypothesis Machines) Neural Networks. 

The OHM Neural Network Node (or neuron) replaces the popular "weighted sum" + "nonlinearity" with a mutlipler-free node that is, by design, better for digital hardware. 

The Bit-Serial implementation of OHM Neural Networks reduces to a very simple data-flow architecture where data bit-streams alternate between least-significant-bit (LSB) and most-signifcant-bit (MSB) computation. 

The LSB computation is analogous to the "weighted sum" in the traditional node. It can be considered a form of feature extraction, or embedding. It increases the bit-width (or precision) of the data in order to not loose information. That is, when adding a k-bit weight to a k-bit input, the output must be (k+1) bits to maintain full precision. 

The MSB computation is analogous to the "nonlinearity" in the traditional node. The OHM nonlinearity has multiple inputs and selects one of these inputs using a weighted order statistic. Depending on the data, the MSB computation can sometimes identify the unqiue input in fewer than k bits and stop early. 

