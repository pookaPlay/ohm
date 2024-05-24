# OHM: Ordered Hypothesis Machines

Experiments with a novel type of neural network that has the potential to be more efficient for digital hardware - instead of using a weighted sum, OHM Neural Networks use a weighted order statistic. The Bit-Serial implementation of OHM Neural Networks (the current focus) reduces to a very simple data-flow architecture where data bit-streams alternate between least-significant-bit (LSB) and most-signifcant-bit (MSB) computation. 

python src/bls/test_OHM.py 

[Some explanation and background material](https://github.com/pookaPlay/ohm/wiki)


