# ToricCASims
A repository for exploring cellular automaton decoders on high dimensional toric codes, based on Toom's rule and the sweep rule.

toric.py contains functions for building and indexing the cells in a d-dim square lattice with 
periodic boundary conditions, as well as constructing logical operators

ca_decoder.py contains functions for performing tooms rule cellular automaton in the presence of losses,
as well as a toom's rule erasure conversion. tooms_with_loss_parallelised() is the main function
to use to explore the performance of the ca decoder

sweep_rule.py contains implementations of the sweep rule decoder for either erasures or errors,
currently not both.

erasure_conversion.py contains functions to explore how different cellular automaton strategies perform
in 2d, including direction changes and erasure conversion. These strategies don't always translate nicely to higher dimensions.
The main function in this file explores how erasure conversion works in the cubic code, 3d with qubits on faces.

