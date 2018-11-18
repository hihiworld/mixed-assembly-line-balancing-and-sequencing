# mixed-assembly-line-balancing-and-scheduling
this is the ready for my first paper'code, which will docplex in python.
docplex is a tool for python user to utlize IBM CPLEX, but only support 12.8 version for local solve.

The title of this paper is 

Balancing and scheduling of flexible mixed model assembly lines
doi：10.1007/s10601-013-9142-6

the contributes of this paper is a CP model and a MIP model, i will write all of them in futrue.

In order to solve MIP model with huge ammount of instance, I use decomposition method, the scheme is 
A(mip) + P(Dispatching) + D(M or D)
