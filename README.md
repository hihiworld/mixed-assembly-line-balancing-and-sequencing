# mixed-assembly-line-balancing-and-scheduling
this is the ready work for my first paper'code, which will use docplex in python.
docplex is a tool for python user to utlize IBM CPLEX, but only support 12.8 version for local solve.

The title of this paper is 

Balancing and scheduling of flexible mixed model assembly lines

doi：10.1007/s10601-013-9142-6

the contributes of this paper is proposing a CP model and a MIP model, I will finish all of them in futrue.

In order to solve MIP model with huge ammount of instance, I will use decomposition method of this paper, the scheme is 
A(mip) + P(Dispatching) + J(M or D)

but maybe your CPLEX is promotional version, I use a small example which contains 5 tasks, 3 producitons, 3 stations. you can run comolete_model.py to check solution

I will add annotation as soon as possible.

If you feel useful, please give me a STAR.
