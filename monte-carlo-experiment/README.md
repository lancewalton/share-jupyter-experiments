# Monte-Carlo Experiments

This project is about determining of using Monte-Carlo methods can help to predict share/index price movements.

Concretely, a Monte-Carlo method is defined as follows: for each day (indexed by n), we calculate the recent (N day) return history of a share/index. We then perform a large number of simulations whereby we start at day n's share price, randomly choose one of the returns from the history and apply that return to the share price to give us a predicted share price at day n+1. We repeat this T times to get a simulated future of the share price from day n+1 to day N+T. Having run many simulations for day n, we can then calculate some things:

* We can calculate envelopes for the future share price with varying degrees of confidence. For example, we can say that 99% of the simulations fall within a specific range (symmetrically by excluding the bottom and top 0.5%)  for each predicted day and we therefore can say that we have 99% confidence that over the T days of lookahead, the share price will fall within that envelope. We may choose other confidence intervals (e.g. 50%, 75%, 90%, 95%, 99%).
* In order to examine the value of these confidence intervals, we can examine the real share price from days n+1 to n+T and see how well it matches the confidence intervals - a regime change would be indicated by the more extreme values. By this means, we might be able to establish optimum history lengths and lookaheads.
* By running the Monte-Carlo simulation backwards from day n-1 to day n-T, we can see how well the simulations fit the actual data they were modelled on. This can tell us the degree to which we should trust this particular set of N returns on day n.
* We could combine several such sets of simulations with different history lengths, maybe using the backward simulation mechanism to tell us how valuable each history length is on a particular day and therefore how much weight we should give to it on a given day, in order to produce an aggregate prediction of the confidence intervals.
* We could either combine the above linearly, or maybe we could use a neural network to give us some non-linearity
* Maybe we could optimise the set of history lengths we use on any particular day by using the backward simulation approach and summing the 'confidence level' each real price reaches over the history and dividing by the number of days in the history. If that exceeds a tolerance, then the length of history is deemed unreliable and we should try longer or shorter histories.
* We could make the probability of choosing a particular return in a day in the history non-uniform - use an exponential decay, a triangular function, or a raised cosine. These might effect the way we calculate confidence intervals in either or both of the forward and backward simulations.

We have a bunch of time-series data we can use in ~/Projects/FTSEData.
