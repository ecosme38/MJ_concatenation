# Majority Judgement (MJ) with concatenation of multiple votes

The majority judgment voting system is described on wikipedia: <https://en.wikipedia.org/wiki/Majority_judgment>

I like the French version: <https://fr.wikipedia.org/wiki/Jugement_majoritaire>, or the web site of the Mieux Voter association: <https://mieuxvoter.fr/>

The tools provided here implement a majority judgment algorithm.

These tools **do not** provide a technical interface to collect the votes. This can be done with (free) online platforms such as the one provided by the previously cited association.

These tools allow to concatenate several votes into one single ranking. The different votes can be weighted with integer values. I do not know yet if this is consistent with the theory, but it is a practical way I found to get one ranking from different votes, e.g. based on multiple criteria, and it is the real added value of these tools.

Votes must be formatted in CSV files as described in the notebook.