# Using Embedding for Semantic Chunking

I needed this for a RAG search I am making, it was fun, besides that `merge_most_similar_adjacent` part, had to get some help from deepseek. That mergeing algorithm is not hard, but very tricky.
I would like to update this to give a binary tree of the merges, as that can be searched really well with very high quality results I bet.

## Neat properties
Here is a corrolation graph between all of the sentince vectors. You can see squares that represent diffrent topics along the diaginal. Super cool
![Image](/Figure.png)


The text used is from [https://www.reddit.com/r/shortstories/comments/1jicp3r/mf_the_end_of_the_world/]
