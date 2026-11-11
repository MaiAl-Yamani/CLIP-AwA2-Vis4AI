# Phase I 
Favour Dilichukwu Ozioma - RQ1

Zolboo Chimedregzen K12349108 - RQ2

**RQ1:** Which animal classes does CLIP confuse most frequently?


**RQ2:** Are animals with more shared visual attributes more likely to be misclassified as each other?
**GenAI:** ChatGPT (OpenAI)

As for RQ2, I used ChatGPT to help me with analythic methods and visualizations to solve the RQ. It was very useful for developing and planning the analysis step by step when I wanted to dig deeper in a result or compare different approaches. Unfortunately, not every suggestion was suitable, so I still had to evaluate them and make final decisions which made the most sense for the task.

####  Failure patterns
The standard Spearman test gave a very small p-value (4.7119e-21). I questioned if the significance test was appropriate to use since the same animal classes appeared in multiple pairs, hence the observations are not independent. So I decided to use a class-label permutation test with 5,000 permutations as a balance between accuracy and computation time.

Some suggestions were also more complicated than necessary. ChatGPT proposed an interactive visualization for misclassified images, but I decided that a static gallery was sufficient for the analysis and easier to present in 5 min presentation. 

#### Best practices
Compare different methods before landing on one - I compared suggested regular and clustered heatmap and kept the clustered because it made the similarity patterns easier to identify.

Question the suggested statistical methods -  standard Spearman significance test did not take into consideration the dependence between animal pairs. Checking this helped me choose a more appropriate test (class label permutance) and avoid reaching misleading conclusions.


