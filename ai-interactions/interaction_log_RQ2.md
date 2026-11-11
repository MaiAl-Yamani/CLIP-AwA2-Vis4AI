# Phase I - GenAI interaction record RQ 2
Zolboo Chimedregnzen - k12349108 

**Model:** CLIP ViT-B/16 (OpenAI)  

**Dataset:** Animals with Attributes 2 (AwA2) 

**RQ2:** Are animals with more shared visual attributes more likely to be misclassified as each other?

**GenAI:** ChatGPT (OpenAI)

## 1. Selecting visual attributes

**Prompt:** Review the AwA2 attributes and select those most relevant to visual appearance between animal classes. 

**AI suggestion:** Found 33 attributes out of 85 that define appearance, such as color, markings and body features.

**Decision:** Accepted.

**Output:** A binary attribute matrix containing 50 animal classes and 33 selected attributes. We also visualized how frequently the selected attributes occur across classes.

**Limitations:** Some excluded attributes may still be affecting classification errors.

## 2. Attribute similarity

**Prompt:** Calculate pairwise similarity between all 50 animal classes using their selected binary attributes. Suggest a suitable similarity measure and identify the most similar pairs.

**AI suggestion:** Use Jaccard similarity as the attributes are binary. It measures the proportion of shared positive attributes among all attributes present in either class.

**Decision:** Accepted. We calculated Jaccard similarity for all unique pairs of animal classes.

**Output:** 50x50 similarity matrix and table with 1,225 unique animal pairs. Similarity score between german shepherd and siamese cat was very high (0.825) despite the fact that they look very different.

## 3. Similarity scores

**Prompt:** Lets find out why the similarity score between german shepherd and siamese cat is very high.

**AI suggestion:** Check which attributes both animals share and whether common features contribute to the high similarity.

**Decision:** Accepted. 

**Output:** Both animals shared many general attributes, which contributed to the high score despite their different appearances (black', 'brown', 'gray', 'patches', 'furry' etc.). This shows that attribute similarity does not always match  similarity that is behold in our eyes.

**Limitations:** Jaccard treats all attributes as equal, although some are more useful for distinguishing animals than some common features.

## 4. Similarity heatmap

**Prompt:** Visualize the Jaccard similarity matrix and heatmaps for visualizing the patterns.

**AI suggestion:** Create two heatmaps: one using the original class order and another using hierarchical clustering to group animals with similar attributes.

**Decision:** Modified. Compared and kept only the clustered heatmap for simplicity.

**Output:** A clustered heatmap showing attribute similarity between all 50 animal classes.

## 5. Comparing attribute similarity with confusion

**Prompt:** Use the CLIP predictions from RQ1 to calculate confusion rates. Compare them with the Jaccard similarity scores and scatterplot them.

**AI suggestion:** Reuse the CLIP predictions from RQ1, calculate class-normalized confusion rates, and compare them with the similarity scores. Since confusion can differ depending on direction, calculate both directions before combining them.

**Decision:** Accepted. We calculated confusion rates for each direction and used their average as the confusion score for each animal pair.

**Output:** A scatterplot comparing Jaccard similarity and confusion rates for 1,225 animal pairs. We could already see pairs with very high rates, such as ox-buffalo and ox–cow.

## 6. Statistic testing

**Prompt:** Calculate the Spearman correlation between attribute similarity and model confusion.

**AI suggestion:** Use Spearman correlation and its p-value to evaluate the relationship between the two variables.

**Decision:** Refined. A class-label permutation test with 5,000 permutations was used.

**Output:** Weak positive correlation (Spearman rho 0.265), permutation p-value 0.0002 and histogram of the permutation test. It does not prove that shared attributes solely cause misclassifications. 

## 7. Misclassified images

**Prompt:** Visualization showing misclassified images of the animals from the most frequently confused animal pairs. Include the true and pred labels and errors.

**AI suggestion:** Create a gallery showing real misclassified images. An interactive tool could allow users to choose different animal pairs.

**Decision:** Modified. Static gallery.

**Output:** Images of two examples for each of 4 directed confusion pairs: blue whale - humpback whale, ox - cow, ox - buffalo and rat - mouse, with the true, predicted label and total number of errors for the pair.



