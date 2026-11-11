# CLIP-AwA2-Vis4AI
**Project:**  CLIP classification of visually similar animals  
**Model:** CLIP ViT-B/16 (OpenAI)  https://github.com/mlfoundations/open_clip https://arxiv.org/abs/2103.00020  
**Dataset:** Animals with Attributes 2 (AwA2)   https://cvml.ista.ac.at/AwA2/ https://arxiv.org/abs/1707.00600

We will investigate how well the pretrained CLIP model distinguishes visually similar animal species using AwA2 dataset.
In the phase 1, we treated CLIP as a black-box model by analyzing its predictions on the selected AwA2 images.   

Phase I:  
We tried to answer the following questions:  
RQ1: Which animal classes does CLIP confuse most frequently? We analyzed classification errors and confusion patterns across 5000 images subset. The selected image filenames are provided in awa2_subset_5000.csv.  
RQ2: Are animals with more shared visual attributes more likely to be misclassified as each other? We focused on visually similar species and observed if shared visual attributes are associated with more frequent misclassifications.

GenAI-assisted: Zolboo Chimedregzen RQ2 (ChatGPT) & Favour Dilichukwu Ozioma RQ1 (Claude)
Human-only: Mai Elyamani (Data Visualization and Error Analysis) & Md Hassin Rahman Anoy (Dataset Preparation and CLIP Predictions)

## Phase I — Results and Findings

### Classification Performance

Both pairs evaluated pretrained CLIP ViT-B/16 on the same balanced subset of 5,000 AwA2 images, containing 100 images from each of the 50 animal classes.

- **Top-1 accuracy:** 92.08%
- **Correct predictions:** 4,604
- **Misclassifications:** 396
- **Top-5 accuracy (GenAI-assisted analysis):** 99.76%

### RQ1: Which animal classes does CLIP confuse most frequently?

The confusion matrix, per-class accuracy analysis, and misclassification gallery revealed that CLIP performs well overall but makes recurring errors between certain related animal classes.

The most frequent directional misclassifications were:

| True class | Predicted class | Errors |
|---|---|---:|
| Blue whale | Humpback whale | 67 |
| Ox | Buffalo | 26 |
| Ox | Cow | 24 |
| Rat | Mouse | 15 |
| Cow | Ox | 10 |
| Otter | Beaver | 10 |

Blue whale (28% accuracy) and ox (50% accuracy) were particularly challenging classes.

**Finding:** CLIP's errors are concentrated in specific animal classes, especially among visually or semantically related animals.

### RQ2: Are animals with more shared visual attributes more likely to be misclassified as each other?

Both pairs investigated the relationship between AwA2 attribute similarity and CLIP's classification confusion.

| Analysis | Human-only | GenAI-assisted |
|---|---|---|
| Attributes | All 85 binary attributes | 33 selected visual attributes |
| Similarity metric | Cosine similarity | Jaccard similarity |
| Class-pair analysis | 2,450 directed pairs | 1,225 undirected pairs |
| Spearman correlation | 0.237 | 0.2645 |
| Permutation test | 2,000 permutations | 5,000 permutations |
| Permutation p-value | ≈ 0.0005 | ≈ 0.0002 |

Both analyses found a **statistically significant but weak positive association** between attribute similarity and classification confusion.

For example, blue whale and humpback whale have high cosine attribute similarity (0.929), and 67% of blue whale images were misclassified as humpback whale.

**Finding:** Shared annotated attributes are associated with more frequent confusion, but attribute similarity alone does not fully explain CLIP's classification errors. These results indicate association, not causation.

## Human-only vs. GenAI-assisted Comparison

**Were there insights found by the human-only pair that the AI-aided pair overlooked, or vice versa?**

Both pairs found that animals sharing more attributes tend to be confused more often by CLIP, but the relationship is weak. The GenAI pair also found that some visually distinct animals had very high attribute similarity scores. Scores did no always match how similar animals look in real life.

**How did their visualization choices differ?**

The AI-aided pair used a clustered Jaccard similarity heatmap to group animals with similar attributes and focused on misclassification. The human pair used a regular heatmap and showed a wider range of misclassified images.

**Did AI assistance change the breadth or depth of the investigation?**

GenAI helped us explore different ways to compare animal attributes and classification errors. It also suggested additional analysis and helped to develop the visualizations.

**Did it help the AI-aided pair move more quickly, or did verifying and correcting AI suggestions introduce additional work?**

GenAI helped with planning and code implementing. However, some suggestions produced unexpected results or coding errors, so we spent additional time checking and correcting them. We cannot say exactly how much time it saved.

**How did the two pairs differ in their confidence in their conclusions, and what evidence justified that confidence?**

Although the GenAI-aided pair used only 33 appearance-related attributes, while the human-only pair used all 85 attributes, both reached a similar conclusion that attribute similarity has a weak positive relationship with classification errors.

