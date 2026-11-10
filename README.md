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
Human-only: Mai Elyamani & 

**Findings**
**Were there insights found by the human-only pair that the AI-aided pair overlooked, or vice versa?**
Both pairs found that animals sharing more attributes tend to be confused more often by CLIP, but the relationship is weak. The GenAI pair also found that some visually distinct animals had very high attribute similarity scores. Scores did no always match how similar animals actually look.

**How did their visualization choices differ?**

The AI-aided pair used a clustered Jaccard similarity heatmap to group animals with similar attributes and focused on misclassification examples. The human pair used a regular heatmap and showed a wider range of misclassified images.
**Did AI assistance change the breadth or depth of the investigation?**

GenAI helped us explore different ways to compare animal attributes and classification errors. It also suggested additional analysis and helped to develop the visualizations.

**Did it help the AI-aided pair move more quickly, or did verifying and correcting AI suggestions introduce additional work?**

GenAI helped with planning and code implementation. However, some suggestions produced unexpected results or coding errors, so we spent additional time checking and correcting them. We cannot say exactly how much time it saved.
**How did the two pairs differ in their confidence in their conclusions, and what evidence justified that confidence?**

Both pairs used statistical tests and found similar results even with different methods which may show that there is a weak relationship between shared attributes and confusion. 
