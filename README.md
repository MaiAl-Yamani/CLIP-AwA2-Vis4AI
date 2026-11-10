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

