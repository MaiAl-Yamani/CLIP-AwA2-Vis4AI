# Phase I - GenAI interaction record RQ 1

Favour Dilichukwu Ozioma - k12539293

**Model:** CLIP ViT-B/16 (OpenAI), open_clip config ViT-B-16-quickgelu

**Dataset:** Animals with Attributes 2 (AwA2), fixed 5,000-image subset (50 classes, 100 images each)

**RQ1:** Which animal classes does CLIP confuse most frequently?

**GenAI:** Claude (Anthropic)

## 1. Setup and data loading

**Prompt:** Write me a Colab notebook for a T4 that runs zero-shot CLIP on Animals with Attributes 2. Use open_clip, model ViT-B-16-quickgelu, pretrained "openai", prompt template "a photo of a {}." I'm using a fixed 5,000-image subset, 50 classes at 100 images each. I want to know which classes it mixes up. (Followed by the data layout: awa2_subset_5000.csv, AwA2 download, zip of the 5,000 images on Drive, outputs to MyDrive/awa2_phase1_outputs/, and "Every section opens with a short markdown note.")

**AI suggestion:** Cells that mount Drive, look for the CSV, download AwA2, keep only the 5,000 listed images and zip them to Drive, then set seed 42 and batch size 128 and check that every image in the CSV exists.

**Decision:** Modified. I moved the model download to Drive so it isn't repeated on every run, and the image and CSV zips are read from Drive.

**Output:** A notebook that runs top to bottom in a fresh Colab session, with all 5,000 images found.

## 2. Zero-shot inference

**Prompt:** Zero-shot inference. Cache the logits. Softmax them, then store the prediction, the confidence, and the rank of the true class for each image.

**AI suggestion:** Encode "a photo of a {}." for each of the 50 classes, score every image by cosine similarity, scale by CLIP's learned temperature and softmax. It also added a comment on why the quickgelu config matches OpenAI's weights.

**Decision:** Accepted.

**Output:** Cached logits (clip_logits_ViT-B-16-quickgelu.npy) and predictions.csv with the prediction, confidence and true-class rank for every image.

**Limitations:** One prompt template, so results may shift under a different wording.

## 3. Accuracy

**Prompt:** Accuracy. Top-1 with a bootstrap 95% CI, and top-5. A bar chart of per-class accuracy. A classification report.

**AI suggestion:** Bootstrap over images with 2,000 resamples for the top-1 interval, a sorted per-class bar chart and a standard classification report.

**Decision:** Accepted.

**Output:** Top-1 accuracy 0.921 (95% CI 0.913 to 0.929), top-5 0.998, 395 errors. Nine classes are perfect. The weakest are blue whale (0.28) and ox (0.50). Without those two, the other 48 classes score 0.943.

**Limitations:** The interval resamples images only, so it doesn't cover the choice of subset or template.

## 4. Confusion matrix

**Prompt:** Confusion matrix. 50 x 50. Plot it row-normalized, and next to it a version with only the errors.

**AI suggestion:** Show the row-normalized matrix beside an errors-only version with the diagonal removed, so the off-diagonal cells are readable.

**Decision:** Accepted.

**Output:** confusion_matrix.png and confusion_matrix_counts.csv.

## 5. Confusion pairs

**Prompt:** Confusion pairs. Directed and undirected tables, top 20 of each. A bar chart of the top 15 directed pairs. Then a per-class summary of what each class is most often mistaken for.

**AI suggestion:** A directed table (true class to predicted class) and an undirected table (A as B plus B as A), a bar chart of the top 15 directed pairs, and a per-class summary table.

**Decision:** Accepted.

**Output:** The most frequent directed confusions are blue whale to humpback whale (67 of 100 images), ox to buffalo (26), ox to cow (24) and rat to mouse (15). Many pairs run one way only: no buffalo image is called ox, and humpback whale is called blue whale 5 times against 67 the other way. Files: confusions_directed.csv, confusions_undirected.csv, class_summary.csv, top_directed_confusions.png.

**Limitations:** The undirected total hides the one-way pattern, which is why both tables are kept.

## 6. Groups

**Prompt:** Groups. Build the groups yourself: felids, canids, cetaceans, hoofed animals, whatever else fits. Then check whether a wrong prediction lands in the true class's own group more often than chance, which is (group size - 1)/49. Add a heatmap of errors from group to group.

**AI suggestion:** Eleven groups: cetaceans, felids, canids, bears, hoofed animals, small rodent-like animals, primates, pinnipeds and otter, small carnivores, megafauna and other. It also set up the chance comparison and the group-to-group heatmap.

**Decision:** Accepted. I did not validate the groups against an outside taxonomy.

**Output:** 315 of the 395 errors (79.7%) stay inside the true class's own group, against 10.1% expected if errors were spread evenly over the 49 wrong classes (ratio 7.89). All 90 cetacean errors and all 27 primate errors stay in their group. The small groups leak the most, for example small carnivores send 14 of their 31 errors into the small rodent-like group. Heatmap saved as group_error_flow.png.

**Limitations:** The groups are Claude's choice and the within-group result depends on them. The "other" group holds a single class (bat), and the chance baseline assumes errors would otherwise be spread evenly over the 49 wrong classes.

## 7. Confidence and rank

**Prompt:** Confidence. Histograms for correct vs wrong predictions. For the errors, also show where the true class ranked.

**AI suggestion:** Two confidence histograms (correct and wrong) and a histogram of the true-class rank for the errors.

**Decision:** Accepted.

**Output:** Correct predictions average 0.92 confidence and wrong ones 0.64. When CLIP is wrong, the true class ranks second in 78% of errors and in the top three in 91%. Some errors are very confident, for example gorilla called chimpanzee at 0.98. Saved as confidence_and_rank.png.

**Limitations:** Confidence separates right from wrong on average but does not catch every error.

## 8. Example images

**Prompt:** Example images. The most confident errors, plus a handful from the top directed confusion.

**AI suggestion:** A static gallery of the 12 most confident errors, then up to 8 images from the top directed confusion (blue whale predicted as humpback whale), each titled with the true label, predicted label and confidence.

**Decision:** Accepted.

**Output:** Two galleries shown in the notebook: the most confident errors overall, and blue whale images predicted as humpback whale.

## 9. Saving and summary

**Prompt:** Saving and summary. CSVs and plots to Drive, then print a short plain-text summary of the main results.

**AI suggestion:** Write all tables and plots to MyDrive/awa2_phase1_outputs/ and print a plain-text summary of the main results.

**Decision:** Accepted.

**Output:** per_class_accuracy.csv, predictions.csv, class_summary.csv, confusions_directed.csv, confusions_undirected.csv, confusion_matrix_counts.csv and the plots listed above.

## Summary of AI use

**Failure patterns:** None. The notebook ran top to bottom and produced every output I expected.

**Weak spots found on review:** Claude chose the groups and I did not validate them. I used a single prompt template throughout.

**Best practices:** Put the exact model string, weights, template and data layout in the prompt. Have the AI cache slow steps on Drive. Ask for confusion tables in both directions. Treat anything the AI defines for you, such as the animal groups, as an assumption and check it.

