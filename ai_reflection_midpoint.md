# Phase I 
Favour Dilichukwu Ozioma - RQ1

Zolboo Chimedregzen K12349108 - RQ2


**RQ2:** Are animals with more shared visual attributes more likely to be misclassified as each other?

**GenAI:** ChatGPT (OpenAI)

As for RQ2, I used ChatGPT to help me with analythic methods and visualizations to solve the RQ. It was very useful for developing and planning the analysis step by step when I wanted to dig deeper in a result or compare different approaches. Unfortunately, not every suggestion was suitable, so I still had to evaluate them and make final decisions which made the most sense for the task.

####  Failure patterns
The standard Spearman test gave a very small p-value (4.7119e-21). I questioned if the significance test was appropriate to use since the same animal classes appeared in multiple pairs, hence the observations are not independent. So I decided to use a class-label permutation test with 5,000 permutations as a balance between accuracy and computation time.

Some suggestions were also more complicated than necessary. ChatGPT proposed an interactive visualization for misclassified images, but I decided that a static gallery was sufficient for the analysis and easier to present in 5 min presentation. 

#### Best practices
Compare different methods before landing on one - I compared suggested regular and clustered heatmap and kept the clustered because it made the similarity patterns easier to identify.

Question the suggested statistical methods -  standard Spearman significance test did not take into consideration the dependence between animal pairs. Checking this helped me choose a more appropriate test (class label permutance) and avoid reaching misleading conclusions.

**RQ1:** which animal classes get confused?

## 2. Method

I ran OpenAI's CLIP ViT-B/16 zero-shot through `open_clip` (config ViT-B-16-quickgelu, weights "openai"), with no fine-tuning. The data is a fixed 5,000-image subset of Animals with Attributes 2: 50 classes, 100 images each. Each class name goes into one template, "a photo of a {}.", and gets encoded and normalized. For every image, CLIP compares it with all 50 text embeddings, scales the scores by its learned temperature and runs a softmax. The label set is closed, so CLIP has to pick one of the 50 classes even when none fits well. I used seed 42, batch size 128 and a Colab T4.

For each image I kept the prediction, the confidence and where the true class ranked among the 50 scores. From that I made per-class accuracy, a 50 x 50 confusion matrix, confusion tables for both directions and for unordered pairs, and a count of errors by animal group. Claude chose the groups (cetaceans, felids, canids, bears, hoofed animals, small rodent-like animals, primates, pinnipeds and otter, small carnivores, megafauna, other) instead of using a taxonomy file.

## 3. Results

### 3.1 Overall accuracy

Top-1 accuracy is 0.921 (95% bootstrap CI 0.913 to 0.929, resampling images) and top-5 is 0.998. That leaves 395 wrong images out of 5,000.

Nine classes are perfect: lion, giant panda, elephant, buffalo, giraffe, tiger, rabbit, squirrel and zebra. At the other end sit blue whale (0.28), ox (0.50), rat (0.73), weasel (0.76), mouse (0.83), otter (0.84), walrus (0.85), cow (0.87), beaver (0.87) and hamster (0.89).

Two classes cause a big chunk of the trouble. Blue whale and ox alone account for 122 of the 395 errors. Take them out and the other 48 classes score 0.943.

### 3.2 Most frequent confusions

Directed confusions (true class predicted as another class), top ten:

| True class | Predicted as | Images | Share of true class |
|---|---|---|---|
| blue whale | humpback whale | 67 | 67% |
| ox | buffalo | 26 | 26% |
| ox | cow | 24 | 24% |
| rat | mouse | 15 | 15% |
| otter | beaver | 10 | 10% |
| cow | ox | 9 | 9% |
| chimpanzee | spider monkey | 8 | 8% |
| beaver | otter | 8 | 8% |
| hamster | mouse | 8 | 8% |
| spider monkey | chimpanzee | 8 | 8% |

Undirected confusions (A as B plus B as A), top ten:

| Pair | A as B | B as A | Total |
|---|---|---|---|
| blue whale / humpback whale | 67 | 5 | 72 |
| cow / ox | 9 | 24 | 33 |
| buffalo / ox | 0 | 26 | 26 |
| mouse / rat | 5 | 15 | 20 |
| beaver / otter | 8 | 10 | 18 |
| chimpanzee / spider monkey | 8 | 8 | 16 |
| hamster / mouse | 8 | 5 | 13 |
| seal / walrus | 6 | 5 | 11 |
| chimpanzee / gorilla | 3 | 7 | 10 |
| humpback whale / killer whale | 3 | 5 | 8 |

Most of these confusions run in one direction. Blue whale becomes humpback whale 67 times, while humpback whale becomes blue whale only 5 times. Ox images get called buffalo 26 times, but no buffalo image is ever called ox. Persian cat turns into siamese cat 6 times and the reverse never happens. Chimpanzee and spider monkey are the odd pair out, with 8 mistakes each way. I suspect one class's text embedding acts as a default for its neighbor, but I didn't test that.

### 3.3 Errors within groups

Of the 395 errors, 315 (79.7%) stay inside the true class's own group. If CLIP spread its errors evenly over the 49 wrong classes, only 10.1% would, so the observed share is 7.9 times higher. Every one of the 90 cetacean errors stays among cetaceans, every one of the 27 primate errors stays among primates, and 77 of the 83 hoofed-animal errors stay among hoofed animals. The small groups leak the most: pinnipeds and otter send 11 of their 39 errors into the small rodent-like group, and small carnivores send 14 of their 31 there.

A few big cases push that share up. Blue whale to humpback whale alone is 67 of the 395 errors.

### 3.4 Confidence and rank of the true class

Correct predictions average 0.92 confidence (median 0.98), and wrong ones average 0.64 (median 0.62). So confidence tells right from wrong on average, but it won't catch everything. The most confident mistakes are two gorilla images (one called chimpanzee, one spider monkey, both at 0.98), a polar bear called grizzly bear (0.98) and an ox called buffalo (0.97).

When CLIP is wrong, the true class is usually right behind. It ranks second in 78% of errors and in the top three in 91%. A confident miss can still bury it: one Persian cat called bobcat has the true class sixth, and the farthest miss in the whole set has it 23rd.

### 3.5 Limits of these results

This is one prompt template, one model and 100 images per class. The confidence interval resamples images only, so it says nothing about the choice of subset or template, and zero-shot accuracy can shift with the template. A different one could reorder the pairs. Claude picked the groups, and I haven't checked them against an outside taxonomy.

## 4. AI use record

Tool: Claude (Anthropic), in the chat interface.

### 4.1 Important prompts

Claude wrote the notebook from one long prompt, which I sent in segments. I don't remember exactly where I split it, so the segments below follow the prompt's numbered sections.

**Segment 1, setup and constraints**

> Write me a Colab notebook for a T4 that runs zero-shot CLIP on Animals with Attributes 2. Use open_clip, model ViT-B-16-quickgelu, pretrained "openai", prompt template "a photo of a {}." I'm using a fixed 5,000-image subset, 50 classes at 100 images each. I want to know which classes it mixes up.
>
> The subset lives in awa2_subset_5000.csv (columns image_path and true_label, with "+" standing in for spaces in the labels). First cell mounts Drive and hunts for the CSV. If it isn't there, ask me to upload it. Then pull AwA2 from https://cvml.ista.ac.at/AwA2/AwA2-data.zip, keep only the 5,000 images the CSV lists, and zip them into Drive so the download only happens once. Everything the notebook writes, cached logits included, goes to MyDrive/awa2_phase1_outputs/.
>
> Every section opens with a short markdown note.

**Segment 2, config and inference**

> 1. Config and loading. Seed 42, batch size 128. Make sure every image in the CSV actually exists before going further.
> 2. Zero-shot inference. Cache the logits. Softmax them, then store the prediction, the confidence, and the rank of the true class for each image.

**Segment 3, accuracy, confusion matrix, confusion pairs**

> 3. Accuracy. Top-1 with a bootstrap 95% CI, and top-5. A bar chart of per-class accuracy. A classification report.
> 4. Confusion matrix. 50 x 50. Plot it row-normalized, and next to it a version with only the errors.
> 5. Confusion pairs. Directed and undirected tables, top 20 of each. A bar chart of the top 15 directed pairs. Then a per-class summary of what each class is most often mistaken for.

**Segment 4, groups and confidence**

> 6. Groups. Build the groups yourself: felids, canids, cetaceans, hoofed animals, whatever else fits. Then check whether a wrong prediction lands in the true class's own group more often than chance, which is (group size - 1)/49. Add a heatmap of errors from group to group.
> 7. Confidence. Histograms for correct vs wrong predictions. For the errors, also show where the true class ranked.

**Segment 5, examples, saving, delivery**

> 8. Example images. The most confident errors, plus a handful from the top directed confusion.
> 9. Saving and summary. CSVs and plots to Drive, then print a short plain-text summary of the main results.
>
> Give me the notebook and nothing else. It has to run top to bottom in a fresh Colab session.

### 4.2 How AI suggestions affected analytical decisions

The prompt fixed the model, weights, template, subset size, seed and the nine-section plan, so Claude's influence sits in what I left open. It chose the group definitions (eleven groups, including odd ones like "pinnipeds and otter" and "small carnivores"), the number of bootstrap resamples (2,000), how the true-class rank is computed and the layout of the errors-only heatmap. It also added a comment on why the quickgelu config matches OpenAI's weights. The groups matter most, because the within-group result in section 3.3 depends on them.

### 4.3 Accepted, modified and refined

What I changed: I moved the model download to Drive so it isn't repeated on every run, and the image and CSV zips are read from Drive as well. That saved run time and repeated downloads across Colab sessions.

What I kept: everything else as Claude wrote it, including the group definitions, the bootstrap and the plotting code.

### 4.4 Failure patterns observed

Nothing went wrong. The notebook ran top to bottom and gave me every output I expected. Looking back at the results, I see two weak spots that aren't AI errors: Claude chose the groups and I didn't validate them, and I used a single prompt template throughout.

### 4.5 Emerging best practices

- Put the exact model string, weights, template and data layout in the prompt. Mine left Claude no room to swap in a different CLIP variant or prompt, so the results can be reproduced.
- Have the AI cache slow steps (logits, image zips) on Drive. The CLIP pass ran once and every later cell read saved files.
- Ask for confusion tables in both directions and as unordered pairs. The directed one shows that ox to buffalo goes one way, which the pair total hides.
- Treat anything the AI defines for you, like the animal groups, as an assumption. Check it, or say it's unchecked, because later results depend on it.
- Read the output files, not just the notebook, and see that they agree. Here the per-group error counts add up to the 395 total errors.
