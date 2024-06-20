# quaternion-phocnet

# Introduction
[PHOCNet](https://arxiv.org/abs/1604.00187) is a state-of-the-art deep CNN for Keyword Spotting (KWS) in handwritten documents. Using Pyramidal Histogram of Characters (PHOC) as labels, PHOCNet can achieve outstanding performance in KWS for both Query-by-Example (QbE) and Query-by-String (QbS).

In our work, we transform PHOCNet from a conventional CNN to a Quaternionic CNN (QCNN). The objective is that with QCNN we can create a parameters-efficient network (appr. 1/4 of parameters) with equivalent (or better) performance and far better generalization ability. We focused on the case of QbE, not limiting our system to work for QbS with minimum adaptations. The final system produced uses KWS to retrieve pages (in image format) from a collection containing a query word.

# Installation
Use [requirements.txt](/requirements.txt) file to setup the environment with the necessary dependencies.

# Usage

### Training
To train a Q-PHOCNet model run [train.py](/experiments/train.py) script with minimum arguments as follows:

```
python train.py -ds <dataset_name> -sn <trained_model_name>.pt
```

Available datasets for training so far are GW and IAM. Using another dataset needs implementation of `torch.utils.data.Dataset` class accordingly. Other useful training arguments:
| option | description |
|:------|:-----------:|
| -lrs    | learning rate step       |
| -gpu_id | the ID of the GPU        |
| -pul    | PHOC unigram levels      |

For all available arguments see `def train()` in `train.py`.

### Retrieval
To retrieve images that contain a QbE run [retrieval_with_qbe.py](/retrieval/retrieval_with_qbe.py) as follows:

```
python retrieval_with_qbe.py -ds <dataset_name> -i <path_to_doc_collection> -m <trained_model_path>
```
The query image is specified by the user at runtime.

# Datasets
We trained our Q-PHOCNet on following datasets:

- [GW dataset](https://fki.tic.heia-fr.ch/databases/washington-database): The dataset is single-writer and contains 4,894 words. We applied data augmentation to get a total of 500,000 word instances.
- [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database): The dataset is multi-writer (657 writers) and contains 115,320 words.

# Examples

### QbE results for the word "about" of arbitrary writing style
<img src="/images/about_retrieval_results_q_phocnet_full.jpg" width="800" height="400" alt="retrieval results with Q-PHOCNet (full) for word 'about' of arbitrary writing style">

### QbE results for the word "last" of arbitrary and curved writing style
<img src="/images/last_curved_retrieval_results_q_phocnet_full.jpg" width="800" height="400" alt="retrieval results with Q-PHOCNet (full) for word 'last' of arbitrary and curved writing style">
This example demonstrates the tolerance of Q-PHOCNet in using distorted word images as queries.

# Evaluation

### Retrieval metrics

$`mAP = \frac{\sum_{q=1}^{Q} AveP(q)}{Q}`$ , where:
- **q**: current query
- **Q**: total number of queries
- **P(q)**: precision of query q

$`mAP_2 = \frac{\sum_{q=1}^{Q} (AP@n)_q }{Q}`$, $`AP@n = \frac{1}{GTP} \sum_{k=1}^{n} P@k \times rel@k`$, where:
- **q**: current query
- **Q**: total number of queries
- **AP@n**: average precision at n
- **GTP**: number of true positives
- **n**: number of results we interested in
- **P@k**: precision at k
- **rel@k**: relevance function equal to 1 if k element is relative to query, 0 otherwise

### Performance
|                |                   | GW      | IAM     |
|:--------------:|:-----------------:|:-------:|:-------:|
|                | params (millions) | mAP (%) | mAP (%) |
|Q-PHOCNet (full)| 17.8              | *96.15* | *72.12* |
|PHOCNet (1/2)   | 18                | 95.45   | 69.55   |
|Q-PHOCNet (1/2) | 4.5               | *95.55* | *56.84* |
|PHOCNet (1/4)   | 4.6               | 94.14   | 54.32   |
|Q-PHOCNet (1/4) | 1.1               | *85.13* | *34.49* |
|PHOCNet (1/8)   | 1.2               | 81.49   | 27.17   |

### Generalization
A generalization ability evaluation of our system was made using 10 words of various, arbitrary and distorted writing styles. The words and result are shown below:

<img src="/images/words_for_generalization_evaluation.jpg" width="800" height="400" alt="words used for generalization ability evaluation">

|                | $mAP_2$ (%) |
|:--------------:|:-----------:|
|Q-PHOCNet (full) | 86.6        |

# Citations

- [Sebastian Sudholt, & Gernot A. Fink. (2017). PHOCNet: A Deep Convolutional Neural Network for Word Spotting in Handwritten Documents](https://arxiv.org/abs/1604.00187)
  * [GitHub repo](https://github.com/ssudholt/phocnet)
- PHOCNet implementation in PyTorch based on Sudholt's implementation by @georgeretsi
  * [GitHub repo](https://github.com/georgeretsi/pytorch-phocnet)
- [Titouan Parcollet, Mirco Ravanelli, Mohamed Morchid, Georges Linarès, Chiheb Trabelsi, Renato De Mori, Yoshua Bengio - "Quaternion Recurrent Neural Networks", OpenReview](https://openreview.net/forum?id=ByMHvs0cFQ)
  * [GitHub repo](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)
 - [A. Fischer, A. Keller, V. Frinken, and H. Bunke: "Lexicon-Free Handwritten Word Spotting Using Character HMMs," in Pattern Recognition Letters, Volume 33(7), pages 934-942, 2012](https://www.sciencedirect.com/science/article/abs/pii/S0167865511002820?via%3Dihub)
 - [U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002](https://link.springer.com/article/10.1007/s100320200071)

### Disclaimer:
This project incorporates parts from other repositories. Corresponding licenses and repositories are included. In third-party files original repository is listed in comments, too.
