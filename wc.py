text = """
Sentiment-Oriented Metric Learning for Text-to-Image Retrieval
In this era of multimedia Web, text-to-image retrieval is a critical function of search engines and visually-oriented online platforms. Traditionally, the task primarily deals with matching a text query with the most relevant images available in the corpus. To an increasing extent, the Web also features visual expressions of preferences, imbuing images with sentiments that express those preferences. Cases in point include photos in online reviews as well as social media. In this work, we study the effects of sentiment information on text-to-image retrieval. Particularly, we present two approaches for incorporating sentiment orientation into metric learning for cross-modal retrieval. Each model emphasizes a hypothesis on how positive and negative sentiment vectors may be aligned in the metric space that also includes text and visual vectors. Comprehensive experiments and analyses on Visual Sentiment Ontology (VSO) and Yelp.com online reviews datasets show that our models significantly boost the retrieval performance as compared to various sentiment-insensitive baselines.

Bilateral Variational Autoencoder for Collaborative Filtering
Preference data is a form of dyadic data, with measurements associated with pairs of elements arising from two discrete sets of objects. These are users and items, as well as their interactions, e.g., ratings.  We are interested in learning representations for both sets of objects, i.e., users and items, to predict unknown pairwise interactions. Motivated by the recent successes of deep latent variable models, we propose Bilateral Variational Autoencoder (BiVAE), which arises from a combination of a generative model of dyadic data with two inference models, user- and item-based, parameterized by neural networks. Interestingly, our model can take the form of a Bayesian variational autoencoder either on the user or item side. As opposed to the vanilla VAE model, BiVAE is ``bilateral'', in that users and items are treated similarly, making it more apt for two-way or dyadic data. While theoretically sound, we formally show that, similarly to VAE, our model might suffer from an over-regularized latent space.  This issue, known as posterior collapse in the VAE literature, may appear due to assuming an over-simplified prior (isotropic Gaussian) over the latent space. Hence, we further propose a mitigation of this issue by introducing constrained adaptive prior (CAP) for learning user- and item-dependent prior distributions. Empirical results on several real-world datasets show that the proposed model outperforms conventional VAE and other comparative collaborative filtering models in terms of item recommendation. Moreover, the proposed CAP further boosts the performance of BiVAE. An implementation of BiVAE is available on Cornac recommender library.

Reproducibility Companion Paper: Visual Sentiment Analysis for Review Images with Item-Oriented and User-Oriented CNN
We revisit our contributions on visual sentiment analysis for online review images published at ACM Multimedia 2017, where we develop item-oriented and user-oriented convolutional neural networks that better capture the interaction of image features with specific expressions of users or items. In this work, we outline the experimental claims as well as describe the procedures to reproduce the results therein. In addition, we provide artifacts including data sets and code to replicate the experiments.

Cornac: A Comparative Framework for Multimodal Recommender Systems
Cornac is an open-source Python framework for multimodal recommender systems. In addition to core utilities for accessing, building, evaluating, and comparing recommender models, Cornac is distinctive in putting emphasis on recommendation models that leverage auxiliary information in the form of a social network, item textual descriptions, product images, etc. Such multimodal auxiliary data supplement user-item interactions (e.g., ratings, clicks), which tend to be sparse in practice. To facilitate broad adoption and community contribution, Cornac is publicly available at https://github.com/PreferredAI/cornac, and it can be installed via Anaconda or the Python Package Index (pip). Not only is it well-covered by unit tests to ensure code quality, but it is also accompanied with a detailed documentation, tutorials, examples, and several built-in benchmarking data sets

Multimodal Review Generation for Recommender Systems
Key to recommender systems is learning user preferences, which are expressed through various modalities. In online reviews, for instance, this manifests in numerical rating, textual content, as well as visual images. In this work, we hypothesize that modelling these modalities jointly would result in a more holistic representation of a review towards more accurate recommendations. Therefore, we propose Multimodal Review Generation (MRG), a neural approach that simultaneously models a rating prediction component and a review text generation component. We hypothesize that the shared user and item representations would augment the rating prediction with richer information from review text, while sensitizing the generated review text to sentiment features based on user and item of interest. Moreover, when review photos are available, visual features could inform the review text generation further. Comprehensive experiments on real-life datasets from several major US cities show that the proposed model outperforms comparable multimodal baselines, while an ablation analysis establishes the relative contributions of the respective components of the joint model.

VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis
Detecting the sentiment expressed by a document is a key task for many applications, e.g., modeling user preferences, monitoring consumer behaviors, assessing product quality. Traditionally, the sentiment analysis task primarily relies on textual content. Fueled by the rise of mobile phones that are often the only cameras on hand, documents on the Web (e.g., reviews, blog posts, tweets) are increasingly multimodal in nature, with photos in addition to textual content. A question arises whether the visual component could be useful for sentiment analysis as well. In this work, we propose Visual Aspect Attention Network or VistaNet, leveraging both textual and visual components. We observe that in many cases, with respect to sentiment detection, images play a supporting role to text, highlighting the salient aspects of an entity, rather than expressing sentiments independently of the text. Therefore, instead of using visual information as features, VistaNet relies on visual information as alignment for pointing out the important sentences of a document using attention. Experiments on restaurant reviews showcase the effectiveness of visual aspect
attention, vis-a-vis visual features or textual attention.

Visual Sentiment Analysis for Review Images with Item-Oriented and User-Oriented CNN
Online reviews are prevalent. When recounting their experience with a product, service, or venue, in addition to textual narration, a reviewer frequently includes images as photographic record. While textual sentiment analysis has been widely studied, in this paper we are interested in visual sentiment analysis to infer whether a given image included as part of a review expresses the overall positive or negative sentiment of that review. Visual sentiment analysis can be formulated as image classification using deep learning methods such as Convolutional Neural Networks or CNN. However, we observe that the sentiment captured within an image may be affected by three factors: image factor, user factor, and item factor. Essentially, only the rst factor had been taken into account by previous works on visual sentiment analysis. We develop item-oriented and user-oriented CNN that we hypothesize would beer capture the interaction of image features with specific expressions of users or items. Experiments on images from restaurant reviews show these to be more effective at classifying the sentiments of review images.
"""


###################################################################################################


import os
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


stopwords = set(STOPWORDS)
stopwords.add("e")
stopwords.add("g")
stopwords.add("acm")

wc = WordCloud(
    width=500,
    height=230,
    margin=2,
    background_color="white",
    max_words=200,
    stopwords=stopwords,
    min_font_size=4,
    max_font_size=40,
    random_state=34,
).generate(text.strip().lower())

plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("./assets/wc.png", bbox_inches="tight", pad_inches=0)
