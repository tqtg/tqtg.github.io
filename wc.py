text = """
Cornac-AB: An Open-Source Recommendation Framework with Native A/B Testing Integration
Recommender systems significantly impact user experience across diverse domains, yet existing frameworks often prioritize offline evaluation metrics, neglecting the crucial integration of A/B testing for forward-looking assessments. In response, this paper introduces a new framework seamlessly incorporating A/B testing into the Cornac recommendation library. Leveraging a diverse collection of model implementations in Cornac, our framework enables effortless A/B testing experiment setup from offline trained models. We introduce a carefully designed dashboard and a robust backend for efficient logging and analysis of user feedback. This not only streamlines the A/B testing process but also enhances the evaluation of recommendation models in an online environment. Demonstrating the simplicity of on-demand online model evaluations, our work contributes to advancing recommender system evaluation methodologies, underscoring the significance of A/B testing and providing a practical framework for implementation. The framework is open-sourced at https://github.com/PreferredAI/cornac-ab.

Topological Representation Learning for E-commerce Shopping Behaviors
Learning compact representation from customer shopping behaviors is at the core of web-scale E-commerce recommender systems. At Amazon, we put great efforts into learning embedding of customer engagements in order to fuel multiple downstream tasks for better recommendation services. In this work, we define the notion of shopping trajectory that consists of customer interactions at the categorical level of products, then construct an end-to-end model namely C-STAR which is capable of learning rich embedding for representing the variable-length customer trajectory. C-STAR explicitly captures the trajectory distribution similarity and trajectory topological semantics, providing a coarse-to-fine trajectory representation learning paradigm both structurally and semantically. We evaluate the model on Amazon proprietary data as well as four public datasets, where the learned embeddings have shown to be effective for customer-centric tasks including customer segmentation and shopping trajectory completion.

Multi-Modal Recommender Systems: Towards Addressing Sparsity, Comparability, and Explainability
Web applications frequently feature a recommender system to help users discover items (e.g., products, content articles) of interest. This tutorial focuses on multi-modality, i.e., the use of side information such as text, images, or graphs to augment the preference data.  In particular, we cover several important aspects of multi-modality.  First is how models rely on the auxiliary modality to address the sparsity of preference observations in order to better bridge users and items. These models are typically designed along modality lines, which we cover comprehensively. Second is how to manage comparison and cross-utilization of multi-modal models.  The former is concerned with streamlining the treatment of models that share the same modality. The latter is concerned with using a model initially designed for one modality with another.  Third is how the auxiliary modalities could act as recommendation explanations, as recipients may find textual, visual, or graphical explanations more intuitive. This is a hands-on tutorial, whereby lectures are supplemented with exercises conducted with Cornac (https://cornac.preferred.ai), a comparative framework for multimodal recommender systems.

Concept-Oriented Transformers for Visual Sentiment Analysis
In the richly multimedia Web, detecting sentiment signals expressed in images would support multiple applications, e.g., measuring customer satisfaction from online reviews, analyzing trends and opinions from social media.  Given an image, visual sentiment analysis aims at recognizing positive or negative sentiment, and occasionally neutral sentiment as well.  A nascent yet promising direction is Transformer-based models applied to image data, whereby Vision Transformer (ViT) establishes remarkable performance on large-scale vision benchmarks.  In addition to investigating the fitness of ViT for visual sentiment analysis, we further incorporate concept orientation into the self-attention mechanism, which is the core component of Transformer.  The proposed model captures the relationships between image features and specific concepts.  We conduct extensive experiments on Visual Sentiment Ontology (VSO) and Yelp.com online review datasets, showing that not only does the proposed model significantly improve upon the base model ViT in detecting visual sentiment but it also outperforms previous visual sentiment analysis models with narrowly-defined orientations.  Additional analyses yield insightful results and better understanding of the concept-oriented self-attention mechanism.

AmpSum: Adaptive Multiple-Product Summarization towards Improving Recommendation Captions
In e-commerce websites, multiple related product recommendations are usually organized into “widgets”, each given a name, as a recommendation caption, to describe the products within. These recommendation captions are usually manually crafted and generic in nature, making it difficult to attach meaningful and informative names at scale. As a result, the captions are inadequate in helping customers to better understand the connection between the multiple recommendations and make faster product discovery. We propose an Adaptive Multiple-Product Summarization framework (AmpSum) that automatically and adaptively generates widget captions based on different recommended products. The multiplicity of products to be summarized in a widget caption is particularly novel. The lack of well-developed labels motivates us to design a weakly supervised learning approach with distant supervision to bootstrap the model learning from pseudo labels, and then fine-tune the model with a small amount of manual labels. To validate the efficacy of this method, we conduct extensive experiments on several product categories of Amazon data. The results demonstrate that our proposed framework consistently outperforms state-of-the-art baselines over 9.47-29.14% on ROUGE and 27.31% on METEOR. With case studies, we illustrate how AmpSum could adaptively generate summarization based on different product recommendations.

Multi-Modal Recommender Systems: Hands-On Exploration
Recommender systems typically learn from user-item preference data such as ratings and clicks. This information is sparse in nature, i.e., observed user-item preferences often represent less than 5% of possible interactions. One promising direction to alleviate data sparsity is to leverage auxiliary information that may encode additional clues on how users consume items. Examples of such data (referred to as modalities) are social networks, item’s descriptive text, product images. The objective of this tutorial is to offer a comprehensive review of recent advances to represent, transform and incorporate the different modalities into recommendation models. Moreover, through practical hands-on sessions, we consider cross model/modality comparisons to investigate the importance of different methods and modalities. The hands-on exercises are conducted with Cornac (https://cornac.preferred.ai), a comparative framework for multimodal recommender systems. The materials are made available on https://preferred.ai/recsys21-tutorial/.

Variational Learning from Implicit Bandit Feedback
Recommendations are prevalent in Web applications (e.g., search ranking, item recommendation, advertisement placement). Learning from bandit feedback is challenging due to the sparsity of feedback limited to system-provided actions. In this work, we focus on batch learning from logs of recommender systems involving both bandit and organic feedbacks. We develop a probabilistic framework with a likelihood function for estimating not only explicit positive observations but also implicit negative observations inferred from the data. Moreover, we introduce a latent variable model for organic-bandit feedbacks to robustly capture user preference distributions. Next, we analyze the behavior of the new likelihood under two scenarios, i.e., with and without counterfactual re-weighting. For speedier item ranking, we further investigate the possibility of using Maximum-a-Posteriori (MAP) estimate instead of Monte Carlo (MC)-based approximation for prediction. Experiments on both real datasets as well as data from a simulation environment show substantial performance improvements over comparable baselines.

Exploring Cross-Modality Utilization in Recommender Systems
Multimodal recommender systems alleviate the sparsity of historical user-item interactions. They are commonly catalogued based on the type of auxiliary data (modality) they leverage, such as preference data plus user-network (social), user/item texts (textual), or item images (visual) respectively. One consequence of this categorization is the tendency for virtual walls to arise between modalities. For instance, a study involving images would compare to only baselines ostensibly designed for images. However, a closer look at existing models' statistical assumptions about any one modality would reveal that many could work just as well with other modalities. Therefore, we pursue a systematic investigation into several research questions: which modality one should rely on, whether a model designed for one modality may work with another, which model to use for a given modality. We conduct cross-modality and cross-model comparisons and analyses, yielding insightful results pointing to interesting future research directions for multimodal recommender systems.

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


import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS


rng = np.random.RandomState(34)


def color_fn(*args, **kwargs):
    r = rng.randint(16, 240)
    g = rng.randint(r // 4 + 1)
    b = 0
    return (r, g, b)


STOPWORDS.update(
    [
        "acm",
        "e",
        "g",
        "work",
        "moreover",
        "show",
        "using",
        "one",
        "two",
        "may",
        "available",
        "well",
        "several",
        "based",
        "result",
        "cross",
        "proposed",
        "oriented",
        "b",
        "multiple",
    ]
)

wc = WordCloud(
    font_path="./assets/fonts/Lato-Regular.ttf",
    width=500,
    height=240,
    margin=16,
    background_color="white",
    color_func=color_fn,
    max_words=94,
    stopwords=STOPWORDS,
    min_font_size=10,
    max_font_size=50,
    random_state=94,
).generate(text.strip().lower())

plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("./assets/wc.jpg", bbox_inches="tight", pad_inches=0)
