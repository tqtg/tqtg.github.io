import os
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


text = " ".join(l.strip() for l in open("abstracts.txt", "r").readlines()).lower()
stopwords = set(STOPWORDS)
stopwords.add("e")
stopwords.add("g")
stopwords.add("acm")

wc = WordCloud(
    width=500,
    height=200,
    background_color="white",
    max_words=100,
    stopwords=stopwords,
    max_font_size=40,
    random_state=34,
).generate(text)

plt.figure(figsize=(12, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("../assets/wc.png")
