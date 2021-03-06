[![](https://img.shields.io/github/issues/shahules786/Twitter-Sentiment)]()
[![](https://img.shields.io/github/license/shahules786/Twitter-Sentiment)]()
[![](https://img.shields.io/github/stars/shahules786/Twitter-Sentiment)]()


# Twitter Sentiment analyzer


<p align="center">
  <img src="https://user-images.githubusercontent.com/25312635/95116850-4d01ff80-0765-11eb-887d-c3fbcf3797d0.png" />
</p>


Sentiment analysis is the task of determining the sentiment of a given expression in natural language, It is essentially a multiclass text classification text where the given input text is classified into positive, neutral, or negative sentiment. But the number of classes can vary according to the nature of the training dataset. This project aims to build a sentiment analyzer specifically for twitter domain.


<p align="center">
  <img src="https://user-images.githubusercontent.com/25312635/94103308-f1c13a80-fe51-11ea-819e-def5948c479f.png" width="50%" />
</p>

## Why a Custom model for twitter domain?

Simply put, a Tweet is a message sent on Twitter. Most of the tweets do not follow normal English grammar and vocabulary mainly due to the limitation of the number of characters allowed in a tweet. This requires special care to yield better performance, hence this project.

**Want to build a similar project? Read my [article](https://shahules786.medium.com/sentiment-analysis-build-your-nlp-project-d41257d06c8c) to find out how I build twittersentimt.**

## Install
`!pip install twittersentiment`

## Examples

- **Using pretrained model**

```python
from twittersentiment import TwitterSentiment
sent = TwitterSentiment.Sentiment()
sent.load_pretrained()
sent.predict("hey how are you?")
```

![basic](https://user-images.githubusercontent.com/25312635/96710969-71dbb100-13ba-11eb-9756-651384688a8b.gif)


- **You can train your own model with custom dataset and your choice of word embedding**


```python
from twittersentiment import TwitterSentiment
import pandas as pd
df = pd.read_csv("your_dataset.csv")
sent = TwitterSentiment.Sentiment()
sent.train(df["text"],df["target"],path="/your_model_save_path",name="6B",dim=100)
sent.train("hey you just trained a custom model")

```



 see [examples](https://github.com/shahules786/Twitter-Sentiment/blob/master/examples/basic.ipynb) for more.


