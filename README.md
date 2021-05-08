# atm
ATM: Adversarial-neural Topic Model

Paper URL:
https://arxiv.org/abs/1811.00265

title:
    ATM:Adversarial-neural Topic Model

authors:
    Rui Wang, Deyu Zhou, Yulan He

Abstract:
    Topic models are widely used for thematic structure discovery in text.
    But traditional topic models often require dedicated inference procedures for specific tasks at hand.
    Also, they are not designed to generate word-level semantic representations.
    To address these limitations, we propose a topic modeling approach based on Generative Adversarial Nets (GANs),
    called Adversarial-neural Topic Model (ATM).
    The proposed ATM models topics with Dirichlet prior and employs a generator network
    to capture the semantic patterns among latent topics. Meanwhile, the generator could also
    produce word-level semantic representations. To illustrate the feasibility of porting ATM
    to tasks other than topic modeling, we apply ATM for open domain event extraction.
    Our experimental results on the two public corpora show that ATM generates more coherence topics,
    outperforming a number of competitive baselines.
    Moreover, ATM is able to extract meaningful events from news articles.


for the ATM Implementation, there isn't an official repository.
Thus, we implemented our solution
starting from the official paper
and taking inspiration from some aspects of a the third part repository:
https://github.com/ravichoudharyds/Topic_Modeling
