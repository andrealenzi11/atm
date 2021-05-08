if __name__ == '__main__':

    import torch
    from gensim import corpora, models
    from sklearn.feature_extraction.text import CountVectorizer
    from torch.utils.data import Dataset

    from tm_utils.business.topics_evaluation import get_topics_words_scores
    from tm_utils.business.topics_utils import get_eng_keyed_vectors, save_scores_diz, save_html_topics_table

    from tm_utils.dao.dataset_dao import TwentyNewsgroupsDao, ImdbReviewsDao, ReutersNewswireDao
    from tm_utils.config.configuration import TWENTY_NEWS_GROUP, IMDB_REVIEWERS, REUTERS, ENG_STOPWORDS

    from src.atm_adversial_neural_tm import TopicModelDataset, GenerativeAdversarialNetworkTopicModel

    INPUT_DATASET = TWENTY_NEWS_GROUP
    print(f"\t\t input_dataset:  {INPUT_DATASET}")
    print("\n")

    NUM_TOP_WORDS = 50
    print(f"\t\t num. top words:  {NUM_TOP_WORDS}")
    print("\n\n")

    if INPUT_DATASET not in [TWENTY_NEWS_GROUP, IMDB_REVIEWERS, REUTERS]:
        raise ValueError(f"Invalid input dataset '{INPUT_DATASET}'!")

    if INPUT_DATASET == TWENTY_NEWS_GROUP:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            TwentyNewsgroupsDao().load_preprocessed()
    elif INPUT_DATASET == IMDB_REVIEWERS:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            ImdbReviewsDao().load_preprocessed()
    elif INPUT_DATASET == REUTERS:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            ReutersNewswireDao().load_preprocessed()
    else:
        raise ValueError(f"Invalid input dataset '{INPUT_DATASET}'!")

    print(f"\n\n >>> Input dataset '{INPUT_DATASET}' info:")
    print("\t\t x_train size: ", len(x_train))
    print("\t\t x_test size: ", len(x_test))
    print("\t\t x_train_prep size: ", len(x_train_prep))
    print("\t\t x_test_prep size: ", len(x_test_prep))
    print("\t\t y_train_prep size: ", len(y_train_prep))
    print("\t\t y_test_prep size: ", len(y_test_prep))

    cv_train = CountVectorizer(input='content', lowercase=True, stop_words=ENG_STOPWORDS,
                               max_df=0.60, min_df=0.0005, binary=False)
    cv_train.fit_transform(x_train_prep)
    valid_tokens = set(cv_train.get_feature_names())

    x_train_prep2_tokenized = list()
    x_train_prep2 = list()
    y_train_prep2 = list()
    for i in range(len(x_train_prep)):
        doc_tokenized = [word for word in x_train_prep[i].split() if word in valid_tokens]
        doc_str = " ".join(doc_tokenized)
        if len(doc_tokenized) > 0:
            x_train_prep2.append(doc_str)
            x_train_prep2_tokenized.append(doc_tokenized)
            y_train_prep2.append(y_train_prep[i])

    dictionary = corpora.Dictionary(x_train_prep2_tokenized)
    print(f"\n dictionary size: {len(dictionary)}")
    id_token_map = {}
    bow_corpus = [dictionary.doc2bow(doc) for doc in x_train_prep2_tokenized]
    print(f"\n corpus size: {len(bow_corpus)}")
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    dataset = TopicModelDataset(corpus=corpus_tfidf,
                                len_dict=len(dictionary))
    print(f"\n dataset size: {len(dataset)}")
    # print("docs in format token-tfidf sorted:")
    # for d in dataset:
    #     print(sorted({dictionary[i]: elem for i, elem in enumerate(d) if elem != 0.0}.items(),
    #                  key=lambda item: item[1],
    #                  reverse=True))

    print("\n >>> Load pretrained FasText english keyed vectors")
    eng_keyed_vectors = get_eng_keyed_vectors()

    for NUM_TOPICS in (10, 20, 30):

        print(f"\n\n\n ########## NUM_TOPICS = {NUM_TOPICS} ##########")

        # Hyper-parameters
        args = {
            'num_topics': NUM_TOPICS,
            'enc_mid_layer': 100,
            'dec_mid_layer': 100,
            'lambda_gp': 10,  # Loss weight for gradient penalty
            'vocab_size': len(dictionary),
            'batch_size': 512,
            'n_epochs': 100,
            'n_critic': 5,  # train Generator each 'n_critic' epochs
        }

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args['batch_size'],
                                             shuffle=True,
                                             drop_last=True)

        gan_tm = GenerativeAdversarialNetworkTopicModel(hyperparams_diz=args)

        print("\n >>> Start fitting...")
        gan_tm.fit(dataloader=loader,
                   learning_rate_generator=0.0001,
                   betas_generator=(0, 0.9),
                   learning_rate_discriminator=0.0001,
                   betas_discriminator=(0, 0.9))

        print("\n >>> Extract topics...")
        topics_matrix = gan_tm.extract_topics(dictionary=dictionary,
                                              n_top_words=NUM_TOP_WORDS)

        for i, topic in enumerate(topics_matrix):
            print(f"\t\t ({i + 1}) [#tokens={len(topic)}]  {topic}")

        print("\n >>> compute scores")
        scores_diz = get_topics_words_scores(topics=topics_matrix,
                                             kv=eng_keyed_vectors,
                                             corpus=x_train_prep2)
        print(scores_diz)

        save_scores_diz(scores_diz=scores_diz,
                        name="ATM",
                        num_topics=NUM_TOPICS,
                        dataset_name=INPUT_DATASET)

        save_html_topics_table(topics_matrix=topics_matrix,
                               name="ATM",
                               num_topics=NUM_TOPICS,
                               num_top_words=NUM_TOP_WORDS,
                               dataset_name=INPUT_DATASET)
        print("\n\n")

    print("\n\n >>> Complete!")
