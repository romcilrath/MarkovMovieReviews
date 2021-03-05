# ABOUT
------------------------------------------------------------------------------------------------------------------------
This is a small scale personal Python application I put together in my spare time.
My goal was to gain a better practical understanding of Markov Chains, specifically for text generation.
Generating some nonsense movie reviews was a fun way of achieving that goal.
This takes in a large amount of text and trains a model with it that can be used to print stochastic markov text chains.

For example training 'data/movie_reviews_complete.txt' (with k = 2) the seed "This movie" outputs:
    This movie has it all.
    This movie was great.
    This movie was shot in a long time, and of course and he does best, and some scary ones.
    This movie has a very long time, with the later version remake.
    This movie doesnt have much of the world of her character and an occasional lacklustre gag, but still good.
    This movie was just a little bit like a bit of soft core sex.
    This movie does not come from the year.
    This movie is the one that is an enjoyable and when I was just a masterpiece of tone for the schizophrenic character.
    This movie was a great movie.
    This movie is very different characters and the good guys are the three main characters.

Sometimes they almost make sense! A fun demonstration of what Markov Chains can do.

Warning: Input data movie reviews were not censored, my apologies if the Markov Chain says anything rude!

You can also create your own text file where each portion of text is separated by line.
For example with the provided data each line is a complete movie review.

# HOW TO USE
------------------------------------------------------------------------------------------------------------------------
*Important:*
Before beginning you must unzip the data and models such that when finished the file structure looks like this:
    MarkovMovieReviews
        |--data
        |--|--movie_reviews_complete.txt
        |--|--movie_reviews_small.txt
        |--models
        |--|--complete_k_1_model.pkl
        |--|--complete_k_2_model.pkl
        |--|--small_k_3_model.pkl
        |--|--small_k_4_model.pkl
        main.py
        markov_text_chain.py
        README.txt
This was to circumvent GitHub's 100MB file size limit while still providing my formatted data and prepared models.

You can just run main to get a demo of some models I prepared.
Also the below will instruct you in more detail how to use some of the functions.

Training:
    >model = MarkovTextChainModel('complete_k_2_model', 2, 'data/movie_reviews_complete.txt')
    >model.save('models/')

    Will train a model named 'complete_k_2' with k = 2 on the text from 'data/movie_reviews_complete.txt'
    Then save the model in the 'models/' directory as 'models/complete_k_2_model.pkl'

Loading:
    > model = markov_text_chain.load_model('data/movie_reviews_complete.txt')

    Will unpickle the model from the file 'data/movie_reviews_complete.txt'

Generating Markov Text Chains:
    >chain = model.stochastic_chain("The director was", 50, True)
    >markov_text_chain.print_chain(chain)

    Will generate a stochastic text chain from the model with the seed "The director was" that is no longer than...
        ...50 words (punctuation counts as a word), and the chain also will stop after an end sentence punctuation
    Then prints out the chain with text and punctuation properly formatted and spaced

    Note:
    A seed is needed to start the chain, choose something likely to be in the data set.
    Also the seed must be exactly k words long.


# DATA
------------------------------------------------------------------------------------------------------------------------
In the data/ directory there are two files, both composed of publicly available data compiled by Andrew Mass.
    Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
    ACL 2011 Paper: http://www.aclweb.org/anthology/P11-1015

The data set provides 50k highly polar movie reviews (this data set was originally used for sentiment classification).
This has the fun byproduct: the resultant text chains have some very strong opinions about made-up movies.

The data has been organized such that each review is its own line.
The MarkovTextChainModel class in the markov_text_chain.py ideally can be used on any similar set of text data.

In the data/ directory:
    1. movie_reviews_complete.txt   : the test and train sets compiled into one .txt file.
    2. movie_reviews_small.txt      : the first 27785 reviews of (1.)

The MarkovTextChainModel class in the markov_text_chain.py ideally can be used on any similar set of text data.