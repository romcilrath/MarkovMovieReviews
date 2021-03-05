from markov_text_chain import MarkovTextChainModel, load_model, print_chain


if __name__ == '__main__':
    complete_k_1 = load_model('models/complete_k_1_model.pkl')
    complete_k_2 = load_model('models/complete_k_2_model.pkl')
    small_k_3 = load_model('models/small_k_3_model.pkl')
    small_k_4 = load_model('models/small_k_4_model.pkl')
    models = [complete_k_1, complete_k_2, small_k_3, small_k_4]
    data_paths = ['data/movie_reviews_complete.txt', 'data/movie_reviews_complete.txt', 'data/movie_reviews_small.txt', 'data/movie_reviews_small.txt']
    k_values = [1, 2, 3, 4]
    seeds = ["This", "This movie", "This movie was", "This movie was a"]

    for model, path, k, seed in zip(models, data_paths, k_values, seeds):
        print("Trained on \'" + path + "\' with k = " + str(k))
        print("Seed: " + seed)
        for i in range(0, 10):
            print("\t" + str(i+1) + ". ", end=' ')
            print_chain(model.stochastic_chain(seed))
            print()
        print()
