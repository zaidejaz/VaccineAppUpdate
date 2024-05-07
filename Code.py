import numpy as np
import theano.tensor as T
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, GRU
from keras.optimizers import Adagrad, RMSprop, Adam, SGD

# Existing code...
# Define your data loading, preprocessing, and evaluation functions

# Content-based RNN model
def get_content_rnn_model(num_items, max_sequence_length, embedding_dim, hidden_units):
    item_input = Input(shape=(max_sequence_length,), dtype='int32')
    item_embed = Embedding(input_dim=num_items, output_dim=embedding_dim, input_length=max_sequence_length)(item_input)
    item_rnn = GRU(hidden_units)(item_embed)
    item_flat = Flatten()(item_rnn)
    output = Dense(1, activation='sigmoid')(item_flat)

    model = Model(inputs=item_input, outputs=output)
    return model

# Existing code...

def get_model(num_users, num_items, latent_dim, content_rnn_embedding_dim, content_rnn_hidden_units, regs=[0,0]):
    # GMF model
    model_GMF = Model(input=[user_input, item_input], output=prediction)

    # Content-based RNN model
    model_content_rnn = get_content_rnn_model(num_items, max_sequence_length, content_rnn_embedding_dim, content_rnn_hidden_units)

    # Combine the GMF and content-based RNN models
    combined_output = Concatenate()([model_GMF.output, model_content_rnn.output])
    combined_output = Dense(1, activation='sigmoid')(combined_output)

    combined_model = Model(inputs=[model_GMF.input, model_content_rnn.input], outputs=combined_output)

    return model_GMF, model_content_rnn, combined_model

# Existing code...

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    # Existing code...

    # Build model
    model_GMF, model_content_rnn, combined_model = get_model(num_users, num_items, num_factors, content_rnn_embedding_dim, content_rnn_hidden_units, regs)
    if learner.lower() == "adagrad":
        combined_model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        combined_model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        combined_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        combined_model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Load your content-based RNN model and its dataset
    content_rnn_dataset = load_content_rnn_dataset()
    max_sequence_length = content_rnn_dataset.max_sequence_length
    num_items = content_rnn_dataset.num_items

    # Existing code...

    # Training loop
    for epoch in range(epochs):
        # ...

        # Training
        hist = combined_model.fit([np.array(user_input), np.array(item_input), content_rnn_dataset.item_sequences],
                                  np.array(labels), batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            # ...

            # Obtain predictions from the content-based RNN model
            content_rnn_predictions = model_content_rnn.predict(content_rnn_dataset.item_sequences, batch_size=batch_size)

            # Combine predictions from GMF and content-based RNN using a weight or combination function
            combined_predictions = weight * gmf_predictions + (1 - weight) * content_rnn_predictions

            # Evaluate the performance of your weighted hybrid model
            # ...

            # Save the best model
            # ...

    # ...

    print("End. Best Iteration %d: HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" % (model_out_file))