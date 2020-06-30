import trace

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, concatenate, LSTM, Embedding, \
    add, RepeatVector, Wrapper, Lambda, TimeDistributed, InputSpec, RNN, Layer, Reshape
import tensorflow.keras.backend as kb
import tensorflow.keras as k
from tensorflow.keras.utils import plot_model


def build_shallow_cnn(input_dim=(196, 196), nr_layers=2):
    if nr_layers < 1:
        raise Exception("Implementation problem. At least 1 cnn layer is needed.")
    conv_size = 5
    nr_filters = 8
    cnn_base = Input(shape=(input_dim[0], input_dim[1], 3))
    cnn_input = cnn_base
    for i in range(nr_layers):
        cnn_base = Conv2D(nr_filters, (conv_size, conv_size), activation='relu')(cnn_base)
        cnn_base = MaxPooling2D(pool_size=(conv_size - 2, conv_size - 2))(cnn_base)

        conv_size += 2
        nr_filters *= 2

    cnn_base = Dropout(0.5)(cnn_base)
    cnn_base = Flatten()(cnn_base)

    return cnn_base, cnn_input


def build_resnet50_feat():
    cnn_part = Input(shape=(2048,))

    return cnn_part


def build_incv3_feat():
    cnn_input = Input(shape=(2048,))
    # cnn_part = Dense(1024, activation='relu')(cnn_input)
    return cnn_input


def build_basic_model(cnn_input, cnn_part, sentence_length, vocabulary_size, embedding_matrix, embedding_dimensions=200,
                      loss_function='categorical_crossentropy', optimizer='RMSprop', nr_nodes=256, nr_gpus=1,
                      concat_add='add',
                      trainable_embedding=False, dropout=True):
    # Use the extracted feature from a cnn and add a dense layer to it.

    fe2 = Dense(nr_nodes, activation='relu')(cnn_part)
    if dropout:
        fe2 = Dropout(0.5)(fe2)
    input_desc = Input((sentence_length,))
    # Build the LSTM part of the model.
    se1 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)

    se2 = LSTM(nr_nodes)(se1)
    if dropout:
        se2 = Dropout(0.5)(se2)
        print(" ADDED DROPOUT  ")
    # Merge the two
    # Add or concatenate
    if concat_add == "concatenate":
        decoder1 = concatenate([fe2, se2])
    elif concat_add == "add":
        decoder1 = add([fe2, se2])

    # Add dense layer and prediction layer
    decoder2 = Dense(nr_nodes, activation='relu')(decoder1)
    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    # Build and compile the model
    model = k.Model(inputs=[cnn_input, input_desc], outputs=outputs)
    # if nr_gpus > 1:
    #     model = multi_gpu_model(model, gpus=nr_gpus)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


class zhouLayer():
    pass


def build_category_merge_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                               embedding_dimensions=200,
                               loss_function='categorical_crossentropy', optimizer='RMSprop', nr_nodes=256, nr_gpus=1,
                               concat_add='add', trainable_embedding=False, dropout=True, cat_embedding=False,
                               attribute_included=False):
    # Category input
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
    else:
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)

    # Textual input
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)

    if dropout:
        ce2 = Dropout(0.5)(ce2)
    se3 = LSTM(nr_nodes)(se2)
    if dropout:
        se3 = Dropout(0.5)(se3)
        print(" ADDED DROPOUT  ")
    # Merge the two
    # Add or concatenate
    if concat_add == "concatenate":
        decoder1 = concatenate([ce2, se3])
    elif concat_add == "add":
        decoder1 = add([ce2, se3])

    # Add dense layer and prediction layer
    decoder2 = Dense(nr_nodes, activation='relu')(decoder1)
    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    # Build and compile the model
    model = k.Model(inputs=[ce1, input_desc], outputs=outputs)
    # if nr_gpus > 1:
    #     model = multi_gpu_model(model, gpus=nr_gpus)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_category_parinject_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                                   embedding_dimensions=200, loss_function='categorical_crossentropy',
                                   optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add',
                                   trainable_embedding=False, dropout=True, cat_embedding=False, attribute_included=False):
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
    else:
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)
    ce3 = RepeatVector(sentence_length)(ce2)
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)

    model = concatenate([ce3, se2])

    # Encode --> add LSTM layers
    if dropout:
        model = Dropout(0.5)(model)
    model = LSTM(nr_nodes, activation='sigmoid')(model)

    # Add decoder / prediction layers
    model = Dense(nr_nodes, activation='relu')(model)
    if dropout:
        model = Dropout(0.5)(model)

    output = Dense(vocabulary_size, activation='softmax')(model)

    model = k.Model(inputs=[ce1, input_desc], outputs=output)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_category_brownlee_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                                  embedding_dimensions=200, loss_function='categorical_crossentropy',
                                  optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add',
                                  trainable_embedding=False, dropout=True, cat_embedding=False, attribute_included=False):
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
        print("Adding category embedding")
    else:
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)
    ce3 = RepeatVector(sentence_length)(ce2)
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)

    se3 = LSTM(nr_nodes, return_sequences=True)(se2)
    if dropout:
        se3 = Dropout(0.5)(se3)
        print("ADDED DROPOUT")
    se4 = Dense(nr_nodes)(se3)
    # embd4 = TimeDistributed(Dense(nodes_per_layer, activation='relu'))(embd3)

    merged = concatenate([ce3, se4])
    if dropout:
        merged = Dropout(0.5)(merged)

    # Language model'decoder
    lm2 = LSTM(nr_nodes * 2)(merged)
    lm3 = Dense(nr_nodes * 2, activation='relu')(lm2)

    if dropout:
        lm3 = Dropout(0.5)(lm3)

    outputs = Dense(vocabulary_size, activation='softmax')(lm3)

    # Tying it together
    model = k.Model(inputs=[ce1, input_desc], outputs=outputs)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_img_cat_merge_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                              embedding_dimensions=200, loss_function='categorical_crossentropy',
                              optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add',
                              trainable_embedding=False, dropout=True, cat_embedding=False, attribute_included=False):
    # Category input
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
    else:
        print("Adding category embedding")
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)


    # Textual input
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)

    if dropout:
        ce2 = Dropout(0.5)(ce2)
    se3 = LSTM(nr_nodes)(se2)
    if dropout:
        se3 = Dropout(0.5)(se3)
        print(" ADDED DROPOUT  ")

    # Image input
    fe1 = Input(shape=(2048,))
    fe2 = Dense(nr_nodes, activation='relu')(fe1)

    if dropout:
        fe2 = Dropout(0.5)(fe2)
    # Merge the three
    # Add or concatenate
    if concat_add == "concatenate":
        decoder1 = concatenate([ce2, se3, fe2])
    elif concat_add == "add":
        decoder1 = concatenate([ce2, se3, fe2])

    # Add dense layer and prediction layer
    decoder2 = Dense(nr_nodes, activation='relu')(decoder1)
    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    # Build and compile the model
    model = k.Model(inputs=[ce1, input_desc, fe1], outputs=outputs)
    # if nr_gpus > 1:
    #     model = multi_gpu_model(model, gpus=nr_gpus)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


# \\todo improve
def build_img_cat_parinject_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                                  embedding_dimensions=200, loss_function='categorical_crossentropy',
                                  optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add',
                                  trainable_embedding=False, dropout=True, cat_embedding=False, attribute_included=False):
    # Category input
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
    else:
        print("Adding category embedding")
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)

    ce3 = RepeatVector(sentence_length)(ce2)
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)
    # Image input
    fe1 = Input(shape=(2048,))
    fe2 = Dense(nr_nodes, activation='relu')(fe1)
    if dropout:
        fe2 = Dropout(0.5)(fe2)
    fe3 = RepeatVector(sentence_length)(fe2)

    model = concatenate([ce3, se2, fe3])

    # Encode --> add LSTM layers
    if dropout:
        model = Dropout(0.5)(model)
    model = LSTM(nr_nodes, activation='sigmoid')(model)

    # Add decoder / prediction layers
    model = Dense(nr_nodes, activation='relu')(model)
    if dropout:
        model = Dropout(0.5)(model)

    output = Dense(vocabulary_size, activation='softmax')(model)

    model = k.Model(inputs=[ce1, input_desc, fe1], outputs=output)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_img_cat_brownlee_model(nr_categories, sentence_length, vocabulary_size, embedding_matrix,
                                 embedding_dimensions=200, loss_function='categorical_crossentropy',
                                 optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add',
                                 trainable_embedding=False, dropout=True, cat_embedding=False, attribute_included=False):
    if cat_embedding == False:
        ce1 = Input(shape=(nr_categories,))
        ce2 = Dense(nr_nodes, activation='relu')(ce1)
    else:
        print("Adding category embedding")
        if attribute_included == False:
            ce1 = Input(shape=(1,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True)(ce1)
            ce2 = Reshape((nr_nodes,))(ce2)
        else:
            ce1 = Input(shape=(nr_categories,))
            ce2 = Embedding(nr_categories, nr_nodes, trainable=True, mask_zero=True)(ce1)
            ce2 = Flatten()(ce2)
            ce2 = Dense(nr_nodes, activation='relu')(ce2)
    ce3 = RepeatVector(sentence_length)(ce2)
    # Textual input
    input_desc = Input((sentence_length,))
    se2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)
    se3 = LSTM(nr_nodes, return_sequences=True)(se2)
    if dropout:
        se3 = Dropout(0.5)(se3)
        print("ADDED DROPOUT")
    se4 = Dense(nr_nodes)(se3)
    # Image input
    fe1 = Input(shape=(2048,))
    fe2 = Dense(nr_nodes, activation='relu')(fe1)
    if dropout:
        fe2 = Dropout(0.5)(fe2)
    fe3 = RepeatVector(sentence_length)(fe2)
    merged = concatenate([ce3, se4, fe3])
    if dropout:
        merged = Dropout(0.5)(merged)

    # Language model'decoder
    lm2 = LSTM(nr_nodes * 2)(merged)
    lm3 = Dense(nr_nodes * 2, activation='relu')(lm2)

    if dropout:
        lm3 = Dropout(0.5)(lm3)

    outputs = Dense(vocabulary_size, activation='softmax')(lm3)

    # Tying it together
    model = k.Model(inputs=[ce1, input_desc, fe1], outputs=outputs)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_brownlee_model(cnn_input, sentence_length, vocabulary_size, embedding_matrix, embedding_dimensions=200,
                         loss_function='categorical_crossentropy', optimizer='RMSprop', image_manipulation=False,
                         nodes_per_layer=256, dropout=True, trainable_embedding=False):
    fe2 = Dense(nodes_per_layer, activation='relu')(cnn_input)
    if dropout:
        fe2 = Dropout(0.5)(fe2)
    fe3 = RepeatVector(sentence_length)(fe2)

    input_desc = Input(shape=(sentence_length,))
    embd2 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                      mask_zero=True)(input_desc)
    embd3 = LSTM(nodes_per_layer, return_sequences=True)(embd2)
    if dropout:
        embd3 = Dropout(0.5)(embd3)
        print("ADDED DROPOUT")
    embd4 = Dense(nodes_per_layer)(embd3)
    # embd4 = TimeDistributed(Dense(nodes_per_layer, activation='relu'))(embd3)

    merged = concatenate([fe3, embd4])
    if dropout:
        merged = Dropout(0.5)(merged)

    # Language model'decoder
    lm2 = LSTM(nodes_per_layer * 2)(merged)
    lm3 = Dense(nodes_per_layer * 2, activation='relu')(lm2)

    if dropout:
        lm3 = Dropout(0.5)(lm3)

    outputs = Dense(vocabulary_size, activation='softmax')(lm3)

    # Tying it together
    model = k.Model(inputs=[cnn_input, input_desc], outputs=outputs)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_par_inject_model(cnn_input, sentence_length, vocabulary_size, embedding_matrix, embedding_dimensions=200,
                           loss_function='categorical_crossentropy', optimizer='RMSprop', image_manipulation=False,
                           nodes_per_layer=256, dropout=True, trainable_embedding=False):
    input_desc = Input((sentence_length,))

    # Get image feature
    fe1 = cnn_input
    fe2 = Dense(nodes_per_layer, activation='relu')(fe1)
    if dropout:
        print("ADDED DROPOUT")
        fe2 = Dropout(0.5)(fe2)

    # Embed the text
    embedding = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                          mask_zero=True)(input_desc)

    # Use RepeatVector to generate the correct layer size
    fe3 = RepeatVector(sentence_length)(fe2)

    # Concatenate the image features with the embedding
    model = concatenate([fe3, embedding])

    # Encode --> add LSTM layers
    if dropout:
        model = Dropout(0.5)(model)
    model = LSTM(nodes_per_layer, activation='sigmoid')(model)

    # Add decoder / prediction layers
    model = Dense(nodes_per_layer, activation='relu')(model)
    if dropout:
        model = Dropout(0.5)(model)

    output = Dense(vocabulary_size, activation='softmax')(model)

    model = k.Model(inputs=[cnn_input, input_desc], outputs=output)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_webshopincluded_model(cnn_input, cnn_part, sentence_length, vocabulary_size, embedding_matrix,
                                embedding_dimensions=200, nr_webshops=4, loss_function='categorical_crossentropy',
                                optimizer='RMSprop', nr_nodes=256, nr_gpus=1, concat_add='add', concat_moment="early",
                                trainable_embedding=False, webshop_embedding=False):
    # Use the extracted feature from a cnn and add a dense layer to it.
    fe1 = Dropout(0.5)(cnn_part)
    fe2 = Dense(nr_nodes, activation='relu')(fe1)
    input_desc = Input((sentence_length,))
    # Build the LSTM part of the model.
    se1 = Embedding(vocabulary_size, embedding_dimensions, trainable=trainable_embedding, weights=[embedding_matrix],
                    mask_zero=True)(input_desc)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(nr_nodes)(se2)

    # Add webshop input

    if webshop_embedding == False:
        we1 = Input(shape=(nr_webshops,))
        we2 = Dense(nr_nodes)(we1)
    else:
        we1 = Input(shape=(1,))
        we2 = Embedding(nr_webshops, nr_nodes, trainable=True)(we1)
        we2 = Reshape((nr_nodes,))(we2)  # ensures embedding dimensions are the same
    # Merge the three
    # Add or concatenate
    if concat_add == "concatenate":
        if concat_moment == "early":
            decoder1 = concatenate([we2, fe2, se3])
        else:
            decoder1 = concatenate([fe2, se3])
    elif concat_add == "add":
        if concat_moment == "early":
            decoder1 = add([we2, fe2, se3])
        else:
            decoder1 = add([fe2, se3])

    # Add dense layer and prediction layer
    decoder2 = Dense(nr_nodes, activation='relu')(decoder1)

    if concat_moment == "late":
        if concat_add == "add":
            decoder2 = add([decoder2, we2])
        elif concat_add == "concatenate":
            decoder2 = concatenate([decoder2, we2])

    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    # Build and compile the model
    model = k.Model(inputs=[we1, input_desc, cnn_input], outputs=outputs)
    # if nr_gpus > 1:
    #     model = multi_gpu_model(model, gpus=nr_gpus)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_attention_model(vocabulary_size, nr_nodes, embedding_dimensions, embedding_matrix, max_capt_length, feat_dims,
                          loss_function='categorical_crossentropy', optimizer='adam', run_eagerly=False):
    captions_input = Input(shape=(None,), name="captions_input")
    image_features_input = Input(shape=(feat_dims[0] * feat_dims[1], feat_dims[2]), name="image_features_input")
    # captions = Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
    #                      trainable=False)(captions_input)

    # This does not work; probably the order of the matrix and the vocabulary does not match or something
    captions = Embedding(vocabulary_size, embedding_dimensions, trainable=False,
                         weights=[embedding_matrix])(captions_input)

    averaged_image_features = Lambda(lambda x: kb.mean(x, axis=1))
    averaged_image_features = averaged_image_features(image_features_input)
    initial_state_h = Dense(nr_nodes)(averaged_image_features)
    initial_state_c = Dense(nr_nodes)(averaged_image_features)

    image_features = TimeDistributed(Dense(feat_dims[2], activation="relu"))(image_features_input)

    encoder = LSTM(nr_nodes, return_sequences=True, return_state=True, recurrent_dropout=0.1)
    attended_encoder = ExternalAttentionRNNWrapper(encoder, return_attention=True)

    output = TimeDistributed(Dense(vocabulary_size, activation="softmax"), name="output")

    # for training purpose
    attended_encoder_training_data, _, _, _ = attended_encoder([captions, image_features],
                                                               initial_state=[initial_state_h, initial_state_c])
    training_output_data = output(attended_encoder_training_data)
    kwargs = {"run_eagerly": True}
    training_model = k.Model(inputs=[captions_input, image_features_input], outputs=training_output_data)

    initial_state_inference_model = k.Model(inputs=[image_features_input], outputs=[initial_state_h, initial_state_c])

    inference_initial_state_h = Input(shape=(nr_nodes,))
    inference_initial_state_c = Input(shape=(nr_nodes,))
    attented_encoder_inference_data, inference_encoder_state_h, inference_encoder_state_c, inference_attention = attended_encoder(
        [captions, image_features],
        initial_state=[inference_initial_state_h, inference_initial_state_c]
    )

    inference_output_data = output(attented_encoder_inference_data)

    inference_model = k.Model(
        inputs=[image_features_input, captions_input, inference_initial_state_h, inference_initial_state_c],
        outputs=[inference_output_data, inference_encoder_state_h, inference_encoder_state_c, inference_attention])
    if run_eagerly:
        training_model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"], run_eagerly=True)
    else:
        training_model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return training_model, inference_model, initial_state_inference_model


class ExternalAttentionRNNWrapper(Wrapper):
    """
        The basic idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.
        This technique is similar to the input-feeding method described in the paper cited.
        The only difference compared to the AttentionRNNWrapper is, that this layer
        applies the attention layer not on the time-depending input but on a second
        time-independent input (like image clues) as described in:
            Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
            https://arxiv.org/abs/1502.03044
    """

    def __init__(self, layer, weight_initializer="glorot_uniform", return_attention=False, **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer
        self.return_attention = return_attention
        self._num_constants = None
        # print(" INITIALIZING PARENT CLASS ")
        super(ExternalAttentionRNNWrapper, self).__init__(layer, **kwargs)
        # print(" PARENT CLASS INITIALIZED ")
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

    def _validate_input_shape(self, input_shape):
        if len(input_shape) >= 2:
            if len(input_shape[:2]) != 2:
                raise ValueError(
                    "Layer has to receive two inputs: the temporal signal and the external signal which is constant for all time steps")
            if len(input_shape[0]) != 3:
                raise ValueError(
                    "Layer received a temporal input with shape {0} but expected a Tensor of rank 3.".format(
                        input_shape[0]))
            if len(input_shape[1]) != 3:
                raise ValueError(
                    "Layer received a time-independent input with shape {0} but expected a Tensor of rank 3.".format(
                        input_shape[1]))
        else:
            raise ValueError(
                "Layer has to receive at least 2 inputs: the temporal signal and the external signal which is constant for all time steps")

    def build(self, input_shape):
        print(" BUILDING THE MODEL  ")
        self._validate_input_shape(input_shape)

        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        temporal_input_dim = input_shape[0][-1]
        static_input_dim = input_shape[1][-1]

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape[0])[-1]

        self._W1 = self.add_weight(shape=(static_input_dim, temporal_input_dim), name="{}_W1".format(self.name),
                                   initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_W2".format(self.name),
                                   initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(temporal_input_dim + static_input_dim, temporal_input_dim),
                                   name="{}_W3".format(self.name), initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(temporal_input_dim,), name="{}_b2".format(self.name),
                                   initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(temporal_input_dim,), name="{}_b3".format(self.name),
                                   initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(temporal_input_dim, 1), name="{}_V".format(self.name),
                                  initializer=self.weight_initializer)
        print(" BUILDING PARENT CLASS  ")
        super(ExternalAttentionRNNWrapper, self).build()

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights

    def compute_output_shape(self, input_shape):
        # print(" COMPUTING OUTPUT SHAPE >> input_shape is  ", input_shape)
        self._validate_input_shape(input_shape)

        output_shape = self.layer.compute_output_shape(input_shape[0])
        # print(" output_shape is  ", output_shape)
        if self.return_attention:
            # print(" RETURNING ATTENTION >> ")
            if not isinstance(output_shape, list):
                output_shape = [output_shape]

            output_shape = output_shape + [(None, input_shape[1][1])]
            # print(" RETURNING ATTENTION >> ", output_shape)
        return output_shape

    def step(self, x, states):
        h = states[0]
        # states[1] necessary?

        # comes from the constants
        X_static = states[-2]
        # equals Kb.dot(static_x, self._W1) + self._b2 with X.shape=[bs, L, static_input_dim]
        total_x_static_prod = states[-1]
        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = kb.expand_dims(kb.dot(h, self._W2), 1)
        additive_atn = kb.tanh(total_x_static_prod) + kb.tanh(hw)
        attention = kb.softmax(kb.dot(additive_atn, self._V), axis=1)
        static_x_weighted = kb.sum(attention * X_static, [1])

        x = kb.dot(kb.concatenate([x, static_x_weighted], 1), self._W3) + self._b3
        h, new_states = self.layer.cell.call(x, states[:-2])
        # append attention to the states to "smuggle" it out of the RNN wrapper
        attention = kb.squeeze(attention, -1)
        h = kb.concatenate([h, attention])
        return h, new_states

    def call(self, x, constants=None, mask=None, initial_state=None):
        # print("   ARE WE USING 'CALL' HERE?   ")
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        # print(" INPUT SHAPE IS ", input_shape)
        # print(" x is ", len(x))
        if len(x) > 2:
            initial_state = x[2:]
            x = x[:2]
            assert len(initial_state) >= 1

        static_x = x[1]
        x = x[0]

        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]
        else:
            initial_states = self.layer.get_initial_state(x)

        if not constants:
            constants = []
        constants += self.get_constants(static_x)

        last_output, outputs, states = kb.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )

        # output has at the moment the form:
        # (real_output, attention)
        # split this now up

        output_dim = self.layer.compute_output_shape(input_shape)[0][-1]
        last_output = last_output[:output_dim]

        attentions = outputs[:, :, output_dim:]
        outputs = outputs[:, :, :output_dim]

        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output

            # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]
        # print( "<<< CALL OUTPUT >>>", len(output))
        return output

    def _standardize_args(self, inputs, initial_state, constants, num_constants):
        """Standardize `__call__` to a single list of tensor inputs.
        When running a model loaded from file, the input tensors
        `initial_state` and `constants` can be passed to `RNN.__call__` as part
        of `inputs` instead of by the dedicated keyword arguments. This method
        makes sure the arguments are separated and that `initial_state` and
        `constants` are lists of tensors (or None).
        # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None
        # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
        """
        print("INPUTS >>>> ", inputs, "  INIT_STATE   ", initial_state, " >> CONSTANTS >>", constants)

        if isinstance(inputs, list) and len(inputs) > 2:
            assert initial_state is None and constants is None
            if num_constants is not None:
                constants = inputs[-num_constants:]
                inputs = inputs[:-num_constants]
            initial_state = inputs[2:]
            inputs = inputs[:2]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)
        print("INPUTS >>>> ", inputs, "  INIT_STATE   ", initial_state, " >> CONSTANTS >>", constants)
        return inputs, initial_state, constants

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        # print(" USING '__CALL__' FUNCTION HERE   >> ")
        # print("  LEN INPUTS   >>>", len(inputs))

        if len(inputs) == 6:
            inputs, initial_state, constants = self._standardize_args(inputs, initial_state, constants,
                                                                      self._num_constants)
        # print("INPUTS >>>> ", inputs, "  INIT_STATE   ", initial_state, " >> CONSTANTS >>", constants)
        if initial_state is None and constants is None:
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            # print(" INIT STATE NOT NONE, ADDING IT TO KWARGS AND EXTENDING ADDITIONAL INPUTS PARAMETER >> ")
            kwargs['initial_state'] = initial_state
            additional_inputs += (initial_state)
            self.state_spec = [InputSpec(shape=kb.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=kb.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        # print(" ADDITIONAL INPUTS  ", additional_inputs[0], type(additional_inputs[0]))
        is_keras_tensor = kb.is_keras_tensor(additional_inputs[0])
        # print(" WAAROM IS DIT NIET TRUE: ", is_keras_tensor)
        for tensor in additional_inputs:
            if kb.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an ExternalAttentionRNNWrapper'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # print(" MAKING CALL TO PARENT CLASS WITH INPUTS AS KERAS TENSOR")
            # Compute the full input spec, including state and constants
            full_input = inputs + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            # try:
            output = super(ExternalAttentionRNNWrapper, self).__call__(full_input, **kwargs)
            # except ValueError:
            #     output = super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)
            self.input_spec = self.input_spec[:len(original_input_spec)]
            return output
        else:
            # print(" MAKING CALL TO PARENT CLASS, INPUTS ARE NOT A KERAS TENSOR  >> ")
            # print(len(inputs))
            # print(" KWARGS  >>>", kwargs.keys())
            # print("  LEN INPUT SPEC   ", len(self.input_spec))

            # if len(inputs) == 4:
            #     inputs = inputs[0:2]
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, kb.dot(x, self._W1) + self._b2]
        return constants

    def get_config(self):
        config = {'return_attention': self.return_attention, 'weight_initializer': self.weight_initializer}
        base_config = super(ExternalAttentionRNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
