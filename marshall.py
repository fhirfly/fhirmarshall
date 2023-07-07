from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModel

def create_model():
    # Transformer encoder for text data
    text_inputs = layers.Input(shape=(), dtype=tf.string, name='text')
    transformer_model = TFAutoModel.from_pretrained("bert-base-uncased")
    transformer_outputs = transformer_model(text_inputs)
    text_outputs = layers.Dense(128, activation='relu')(transformer_outputs[0][:, 0, :])
    
    # Feed-forward network for numerical and categorical data
    num_inputs = layers.Input(shape=(num_features,), name='numeric')
    cat_inputs = layers.Input(shape=(cat_features,), name='categorical')
    merged = layers.concatenate([num_inputs, cat_inputs])
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    
    # Concatenate outputs and add final dense layers
    merged = layers.concatenate([text_outputs, x])
    outputs = layers.Dense(16, activation='relu')(merged)
    outputs = layers.Dense(1, activation='sigmoid')(outputs)
    
    model = Model(inputs=[text_inputs, num_inputs, cat_inputs], outputs=[outputs])

    return model

# Compile the model
model = create_model()
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
