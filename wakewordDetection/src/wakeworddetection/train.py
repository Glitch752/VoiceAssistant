DATA_FILE = './processed/data.npz'

def train_model():
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load(DATA_FILE)
    X = data['X']
    Y = data['y']

    # Maybe we should be normalizing? I'm not sure.
    # X = X.astype('float32') / np.max(np.abs(X))

    # Train/val split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    # Add channel dim, e.g. (32, 13) to (32, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    import tensorflow as tf
    # It's an IDE issue if keras isn't found; don't worry about it
    from tensorflow.keras import layers, models, callbacks

    input_shape = X_train.shape[1:]  # e.g., (32, 32, 1) if MFCC or spectrogram

    model = models.Sequential([
        # I pulled this out of nowhere, it's probably not the best layer configuration
        layers.InputLayer(input_shape),
        
        # layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.2),
        # layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.2),
        # layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.GlobalAveragePooling2D(),

        # layers.Dense(32, activation='relu'),
        # layers.Dropout(0.2),
        # layers.Dense(1, activation='sigmoid')
        
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True) # Stop training if no improvement
        ]
    )
    
    model.save('./processed/wakeword_model.keras')