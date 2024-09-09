import numpy as np
import tensorflow as tf
import keras
from keras import layers

def transformer_block(x, head_size, num_heads, ff_dim, dropout=0):
    # Attention block
    attn_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.Add()([x, attn_output])
    out1 = layers.LayerNormalization(epsilon=1e-6)(out1)

    # Feed Forward Network
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(x.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    out2 = layers.Add()([out1, ffn_output])
    return layers.LayerNormalization(epsilon=1e-6)(out2)

def build_sequential_action_model(input_shape, panel_options, loc_options, orientation_options, embed_dim, head_size, num_heads, ff_dim, num_blocks, dropout=0):
    # Inputs
    sequence_input = layers.Input(shape=input_shape, name="sequence_input")
    action_input = layers.Input(shape=(1,), name="action_input")  # Indicates the current action to predict

    # Embedding Layer
    x = layers.Embedding(input_dim=panel_options + loc_options + orientation_options, output_dim=embed_dim)(sequence_input)
    x = layers.Dropout(dropout)(x)

    # Transformer Blocks
    for _ in range(num_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    # Panel Prediction (12 options)
    panel_output = layers.Dense(panel_options, activation="softmax", name="panel_output")(x)

    # X and Y Predictions (51 options each, continuous)
    x_output = layers.Dense(1, activation="tanh", name="x_output")(x)
    y_output = layers.Dense(1, activation="tanh", name="y_output")(x)

    # Orientation Prediction (24 options)
    orientation_output = layers.Dense(orientation_options, activation="softmax", name="orientation_output")(x)

    # Stack outputs
    combined_output = layers.Concatenate(axis=-1)([panel_output, x_output, y_output, orientation_output])

    # Select the output corresponding to the action input
    def select_output(inputs):
        combined, action = inputs
        action_index = tf.cast(action[0, 0], tf.int32)  # Ensure action index is integer
        # Select the correct range of combined_output based on the action_input
        if action_index == 0:  # Panel prediction
            return combined[:, :panel_options]
        elif action_index == 1:  # X prediction
            return combined[:, panel_options:panel_options + 1]
        elif action_index == 2:  # Y prediction
            return combined[:, panel_options + 1:panel_options + 2]
        elif action_index == 3:  # Orientation prediction
            return combined[:, -orientation_options:]
        else:
            raise ValueError("Invalid action index")

    # Output selection based on action_input
    selected_output = layers.Lambda(select_output)([combined_output, action_input])

    # Create Model
    model = keras.Model(inputs=[sequence_input, action_input], outputs=selected_output)
    return model


# Model Parameters
input_shape = (None,)  # Variable-length input sequences
panel_options = 12  # Number of panel choices
loc_options = 51  # Discretized x and y locations
orientation_options = 24  # Number of orientation choices
embed_dim = 64
head_size = 64
num_heads = 4
ff_dim = 128
num_blocks = 2
dropout = 0.1

# Build the model
model = build_sequential_action_model(input_shape, panel_options, loc_options, orientation_options, embed_dim, head_size, num_heads, ff_dim, num_blocks, dropout)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.summary()

# Function to run the model for each action step
def predict_action(model, current_sequence, action_type):
    action_input = np.array([[action_type]])  # Specify which action to predict
    predicted_output = model.predict([current_sequence, action_input])
    return predicted_output

# Initialize the sequence with an empty state
sequence = np.zeros((1, 1))  # Start with an empty or zero state

# Simulate placing two components
placements = []

for component in range(2):  # Loop for two components
    # Step 1: Predict the panel choice (12 options)
    panel_choice = predict_action(model, sequence, 0)
    panel_choice = np.argmax(panel_choice)  # Choose the panel with the highest probability
    placements.append(panel_choice)
    print(f"Component {component + 1} Panel Choice: {panel_choice}")
    
    # Step 2: Predict the x location (-1 to 1)
    x_loc_choice = predict_action(model, sequence, 1)
    x_loc_choice = np.clip(x_loc_choice, -1, 1)  # Ensure it falls within -1 and 1
    placements.append(x_loc_choice[0, 0])  # Extract the single predicted value
    print(f"Component {component + 1} X Location: {x_loc_choice[0, 0]}")

    # Step 3: Predict the y location (-1 to 1)
    y_loc_choice = predict_action(model, sequence, 2)
    y_loc_choice = np.clip(y_loc_choice, -1, 1)  # Ensure it falls within -1 and 1
    placements.append(y_loc_choice[0, 0])
    print(f"Component {component + 1} Y Location: {y_loc_choice[0, 0]}")
    
    # Step 4: Predict the orientation (24 options)
    orientation_choice = predict_action(model, sequence, 3)
    orientation_choice = np.argmax(orientation_choice)  # Choose the orientation with the highest probability
    placements.append(orientation_choice)
    print(f"Component {component + 1} Orientation: {orientation_choice}")
    
    # Update sequence with the chosen actions
    # Expand sequence for the next component by appending the latest choices
    sequence = np.append(sequence, [[panel_choice, x_loc_choice[0, 0], y_loc_choice[0, 0], orientation_choice]], axis=1)

# Print final placements
print("Final placements for two components:", placements)



# # Define a sample input sequence (e.g., representing some partial configuration)
# sample_sequence = np.array([[1, 2, 3]])  # Example sequence of integers representing current state

# # Define the action to predict:
# # 0 - panel, 1 - x position, 2 - y position, 3 - orientation
# action_to_predict = 0  # Start by predicting the "panel" choice

# # Build the model using the provided function
# input_shape = (None,)  # Variable-length input sequences
# num_tokens = 100  # Example number for total tokens (panels, orientations, etc.)
# embed_dim = 64
# head_size = 64
# num_heads = 4
# ff_dim = 128
# num_blocks = 2
# dropout = 0.1

# # Build the model
# model = build_sequential_action_model(input_shape, num_tokens, embed_dim, head_size, num_heads, ff_dim, num_blocks, dropout)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# # Convert the action input to a numpy array
# action_input = np.array([[action_to_predict]])

# # Run the model for a single prediction
# predicted_output = model.predict([sample_sequence, action_input])

# # Print the predicted output
# print("Predicted Output for Action", action_to_predict, ":", predicted_output)

# # You can repeat the process by updating the action input
# # For example, to predict the x position next, set action_to_predict = 1
# action_to_predict = 1
# action_input = np.array([[action_to_predict]])

# # Predict the x location
# predicted_output_x = model.predict([sample_sequence, action_input])
# print("Predicted Output for Action", action_to_predict, ":", predicted_output_x)

# # Continue for y position and orientation predictions similarly.

# # Model Parameters
# input_shape = (None,)  # Variable-length input sequences
# num_tokens = 100  # Example number for total tokens (panels, orientations, etc.)
# embed_dim = 64
# head_size = 64
# num_heads = 4
# ff_dim = 128
# num_blocks = 2
# dropout = 0.1

# # Build Model
# model = build_sequential_action_model(input_shape, num_tokens, embed_dim, head_size, num_heads, ff_dim, num_blocks, dropout)
# model.summary()
