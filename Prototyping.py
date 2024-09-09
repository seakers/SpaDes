import keras
import keras_nlp
import numpy as np
import tensorflow as tf


# Create a single transformer decoder layer.
decoder = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=2, num_heads=3)

# Create a simple model containing the decoder.
decoder_input = keras.Input(shape=(2, 3))
tf.print(decoder_input)
encoder_input = keras.Input(shape=(2, 3))
tf.print(encoder_input)
output = decoder(decoder_input, encoder_input)
tf.print(output)
model = keras.Model(
    inputs=(decoder_input, encoder_input),
    outputs=output,
)

# Call decoder on the inputs.
decoder_input_data = np.random.uniform(size=(2, 2, 3))
print(decoder_input_data.shape)
tf.print(decoder_input_data)
encoder_input_data = np.random.uniform(size=(2, 2, 3))
print(encoder_input_data.shape)
tf.print(encoder_input_data)
decoder_output = model((decoder_input_data, encoder_input_data))
print(decoder_output.shape)
tf.print(decoder_output)

# import tensorflow as tf
# from keras import layers
# from keras_nlp import layers as layers_nlp

# class RLModel(tf.keras.Model):
#     def __init__(self, num_components, num_panels, embedding_dim=128, num_heads=4, ff_dim=256, num_layers=3, **kwargs):
#         super(RLModel, self).__init__(**kwargs)
#         self.num_components = num_components
#         self.sequence_length = 4 * num_components  # Total length
#         self.embedding_dim = embedding_dim
        
#         # Embedding for categorical features
#         self.panel_embedding = layers.Embedding(input_dim=num_panels, output_dim=embedding_dim)  # For panelChoice
#         self.orientation_embedding = layers.Embedding(input_dim=24, output_dim=embedding_dim)  # For orientation
        
#         # Transformer Decoder Layers
#         self.decoder_layers = [
#             layers_nlp.TransformerDecoder(intermediate_dim=ff_dim, num_heads=num_heads)
#             for _ in range(num_layers)
#         ]
        
#         # Output layers for each type of action
#         self.panel_output_layer = layers.Dense(num_panels, activation='softmax')  # Panel choices
#         self.x_output_layer = layers.Dense(51, activation='softmax')  # x-locations
#         self.y_output_layer = layers.Dense(51, activation='softmax')  # y-locations
#         self.orientation_output_layer = layers.Dense(24, activation='softmax')  # Orientations

#     def call(self, inputs, action_type=0, training=False):
#         # Split inputs into categorical and continuous components
#         panel_choice, x_loc, y_loc, orientation = tf.split(inputs, num_or_size_splits=4, axis=-1)
        
#         # Embed categorical inputs
#         panel_emb = self.panel_embedding(tf.cast(panel_choice, tf.int32))
#         orientation_emb = self.orientation_embedding(tf.cast(orientation, tf.int32))
        
#         # Concatenate embeddings and continuous inputs
#         x = tf.concat([panel_emb, tf.expand_dims(x_loc, -1), tf.expand_dims(y_loc, -1), orientation_emb], axis=-1)
        
#         # Pass through Transformer Decoder layers
#         for decoder_layer in self.decoder_layers:
#             x = decoder_layer(x, training=training)
        
#         # Output layer based on action type
#         if action_type == 0:
#             return self.panel_output_layer(x)  # Panel choice
#         elif action_type == 1:
#             return self.x_output_layer(x)  # x-location
#         elif action_type == 2:
#             return self.y_output_layer(x)  # y-location
#         elif action_type == 3:
#             return self.orientation_output_layer(x)  # Orientation
#         else:
#             raise ValueError("Invalid action type. Must be 0, 1, 2, or 3.")

# # Hyperparameters and example usage
# num_components = 15
# num_panels = 5

# # Initialize model
# model = RLModel(num_components=num_components, num_panels=num_panels)

# # Example input: Batch of sequences (batch_size, sequence_length)
# inputs = tf.zeros((1, 1, 4 * num_components), dtype=tf.float32)

# # Get the model output for different actions
# for i in range(60):  # 0: panel, 1: x, 2: y, 3: orientation
#     action_type = i % 4
#     output = model(inputs, action_type=action_type)
#     print(f"Output for action type {action_type}:", tf.print(output))  # Should be (batch_size, sequence_length, num_classes_for_action)
#     print("Shape:", output.shape)  # Should be (batch_size, sequence_length, num_classes_for_action)


# # class SpacecraftDesignTransformer(tf.keras.Model):
# #     def __init__(self, num_components, num_panels, embedding_dim=128, num_heads=4, ff_dim=256, num_layers=3, **kwargs):
# #         super(SpacecraftDesignTransformer, self).__init__(**kwargs)
# #         self.num_components = num_components
# #         # self.sequence_length = 4 * num_components  # Total length
# #         self.sequence_length = 1 # one action at a time
# #         self.embedding_dim = embedding_dim
        
# #         # Embedding for categorical features
# #         self.panel_embedding = layers.Embedding(input_dim=num_panels, output_dim=embedding_dim)  # For panelChoice
# #         self.orientation_embedding = layers.Embedding(input_dim=24, output_dim=embedding_dim)  # For orientation
        
# #         # Transformer Decoder Layers
# #         self.decoder_layers = [
# #             layers_nlp.TransformerDecoder(intermediate_dim=ff_dim, num_heads=num_heads)
# #             for _ in range(num_layers)
# #         ]
        
# #         # Output layers for each type of action
# #         self.panel_output_layer = layers.Dense(num_panels, activation='softmax')  # Panel choices
# #         self.x_output_layer = layers.Dense(51, activation='softmax')  # x-locations
# #         self.y_output_layer = layers.Dense(51, activation='softmax')  # y-locations
# #         self.orientation_output_layer = layers.Dense(24, activation='softmax')  # Orientations

# #     def call(self, inputs, act=0, training=False):
# #         # Split inputs into categorical and continuous components
# #         panel_choice, x_loc, y_loc, orientation = tf.split(inputs, num_or_size_splits=4, axis=-1)
        
# #         # Embed categorical inputs
# #         panel_emb = self.panel_embedding(tf.cast(panel_choice, tf.int32))
# #         orientation_emb = self.orientation_embedding(tf.cast(orientation, tf.int32))
        
# #         # Concatenate embeddings and continuous inputs
# #         x = tf.concat([panel_emb, tf.expand_dims(x_loc, -1), tf.expand_dims(y_loc, -1), orientation_emb], axis=-1)
        
# #         # Pass through Transformer Decoder layers
# #         for decoder_layer in self.decoder_layers:
# #             x = decoder_layer(x, training=training)
        
# #         # Output layer based on action type
# #         if act == 0:
# #             x = self.panel_output_layer(x)  # Panel choice
# #         elif act == 1:
# #             x = self.x_output_layer(x)  # x-location
# #         elif act == 2:
# #             x = self.y_output_layer(x)  # y-location
# #         elif act == 3:
# #             x = self.orientation_output_layer(x)  # Orientation
        
# #         return x

# # # Hyperparameters and example usage
# # num_components = 15
# # num_panels = 5

# # # Initialize model
# # model = SpacecraftDesignTransformer(num_components=num_components, num_panels=num_panels)

# # # Example input: Batch of sequences (batch_size, sequence_length)
# # inputs = tf.zeros((1, 4 * num_components), dtype=tf.float32)

# # # Get the model output for different actions
# # for act in range(4):  # 0: panel, 1: x, 2: y, 3: orientation
# #     output = model(inputs, act=act)
# #     print(f"Output for action {act}:")
# #     print("Shape:", output.shape)  # Should be (batch_size, sequence_length, num_classes_for_action)
# #     print(tf.print(output))



# # import tensorflow as tf
# # from keras import layers as layersKeras
# # from keras_nlp import layers

# # class SpacecraftDesignTransformer(tf.keras.Model):
# #     def __init__(self, num_components, num_panels, embedding_dim=128, num_heads=4, ff_dim=256, num_layers=3, **kwargs):
# #         super(SpacecraftDesignTransformer, self).__init__(**kwargs)
# #         self.num_components = num_components
# #         self.sequence_length = 4 * num_components  # panelChoice, xLoc, yLoc, orientation for each component
# #         self.embedding_dim = embedding_dim
        
# #         # Embedding for categorical features
# #         self.panel_embedding = layersKeras.Embedding(input_dim=num_panels, output_dim=embedding_dim)  # For panelChoice
# #         self.orientation_embedding = layersKeras.Embedding(input_dim=24, output_dim=embedding_dim)  # For orientation

# #         # Transformer Decoder Layers
# #         self.decoder_layers = [
# #             layers.TransformerDecoder(intermediate_dim=ff_dim, num_heads=num_heads)
# #             for _ in range(num_layers)
# #         ]
        
# #         # Output layers
# #         self.output_layer = layersKeras.Dense(num_panels + 51 + 51 + 24, activation='softmax')  # Outputs distribution over all categories

# #     def call(self, inputs, training=False):
# #         # Split inputs into categorical and continuous components
# #         panel_choice, x_loc, y_loc, orientation = tf.split(inputs, num_or_size_splits=4, axis=-1)
        
# #         # Embed categorical inputs
# #         panel_emb = self.panel_embedding(panel_choice)
# #         orientation_emb = self.orientation_embedding(orientation)
        
# #         # Concatenate embeddings and continuous inputs
# #         x = tf.concat([panel_emb, tf.expand_dims(x_loc, -1), tf.expand_dims(y_loc, -1), orientation_emb], axis=-1)
        
# #         # Pass through Transformer Decoder layers
# #         for decoder_layer in self.decoder_layers:
# #             x = decoder_layer(x, training=training)
        
# #         # Output layer to predict the next token in the sequence
# #         x = self.output_layer(x)
# #         tf.print(x)
# #         return x

# # # Hyperparameters and example usage
# # num_components = 15
# # num_panels = 6

# # # Initialize model
# # model = SpacecraftDesignTransformer(num_components=num_components, num_panels=num_panels)

# # # Define parameters
# # batch_size = 1
# # num_components = 15
# # sequence_length = 4 * num_components  # Total number of properties for all components

# # # Create a tensor filled with zeros
# # inputs = tf.zeros((batch_size, sequence_length), dtype=tf.float32)

# # # Set the first element to 1
# # inputs = tf.tensor_scatter_nd_update(inputs, [[0, 0]], [1.0])

# # # Get the model output
# # output = model(inputs)
# # print(output.shape)  # Expected shape: (batch_size, sequence_length, output_vocab_size)
