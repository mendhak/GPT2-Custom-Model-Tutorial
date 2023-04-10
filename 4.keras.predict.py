from tensorflow import keras
import keras_nlp
output_dir = "keras_model_output"
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model = keras.models.load_model(output_dir+"/output.h5", custom_objects={'perplexity_loss': perplexity})
model.summary()

model.predict(["This is a text prompt"])