
import itertools
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

def batched_it(iterable, n):
    "Batch data into iterators of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)


paths = [str(x) for x in Path("./en_corpus/").glob("**/*.txt")]
save_path = 'tokenized_data'
# loading tokenizer from the saved model path
tokenizer = GPT2Tokenizer.from_pretrained(save_path)

tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})

# creating the configurations from which the model can be made

config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)

# creating the model
model = TFGPT2LMHeadModel(config)

for filenames in batched_it(paths[0:10000], 1000):
    single_string = ''
    for filename in list(filenames):
        with open(filename, "r", encoding='utf-8') as f:
            x = f.read()  
            single_string += x + tokenizer.eos_token
   
    string_tokenized = tokenizer.encode(single_string)

    print("\\", end=None)

    examples = []
    block_size = 100
    BATCH_SIZE = 12
    BUFFER_SIZE = 1000
    for i in range(0, len(string_tokenized) - block_size + 1, block_size):
        examples.append(string_tokenized[i:i + block_size])
    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print("/", end=None)


    # Model training

    # defining our optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    # definining our loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # defining our metric which we want to observe
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # compiling the model
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

    num_epoch = 10
    history = model.fit(dataset, epochs=num_epoch)


    # Save the model

    from transformers import WEIGHTS_NAME, CONFIG_NAME
    import os
    output_dir = './model_custom/'
    # creating directory if it is not present
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # save model and model configs
    model.save_pretrained(output_dir)
    model_to_save.config.to_json_file(output_config_file)
    # save tokenizer
    tokenizer.save_pretrained(output_dir)