#!/usr/bin/env python
# coding: utf-8

# # Trading News and Corporate Actions with BERT
# 
# <a href="" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>
# 
# 
# <a href="" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# Algo-trading on corporate actions by leveraging NLP. A replicationa and enhancement of the paper: *Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading (Zhou et al., Findings 2021)*.
# 
# We will perform the following steps:
# 1. Domain adaptation for financial articles by finetuning a BERT model with Masked Language Model (MLM) training on financial news and encyclopedia data. *Zhou et al.* utilized human annotators to label news articles with an event.
# 1. Bi-Level Event Detection: At Token-Level we detect events using a sequence labeling approach. At the higher Article-Level we will augment the corpus with 'CLS' token embedding which contains the the aggregate of all the article's embeddings, and concatenate it with the lower level tokens.
# 1. Recognize security Ticker, using string matching algorithm to recognize tickers within articles.
# 1. Create trading signals on the identified tickers.

# ```bibtex
# @inproceedings{zhou-etal-2021-trade,
#     title = "Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading",
#     author = "Zhou, Zhihan  and
#       Ma, Liqian  and
#       Liu, Han",
#     editor = "Zong, Chengqing  and
#       Xia, Fei  and
#       Li, Wenjie  and
#       Navigli, Roberto",
#     booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
#     month = aug,
#     year = "2021",
#     address = "Online",
#     publisher = "Association for Computational Linguistics",
#     url = "https://aclanthology.org/2021.findings-acl.186",
#     doi = "10.18653/v1/2021.findings-acl.186",
#     pages = "2114--2124",
# }
# ```

# # Notebook Environment
# 
# For a unified research environment, enable the flags below:

# In[ ]:


UPGRADE_PY = False
INSTALL_DEPS = False
if INSTALL_DEPS:
  # !pip install -q tensorboard==2.15.2
  get_ipython().system('pip install -q tensorflow[and-cuda]==2.15.1')
  # !pip install -q tensorflow==2.15.0
  get_ipython().system('pip install -q tensorflow-io-gcs-filesystem==0.36.0')
  get_ipython().system('pip install -q tensorflow-text==2.15.0')
  get_ipython().system('pip install -q tf_keras==2.15.1')
  get_ipython().system('pip install -q tf-slim==1.1.0')
  get_ipython().system('pip install -q tokenizers==0.15.2')
  get_ipython().system('pip install -q torch==2.2.0+cpu')
  get_ipython().system('pip install -q torch-xla==2.2.0+libtpu')
  get_ipython().system('pip install -q torchdata==0.7.1')
  get_ipython().system('pip install -q transformers==4.38.2')

if UPGRADE_PY:
    get_ipython().system('mamba create -n py311 -y')
    get_ipython().system('source /opt/conda/bin/activate py312 && mamba install python=3.11 jupyter mamba -y')

    get_ipython().system('sudo rm /opt/conda/bin/python3')
    get_ipython().system('sudo ln -sf /opt/conda/envs/py312/bin/python3 /opt/conda/bin/python3')
    get_ipython().system('sudo rm /opt/conda/bin/python3.10')
    get_ipython().system('sudo ln -sf /opt/conda/envs/py312/bin/python3 /opt/conda/bin/python3.10')
    get_ipython().system('sudo rm /opt/conda/bin/python')
    get_ipython().system('sudo ln -sf /opt/conda/envs/py312/bin/python3 /opt/conda/bin/python')

get_ipython().system('python --version')


# In[ ]:


import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Transformers cannot use keras3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
IN_KAGGLE = IN_COLAB = False
get_ipython().system('export CUDA_LAUNCH_BLOCKING=1')
get_ipython().system('export XLA_FLAGS=--xla_cpu_verbose=0')

try:
  # https://www.tensorflow.org/install/pip#windows-wsl2
  import google.colab
  from google.colab import drive
  drive.mount('/content/drive')
  DATA_PATH = "/content/drive/MyDrive/EDT dataset"
  IN_COLAB = True
  print('Colab!')
except:
  IN_COLAB = False
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ and not IN_COLAB:
    print('Running in Kaggle...')
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    DATA_PATH = "/kaggle/input"
    IN_KAGGLE = True
    print('Kaggle!')
elif not IN_COLAB and not IN_KAGGLE:
    IN_KAGGLE = False
    DATA_PATH = "./data/"
    print('Normal!')

MODEL_PATH = "google-bert/bert-base-cased"


# # Accelerators Configuration
# 
# If you have a GPU, TPU or in one of the collaborative notebooks. Configure your setup below:

# In[ ]:


import numpy as np
import math
import shutil
import pandas as pd

from pathlib import Path
import re
import pickle
from copy import deepcopy

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision

print(f'Tensorflow version: [{tf.__version__}]')

tf.get_logger().setLevel('INFO')

#tf.config.set_soft_device_placement(True)
#tf.config.experimental.enable_op_determinism()
#tf.random.set_seed(1)
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
except Exception as e:
  # Not an exception, just no TPUs available, GPU is fallback
  # https://www.tensorflow.org/guide/mixed_precision
  print(e)
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_global_policy(policy)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) > 0:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy()

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
    finally:
        print("Running on", len(tf.config.list_physical_devices('GPU')), "GPU(s)")
  else:
    # CPU is final fallback
    strategy = tf.distribute.get_strategy()
    print("Running on CPU")

def is_tpu_strategy(strategy):
    return isinstance(strategy, tf.distribute.TPUStrategy)

print("Number of accelerators:", strategy.num_replicas_in_sync)
os.getcwd()


# # Wrangling the Data
# 
# Our corpus will be processed and labelled to 11 types of corporate events:
# 1. Acquisition(A)
# 1. Clinical Trial(CT)
# 1. Regular Dividend(RD)
# 1. Dividend Cut(DC)
# 1. Dividend Increase(DI)
# 1. Guidance Increase(GI)
# 1. New Contract(NC)
# 1. Reverse Stock Split(RSS)
# 1. Special Dividend(SD)
# 1. Stock Repurchase(SR)
# 1. Stock Split(SS).
# 1. No event (O)
# 
# Articles are structured as follows:
# 
# ```json
# 'title': 'Title',
# 'text': 'Text Body',
# 'pub_time': 'Published datetime',
# 'labels': {
#     'ticker': 'Security symbol',
#     'start_time': 'First trade after article published',
#     'start_price_open': 'The "Open" price at start_time',
#     'start_price_close': 'The "Close" price at start_time',
#     'end_price_nday': 'The "Close" price at the last minute of the following 1-3 trading day. If early than 4pm ET its the same day. Otherwise, it refers to the next trading day.',
#     'end_time_1-3day': 'The time corresponds to end_price_1day',
#     'highest_price_nday': 'The highest price in the following 1-3 trading',
#     'highest_time_nday': 'The time corresponds to highest_price_1-3day',
#     'lowest_price_nday': 'The lowest price in the following 1-3 trading day',
#     'lowest_time_nday': 'The time corresponds to lowest_price_1-3day',
# }
# ```

# In[ ]:


import tensorflow as tf
import transformers

MAX_LEN = 256 # Default 256
SPECIAL_TOKEN = '[CLS]'
NUM_LABELS = 12 # See Labels description above.
LEARN_RATE=5e-5
EPOCHS=100
BATCH_SIZE = 2 * strategy.num_replicas_in_sync # Default 8


# ### Tokenizing News Text
# 
# The text body is tokenized, a simple example is shown below, including how the '[CLS]' (classification problem token) token is leveraged. BERT's transformer inputs expect of shape [batch_size, seq_length] the following inputs:
# - "input_ids": token ids of the input sequences.
# - "attention_mask": has value 1 at the position of all input tokens present before padding and value 0 for the padding tokens.
# - "token_type_ids": the index of the input that created the input token. The first input segment (index 0) includes the start-of-sequence token and its end-of-segment token. The second segment (index 1, if present) includes its end-of-segment token. Padding tokens get index 0.
# 
# Transformers use self-attention mechanisms represent interactions amongst tokens and their contextual information in the input sequence as a weighted-sum. With this mechanism higher layers of the network will aggregate information from all other tokens in the sequence, in our case '[CLS]' will have such information.
# 
# Since we are passing the tokens as a batch, we need to give the tokenizer a maximum length on which to either PAD or TRUNCATE.

# In[ ]:


from transformers import BertTokenizerFast, TFBertModel, BertConfig

# https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#berttokenizerfast
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = TFBertModel.from_pretrained(MODEL_PATH)

text = ["When taken as a whole, the evidence suggests Cramer recommends “hot” stocks", "lending credence to the Hot Hand Fallacy in this context."]

tokenized_sequence = tokenizer.tokenize(text)
tokenized_sequence


# Note how words that weren't in the original model's vocabulary get split using '##', e.g. being *Cramer*'s name.
# 
# Running the full tokenizer will get the token IDs, their position (type ID) and the attention mask for BERT:

# In[ ]:


sample_inputs = inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    max_length=MAX_LEN,  # Maximum length for padding/truncation, adjust as needed
    padding='max_length',
    return_tensors='tf',
    truncation=True
)
sample_inputs


# Below is a textual representation of what the model will see, [CLS] showing the start of the clasification sequence, [SEP] to seperate sequences and [PAD]  to make the word embeddings the same size for batched predictions.

# In[ ]:


tokenizer.decode(sample_inputs["input_ids"].numpy()[0])


# The padding is signalled to be ignored by the model through the attention mask:

# In[ ]:


sample_inputs["attention_mask"].shape


# In[ ]:


sample_inputs["attention_mask"]


# Since we are passing sequences, the model will need to know where one starts and another ends. This is signalled by sequence IDs.
# 
# Below we see that the first sequence is `0`, and the next is `1`, these are relevent in sequence classification, which we will be doing in this article:

# In[ ]:


sample_inputs['token_type_ids']


# If we run this from an out-of-the-box model forward pass, we get a sequence of tokens, though nothing is being predicted for now as there are no dense layers or a classification head yet - we only get the batched sequence as outputted by the last activation layer of the BERT model
# 
# The line `embedding = hidden_state[:, 0, :]` extracts the embeddings corresponding to the [CLS] token, which is often used as a representation of the entire sequence for classification tasks.

# In[ ]:


outputs = model(sample_inputs['input_ids'])
hidden_state = outputs.last_hidden_state
embedding = hidden_state[:, 0, :]


# Get the hidden state with all info.


# The paper authors provided a labelled dataset which we will process and prepare for the BERT inputs.
# 
# In our classification, we will hit some unknown tokens. The authors use ID -100 to denote this, we will use the pretrained tokenizer's ID:

# In[ ]:


from transformers import BertTokenizerFast
UNK_ID = None
OTHER = 'O'

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings, tag2id, unk=UNK_ID):
    encoded_labels = []
    for doc_labels, doc_offset in zip(tags, encodings.offset_mapping):
        # Initialize a label array for this document
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * unk
        label_index = 0

        for i, (start, end) in enumerate(doc_offset):
            # If start != 0, it's not the start of a new word, so we continue
            if start == 0:  # New word
                if label_index < len(doc_labels):
                    doc_enc_labels[i] = tag2id.get(doc_labels[label_index], unk)
                    label_index += 1
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def load_seq_data_from_tsv(path):
    file = open(path, "r", encoding="utf-8-sig")
    lines = list(csv.reader(file, delimiter="\t", quotechar=None))[1:]

    texts = []
    labels = []
    for line in lines:
        texts.append(line[0])
        labels.append(int(line[1]))

    return texts, labels

def load_seq_data_from_json(path, MAX_LEN):
    with open(path, "r") as f:
        data = json.load(f)

    texts = []
    labels = []
    for item in data:
        text = item['title'] + " " + item['text']
        text = " ".join(text.split()[:MAX_LEN])
        texts.append(text)
        labels.append(0)

    return texts, labels

def load_and_cache_dataset(DATA_PATH=DATA_PATH, BERT_MODEL=MODEL_PATH, MAX_LEN=MAX_LEN, num_labels=NUM_LABELS):
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    # Load NER data and mappings
    train_ner_texts, train_ner_tags = read_wnut(os.path.join(DATA_PATH, 'Event_detection/train.txt'))
    test_ner_texts, test_ner_tags = read_wnut(os.path.join(DATA_PATH, 'Event_detection/dev.txt'))
    tags = deepcopy(train_ner_tags)
    tags.extend(test_ner_tags)

    unique_tags = list(set(tag for doc in tags for tag in doc))
    tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # The null class should be the max classes + 1 (or 0 to n-1)
    global UNK_ID, NUM_LABELS
    UNK_ID = len(tag2id)
    NUM_LABELS = UNK_ID + 1

    # Tokenize and encode labels for training and testing data
    train_encodings = tokenizer(train_ner_texts, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=MAX_LEN)
    train_ner_labels = encode_tags(train_ner_tags, train_encodings, tag2id, UNK_ID)
    test_encodings = tokenizer(test_ner_texts, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=MAX_LEN)
    test_ner_labels = encode_tags(test_ner_tags, test_encodings, tag2id, UNK_ID)

    # offset_mapping no longer needed
    train_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    return train_encodings, train_ner_labels, test_encodings, test_ner_labels, tag2id, id2tag

train_encodings, train_ner_labels, test_encodings, test_ner_labels, tag2id, id2tag = load_and_cache_dataset()

input_ids = np.array(test_encodings['input_ids'])
attention_mask = np.array(test_encodings['attention_mask'])
token_type_ids = np.array(test_encodings['token_type_ids']) if 'token_type_ids' in train_encodings else None
ner = np.array(test_ner_labels)
print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)
if token_type_ids is not None:
    print("token_type_ids shape:", token_type_ids.shape)

print("ner_labels shape:", ner.shape)
UNK_ID


# Let's check our tokenized dataset, we have the input ids being the word encodings and the entities labels:

# In[ ]:


id2tag


# In[ ]:


input_ids[:10].shape


# In[ ]:


input_ids[:10]


# In[ ]:


ner[:10].shape


# In[ ]:


ner[:10]


# In[ ]:


assert not np.isnan(input_ids).any()


# ## Inside, Outside, Beginning Tags
# 
# The IOB or BIO format is used in tagging chunks of words in NLP.
# 
# - I before a tag indicates that the tag is inside a chunk.
# - O tag indicates that a token belongs to no chunk.
# - B before a tag indicates that the tag is the beginning of a chunk that immediately follows another chunk without O tags between them.
# 
# ## Imbalanced Dataset
# 
# Before you continue, looking at the tag frequencies - we see that our data set is imbalanced towards the `O` and `UNK` tags.
# 
# 

# In[ ]:


ner


# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

unique, counts = np.unique(ner, return_counts=True)

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
weights = compute_class_weight(class_weight="balanced", classes=np.unique(ner), y=ner.flatten())
weights_dict = {i: weights[i] for i in range(len(weights))}

df_tags = pd.DataFrame({'Tag ID': unique, 'Tag': (id2tag[id] if id in id2tag else 'UNK' for id in unique),'Count': counts})
df_tags['Weight'] = df_tags['Tag ID'].map(lambda i: weights_dict.get(i, 0))

df_sorted = df_tags.sort_values(by='Count', ascending=False).reset_index(drop=True)
df_sorted.loc[df_sorted['Tag ID'] == UNK_ID, 'Weight'] = 0. # Unkown tokens should be ignored totally.

df_sorted


# Give the class wieghts tot the model.

# In[ ]:


df_sorted = df_sorted.sort_values(by='Tag ID', ascending=True)
class_weights = df_sorted[['Tag ID', 'Weight']].set_index('Tag ID').to_dict()['Weight']
class_weights


# ### Attention is all you Need
# 
# Vaswani et. al seminal paper *Attention is All You Need*, made self-attention and transformers mainstream.
# 
# Self-attention, calculates the relevance of each word in a sentence to every other word. This is done through queries (Q=XW^Q), keys (K=XW^K), and values (V=XW^V) transformed by a learned weight matrix (W) from the input embeddings (X). The attention score between two words is computed by taking the dot product of their queries and keys, followed by a softmax:
# 
# $ \text{A}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $
# 
# Where:
# - Q represents the queries matrix of current items,
# - K represents the keys matrix of items to compare against in the input sequence,
# - V represents the values matrix, which are the dot product comparisons between Q and K,
# - d_k represents the dimension of the keys and queries,
# - A are the Attention wieghts.
# 
# In addition, the word embeddings will contain contextual information (dot poduct of A and V), represented as position added to the embedding. This plus the attention wieghts, captures dependencies and relationships.

# ## BERT Classifier
# 
# Built on top of a pretrained BERT (Bidirectional Encoder Representations from Transformers).BERT is an industry tested transformer-based model, pre-trained on a large corpus of text to generate contextual embeddings for input sequences.
# 
# We will use a small pre-trained cased base model with 12-layers + 768-hidden, 12-heads , and 110M parameters. This is the base model used in *Zhou et al. (2021)*. Later in the article, we will use larger BERT models that are more resource demanding to fine-tune. Each model has its own preprocesser, as text inputs need to be converted to token IDs.
# 
# The architecture can be summarized in 3 componets:
# 1. Input embeddings, attention masks and ID types for the preTrained BERT model. Bert applies transformer blocks with self-attention (attention captures language structures). The model outputs embedding sequences (last layer from BERT NxH) and a pooled summary derived from the first 'CLS' token(a 1XH vector).
# 1. The sequence outputs (NxH vector) is passed through dense layers and dropouts for the first NER classification, this maps the high-DIM outputs to logits. Padding of unknown tokens helps the model focus on the tasks.
# 1. NER logits are flattened and concatenated with the pooled summaries to form a new feature vector (NxH + H). The vector is passed again through dense and dropout layers to classify the event as one of the 11 identified (O is ignored).

# # Tensorboard

# In[ ]:


get_ipython().system('rm -rf ./logs')
if IN_COLAB:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', '--logdir ./logs')


# # Model Build and Loss

# In[ ]:


from tensorflow.keras import Model
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy, CategoricalFocalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import constant
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from transformers import TFBertModel, BertConfig

with strategy.scope():
    class MaskedWeightedSparseCategoricalCrossentropy(Loss):
        def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
                    name='masked_weighted_sparse_categorical_crossentropy',
                    class_weight=None, null_class=UNK_ID, attention_mask= None):
            super().__init__(reduction=reduction, name=name)
            self.from_logits = from_logits
            self.null_class = tf.cast(null_class, tf.float16)
            self.class_weight = class_weight
            self.attention_mask = attention_mask
            if attention_mask is not None:
                self.attention_mask = tf.cast(self.attention_mask, tf.float16)
                self.attention_mask = tf.expand_dims(self.attention_mask, -1)

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float16)
            # Mask to exclude any class greater than null_class or invalid class entries to 0
            mask = tf.logical_and(tf.greater_equal(y_true, 0.), tf.less(y_true, self.null_class))
            y_true_masked = tf.where(mask, y_true, tf.zeros_like(y_true))
            if tf.executing_eagerly():
              # tf.print(mask)
              # tf.print(y_true)
              tf.debugging.assert_greater(tf.reduce_sum(tf.cast(mask, tf.float32)), 0., message="All data are masked; check class indices and mask conditions.")

            loss_fn = SparseCategoricalCrossentropy(
                from_logits=self.from_logits,
                reduction=tf.keras.losses.Reduction.NONE
            )

            if self.class_weight is None:
                # Sort, wieghts to be in the right order.
                class_weights_list = [self.class_weight[i] for i in sorted(self.class_weight)]
                class_weights_tensor = constant(class_weights_list, dtype=tf.float16)
                # latter we mask to cancel out the default 0 class.
                loss = loss_fn(y_true_masked,
                            y_pred,
                            sample_weight=tf.gather(class_weights_tensor, tf.cast(y_true_masked, tf.int16)))
            else:
                loss = loss_fn(y_true_masked, y_pred)

            loss = tf.cast(loss, tf.float16)
            mask = tf.cast(mask, tf.float16)
            loss *= mask
            # Avoid div by 0.
            sum_mask = tf.reduce_sum(mask)
            loss = (tf.cast(tf.reduce_sum(loss) / sum_mask, dtype=tf.float32)
                        if sum_mask > 0.
                        else tf.constant(0., dtype=tf.float32))
            if tf.executing_eagerly():
              # tf.print(loss)
              tf.debugging.assert_positive(loss, message="Loss masked to zero.")

            return loss

    # https://www.tensorflow.org/text/tutorials/bert_glue
    def create_model(bert_model, config, num_labels=NUM_LABELS, max_len=MAX_LEN, UNK_ID=UNK_ID, class_weight=None, strategy=strategy):
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
        token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')

        bert_output = bert_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        bert_output = tf.cast(bert_output[0], tf.float32 )

        mask = tf.cast(attention_mask, tf.float32 )
        mask = tf.expand_dims(mask, -1)
        masked_output = bert_output * mask

        ner_logits = Dropout(config.hidden_dropout_prob, name='Dropout_ner_1')(masked_output)
        ner_logits = Dense(2048, name='Dense_ner_1', kernel_initializer=GlorotUniform())(ner_logits)
        ner_logits = Dropout(config.hidden_dropout_prob, name='Dropout_ner_2')(ner_logits)
        output = Dense(num_labels, name='Dense_ner_2', dtype='float32')(ner_logits)

        model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
        # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy
        optimizer = AdamW(learning_rate=LEARN_RATE)
        if not is_tpu_strategy(strategy):
          # TPUs already use bfloat16
          optimizer = LossScaleOptimizer(optimizer, dynamic=True)
        model.compile(optimizer=optimizer,
                loss=MaskedWeightedSparseCategoricalCrossentropy(from_logits=True, class_weight=class_weight, attention_mask=attention_mask),
                metrics=['sparse_categorical_accuracy'])

        return model

    input_ids = np.array(train_encodings['input_ids'])
    attention_mask = np.array(train_encodings['attention_mask'])
    token_type_ids = np.array(train_encodings['token_type_ids']) if 'token_type_ids' in train_encodings else None
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        },
        train_ner_labels
    ))
    train_dataset = (train_dataset.shuffle(buffer_size=10000)
                                    .batch(BATCH_SIZE)
                                    .cache()
                                    .prefetch(tf.data.experimental.AUTOTUNE))
    input_ids = np.array(test_encodings['input_ids'])
    attention_mask = np.array(test_encodings['attention_mask'])
    token_type_ids = np.array(test_encodings['token_type_ids']) if 'token_type_ids' in test_encodings else None
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        },
        test_ner_labels
    ))
    test_dataset = (test_dataset.shuffle(buffer_size=10000)
                            .cache()
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                            .batch(BATCH_SIZE))

    config = BertConfig.from_pretrained(MODEL_PATH)
    config.num_labels = NUM_LABELS
    bert_model = TFBertModel.from_pretrained(MODEL_PATH, config=config)


    model = create_model(bert_model, config, num_labels=len(id2tag), class_weight=class_weights)
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    # {'batch_size', 'write_grads', 'embeddings_layer_names', 'embeddings_data'}
    tensorboard_callback = TensorBoard(log_dir='./logs',
                                    histogram_freq=2,
                                    embeddings_freq=2)
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    early_stopping = EarlyStopping(mode='min', patience=5, start_from_epoch=1)
    # tf.debugging.enable_check_numerics() # - Assert if no Infs or NaNs go through.
    # tf.config.run_functions_eagerly(not is_tpu_strategy(strategy)) # - Easy debugging
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        callbacks=[tensorboard_callback, early_stopping],
                        verbose="auto",
                        validation_data=test_dataset)
model.summary()


# In[ ]:


traindata = train_dataset.unbatch().batch(1).take(1)

y1 = model.predict(traindata)
y1.shape


# In[ ]:


predicted_classes = np.argmax(y1, axis=-1)

print("Logits (example):", y1)
print("Predicted classes:", predicted_classes)


# In[ ]:


inputs, labels = next(iter(traindata))
labels


# In[ ]:


labels.shape


# In[ ]:


mask = tf.logical_and(tf.greater_equal(labels, 0), tf.less(labels, UNK_ID))
mask = tf.cast(mask, tf.float32)
losses = tf.keras.losses.sparse_categorical_crossentropy(labels, y1, from_logits=True, ignore_class=UNK_ID)
losses *= tf.expand_dims(mask, -1)

mean_loss = tf.reduce_sum(losses) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)

print(f"Mask: {mask}")
print(f"Masked Losses: {losses.numpy()}")
print(f"Mean Loss: {mean_loss.numpy()}")


# In[ ]:


import matplotlib.pyplot as plt

# Plot Loss vs. Accuracy
plt.figure(figsize=(10, 4))

# Plot Training vs. validation Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Training vs. validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
import numpy as np

predictions = model.predict(test_dataset)
predicted_label_indices = np.argmax(predictions, axis=-1)

true_labels = np.array(test_ner_labels).flatten()
predicted_labels = predicted_label_indices.flatten()

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print(classification_report(true_labels, predicted_labels, labels=range(0, UNK_ID), target_names=list(id2tag.values())))

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels,
    average='weighted')

ner_correct = np.sum(predicted_labels == true_labels)
ner_total = len(true_labels)

print('NER Accuracy: {:.2f}%'.format(100. * ner_correct / ner_total))
print(f'Weighted Precision: {100. * weighted_precision:.2f}%, Recall: {100. * weighted_recall:.2f}%, F1: {100. * weighted_f1:.2f}%')


# The model has an obvious inbalance issue.

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(id2tag.values()),
            yticklabels=list(id2tag.values()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # Conclusion
# 
# TODO

# ## References
# 
# - [Zhou, Zhihan, Liqian Ma, and Han Liu. "Trade the event: Corporate events detection for news-based event-driven trading." arXiv preprint arXiv:2105.12825 (2021).](https://aclanthology.org/2021.findings-acl.186)
# - [Hugging Face Transformers APIs](https://github.com/huggingface/transformers)
# - [Hugging Face Model Repository and Spaces](https://huggingface.co/models)
# - [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official/vision#table-of-contents)
# - [Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).](https://arxiv.org/abs/1810.04805)
# - [Google Pre-trained BERT Models.](https://github.com/google-research/bert?tab=readme-ov-file)
# - [Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin, "Attention is All You Need". NIPS (2017)](https://research.google/pubs/attention-is-all-you-need/)

# ## Github
# 
# Article and code available on [Github](https://github.com/adamd1985/news-based-event-driven_algotrading)
# 
# Kaggle notebook available [here]()
# 
# Google Collab available [here]()
# 
# ## Media
# 
# All media used (in the form of code or images) are either solely owned by me, acquired through licensing, or part of the Public Domain and granted use through Creative Commons License.
# 
# ## CC Licensing and Use
# 
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
