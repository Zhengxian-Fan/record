from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os, sys
from absl import app
from absl import flags
import modeling
import tensorflow as tf
import tokenization
import semantics

flags = tf.flags

FLAGS = flags.FLAGS

tf.flags.DEFINE_bool(
  "use_gpu", True,
  "")

tf.flags.DEFINE_string(
  "data_dir", None,
  "")

tf.flags.DEFINE_string(
  "config_dir", None,
  "")
tf.flags.DEFINE_string(
  "ckpt", None,
  "")
tf.flags.DEFINE_string(
  "output_path", None,
  "")

data_path = FLAGS.data_dir + '/subtaskA_trial_data.csv'
ans_path = FLAGS.data_dir + '/subtaskA_answers.csv'
output_path = FLAGS.output_path


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
    tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_mlm_output(input_tensor, albert_config, mlm_positions, output_weights, label_ids, label_weights):
  """From run_pretraining.py."""
  input_tensor = gather_indexes(input_tensor, mlm_positions)
  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
        input_tensor,
        units=albert_config.embedding_size,
        activation=modeling.get_activation(albert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
          albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
      "output_bias",
      shape=[albert_config.vocab_size],
      initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [1, -1])
    one_hot_labels = tf.one_hot(
      label_ids, depth=albert_config.vocab_size, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    masked_lm_log_probs = tf.reshape(log_probs,
                                     [-1, log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
      masked_lm_log_probs, axis=-1, output_type=tf.int32)
    # return masked_lm_predictions
    return loss,per_example_loss


def build_model():
  """Module function."""

  input_ids = tf.placeholder(tf.int32, [None, None], "input_ids")
  # input_mask = tf.placeholder(tf.int32, [None, None], "input_mask")
  # segment_ids = tf.placeholder(tf.int32, [None, None], "segment_ids")
  mlm_positions = tf.placeholder(tf.int32, [None, None], "mlm_positions")
  mlm_ids = tf.placeholder(tf.int32, [None, None], "mlm_ids")
  mlm_weights = tf.placeholder(tf.float32, [None, None], "mlm_weights")
  albert_config_path = os.path.join(FLAGS.config_dir, "albert_config.json")
  albert_config = modeling.AlbertConfig.from_json_file(albert_config_path)
  model = modeling.AlbertModel(
    config=albert_config,
    is_training=False,
    input_ids=input_ids,
    # input_mask=input_mask,
    # token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

  loss = get_mlm_output(model.get_sequence_output(), albert_config,
                        mlm_positions, model.get_embedding_table(), mlm_ids, mlm_weights)
  return loss
  # return mlm_ids, input_ids


def get_elements_to_mask(s0, s1):
  """
    Args:
    s0,s1 : two input sentences
    Returns:
    disjoint0,disjoint1: words to mask in s0,s1
  # """
  disjoint0 = [i for i in s0 if not i in s1]
  disjoint1 = [i for i in s1 if not i in s0]
  if not disjoint0:
    # is not good, is good
    if len(s0) != len(s1):
      disjoint0, disjoint1 = s0, s1
    # eg. lamp on desk, desk on lamp
    else:
      disjoint0 = [i for (i, j) in zip(s0, s1) if i != j]
  if not disjoint1:
    if len(s0) != len(s1):
      disjoint0, disjoint1 = s0, s1
    else:
      disjoint1 = [i for (i, j) in zip(s1, s0) if i != j]
  # disjoint0 = list(s0)
  # disjoint1 = list(s1)
  return disjoint0, disjoint1


# # with tf.Session() as sess:
# #   flat_offsets = tf.reshape(
# #       tf.range(0, 4, dtype=tf.int32) * 3, [-1, 1])
# #   positions = tf.constant([[0,1],[0,1],[0,1],[0,1]])
# #   flat_positions = tf.reshape(positions + flat_offsets, [-1])
# #   sequence_tensor = tf.constant([[22,23,24],[24,25,26],[26,27,28],[28,29,30]])
# #   flat_sequence_tensor = tf.reshape(sequence_tensor,
# #
# # 20.60, 20.29
def get_feed_dict(sent, l):
  """
  Args:
  sent,l : sentence and list of words to mask
  """
  input_ids = [[] for _ in l]
  # segment_ids = tf.placeholder(tf.int32, [None, None], "segment_ids")
  mlm_positions = [[] for _ in l]
  mlm_ids = [[] for _ in l]
  mlm_weights = [[] for _ in l]
  index = 1
  longest = 0  # longest whole word
  d = {}
  for ele in sent:
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele))
    for j in range(len(l)):
      if l[j] == ele:
        # mask
        input_ids[j] += [4] * len(tokens)
        mlm_positions[j] += [_ for _ in range(index, index + len(tokens))]
        mlm_weights[j] += [1] * len(tokens)
        mlm_ids[j] += tokens
        l[j] = ''
        longest = max(longest, len(tokens))
      else:
        input_ids[j] += tokens
    index += len(tokens)

  """Do zero padding """
  for i in range(len(l)):
    # '[CLS]' '[SEP]'
    input_ids[i] = [2] + input_ids[i] + [3]
    while len(mlm_positions[i]) < longest:
      mlm_positions[i].append(0)
      mlm_ids[i].append(0)
      mlm_weights[i].append(0)
  d["input_ids:0"] = input_ids
  d["mlm_positions:0"] = mlm_positions
  d["mlm_ids:0"] = mlm_ids
  d["mlm_weights:0"] = mlm_weights
  return d


tokenizer = tokenization.FullTokenizer(
  vocab_file=FLAGS.config_dir + '/30k-clean.vocab',
  spm_model_file=FLAGS.config_dir + '/30k-clean.model',
  do_lower_case=True)


if FLAGS.use_gpu:
    device = "/gpu:0"
else:
    device = "/cpu:0"


with tf.device(device):
  loss = build_model()




sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, FLAGS.ckpt)

with open(data_path) as f, open(ans_path) as g, open(output_path, "w") as o:
  data = csv.reader(f)
  ans = csv.reader(g)
  writer = csv.writer(o)
  # skip first row
  next(data)
  numerator = 0
  denominator = 0
  net_gain = 0
  for line in data:
    denominator += 1
    id, sent0, sent1 = line
    sent0, sent1 = sent0.split(' '), sent1.split(' ')
#     sent0, sent1 = semantics.get_phrases(sent0.lower()), semantics.get_phrases(sent1.lower())
    l0, l1 = list(sent0), list(sent1)
    feed_dict0 = get_feed_dict(sent0, l0)
    feed_dict1 = get_feed_dict(sent1, l1)
    loss0,per_example_loss0 = sess.run(loss, feed_dict=feed_dict0)
    loss1,per_example_loss1 = sess.run(loss, feed_dict=feed_dict1)
    # print(id,sent0,sent1)
    # print("input_ids",feed_dict1["input_ids:0"])
    # print(loss0)
    # print("predictions", p1)
    # print("sent",sent0)
    # # print("prediction",loss0)
    # # print(p1,p2)
    # p1 = list(loss0)
    # p1 = [int(i) for i in p1]
    # # print(tokenizer.convert_ids_to_tokens(feed_dict0["input_ids:0"][0]))
    # print("predict",tokenizer.convert_ids_to_tokens(p1))
    # print(loss0[1].shape)
    print(sent0,sent1)
    print(per_example_loss0,per_example_loss1)
    loss_diff = loss0 - loss1
    res = []
    if loss0 < loss1:
      res = [id, '1']
    else:
      res = [id, '0']
    if next(ans) == res:
      net_gain += abs(loss_diff)
      numerator += 1
    else: net_gain -= abs(loss_diff)
    print(numerator / denominator)
    print(net_gain / denominator)
    writer.writerow([id, ' '.join(sent0), ' '.join(sent1)] + [str(numerator / denominator)] + [str(net_gain / denominator)])

# filtered_output_path = 'subtaskA/filtered_output.csv'
# with open(output_path) as input, open(filtered_output_path, 'w') as output:
#   data = csv.reader(input)
#   writer = csv.writer(output)
#   prev = 0
#
#   for line in data:
#     id, sent0, sent1, acc = line
#     acc = float(acc)
#     sent0, sent1 = sent0.lower(), sent1.lower()
#     if acc<prev:
#       is_equals = 0
#       if len(sent0.split(' ')) != len(sent1.split(' ')):
#         writer.writerow([id, sent0, sent1]+[str(is_equals)])
#     prev = acc
#
# #


# with open('subtaskA/subtaskA_trial_data.csv') as i:
#   data = csv.reader(i)
#   for line in data:
#     id, sent0, sent1 = line
#     print(get_phrases(str(sent0)))

# sent = "usa is located in the northeastern part of new york"
# ranges = semantics.get_phrases(sent)
# print(ranges)
# sent_split = [sent]
#
# print(sent.split(phrases[0]))
# print(sent.split(phrases[1]))
# print(sent.split(phrases[2]))
# for i in range(len(sent)):
#   for




