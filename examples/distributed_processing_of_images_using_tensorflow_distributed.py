# This script is from https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html

# Main functions are 
# - grpc is used
# based on the instructions here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md

#from pyspark import SparkContext, SparkConf
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import os.path
import re
import sys
import tarfile
from subprocess import Popen, PIPE, STDOUT

# To run 'tar' system command in 'read_file_index()' function
#
def run(cmd):
  p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
  return p.stdout.read()

model_dir = '/tmp/imagenet'
image_file = ""
num_top_predictions = 5

# This archive file contains:
#
# -rw-rw-r-- 1 cc cc 95673916 Dec  5 02:14 classify_image_graph_def.pb
# -rw-rw-r-- 1 cc cc     2683 Dec  1 04:28 cropped_panda.jpg
# -rw-rw-r-- 1 cc cc    64986 Nov 18 22:39 imagenet_2012_challenge_label_map_proto.pbtxt
# -rw-rw-r-- 1 cc cc   741401 Nov 18 22:39 imagenet_synset_to_human_label_map.txt
# -rw-rw-r-- 1 cc cc 88931400 Apr 13 05:33 inception-2015-12-05.tgz
# -rw-rw-r-- 1 cc cc    11416 Dec  4 23:07 LICENSE
#
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

IMAGES_INDEX_URL = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
# The number of images to process.
# unclear what this means
image_batch_size = 3
# more than 1000 images to parse...
#
max_content = 1000L

def read_file_index():
  from six.moves import urllib
  content = urllib.request.urlopen(IMAGES_INDEX_URL)
  data = content.read(max_content)
  tmpfile = "/tmp/imagenet.tgz"
  with open(tmpfile, 'wb') as f:
    f.write(data)
  run("tar -xOzf %s > /tmp/imagenet.txt" % tmpfile)
  with open("/tmp/imagenet.txt", 'r') as f:
    lines = [l.split() for l in f]
    input_data = [tuple(elts) for elts in lines if len(elts) == 2]

    # n00004475_6590	http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg
    # n00004475_15899	http://farm4.static.flickr.com/3276/2875184020_9944005d0d.jpg

    return [input_data[i:i+image_batch_size] for i in range(0,len(input_data), image_batch_size)]

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    # n15089472       fat-soluble vitamin
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    # entry {
    #   target_class: 878
    #   target_class_string: "n13044778"
    # }
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


# Is this not called from anywhere?
#
def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    # From:  'imagenet_2012_challenge_label_map_proto.pbtxt'
    # entry {
    #  target_class: 384
    #  target_class_string: "n01514859" 
    # }
    # To:  imagenet_synset_to_human_label_map.txt
    # n01514859       hen
    # 384 -> n01514859 -> hen
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  from six.moves import urllib
  dest_directory = model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    filepath2, _ = urllib.request.urlretrieve(DATA_URL, filepath)
    print("filepath2", filepath2)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  else:
      print('Data already downloaded:', filepath, os.stat(filepath))

###############################################################################
# Start to download images from image-net.org
###############################################################################
maybe_download_and_extract()

###############################################################################
# Extract archive files and read contents
###############################################################################
batched_data = read_file_index()

print "There are %d batches" % len(batched_data)

# model_dir = /tmp/imagenet
# 
# 1000 entries in this file
label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')

# Sample records:
# n00440643	Fosbury flop
# n00440747	skiing
#
uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')

def load_lookup():
  """Loads a human readable English name for each softmax node.

  Args:
    label_lookup_path: string UID to integer node ID.
    uid_lookup_path: string UID to human-readable string.

  Returns:
    dict from integer node ID to human-readable string.
  """
  if not gfile.Exists(uid_lookup_path):
    tf.logging.fatal('File does not exist %s', uid_lookup_path)
  if not gfile.Exists(label_lookup_path):
    tf.logging.fatal('File does not exist %s', label_lookup_path)

  # Loads mapping from string UID to human-readable string
  proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
  uid_to_human = {}
  p = re.compile(r'[n\d]*[ \S,]*')
  for line in proto_as_ascii_lines:
    parsed_items = p.findall(line)
    uid = parsed_items[0]
    human_string = parsed_items[2]
    uid_to_human[uid] = human_string

  # Loads mapping from string UID to integer node ID.
  node_id_to_uid = {}
  proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
  for line in proto_as_ascii:
    if line.startswith('  target_class:'):
      target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
      target_class_string = line.split(': ')[1]
      node_id_to_uid[target_class] = target_class_string[1:-2]

  # Loads the final mapping of integer node ID to human-readable string
  node_id_to_name = {}
  for key, val in node_id_to_uid.items():
    if val not in uid_to_human:
      tf.logging.fatal('Failed to locate: %s', val)
    name = uid_to_human[val]
    node_id_to_name[key] = name

  return node_id_to_name

###############################################################################

node_lookup = load_lookup()


master="mesos://192.168.0.70:5050"
appName="distributed-processing-of-images-using-tensorflow"

#conf = SparkConf().setAppName(appName).setMaster(master)
#conf.set("spark.mesos.coarse", "false")
#sc = SparkContext(conf=conf)

#node_lookup_bc = sc.broadcast(node_lookup)

model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')
with gfile.FastGFile(model_path, 'rb') as f:
  model_data = f.read()

#model_data_bc = sc.broadcast(model_data)

###############################################################################


def run_image(sess, img_id, img_url, node_lookup):
  from six.moves import urllib
  from urllib2 import HTTPError
  try:
    image_data = urllib.request.urlopen(img_url, timeout=1.0).read()
  except HTTPError:
    return (img_id, img_url, None)
  except:
    return (img_id, img_url, None)
  scores = []
  softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
  try:
	  predictions = sess.run(softmax_tensor,
                         {'DecodeJpeg/contents:0': image_data})
  except tf.errors.InvalidArgumentError:
    return (img_id, img_url, None)
  predictions = np.squeeze(predictions)
  top_k = predictions.argsort()[-num_top_predictions:][::-1]
  scores = []
  for node_id in top_k:
    if node_id not in node_lookup:
      human_string = ''
    else:
      human_string = node_lookup[node_id]
    score = predictions[node_id]
    scores.append((human_string, score))
  return (img_id, img_url, scores)

def apply_batch(batch):
  with tf.Graph().as_default() as g:
    graph_def = tf.GraphDef()
    #graph_def.ParseFromString(model_data_bc.value)
    graph_def.ParseFromString(model_data)
    tf.import_graph_def(graph_def, name='')
    #with tf.Session() as sess:
    with tf.Session("grpc://localhost:32773") as sess:
      labelled = [run_image(sess, img_id, img_url, node_lookup) for (img_id, img_url) in batch]
      return [tup for tup in labelled if tup[2] is not None]

#urls = sc.parallelize(batched_data)
#labelled_images = urls.flatMap(apply_batch)
#print batched_data
labelled_images = []
for batch in batched_data:
	labelled_images.append(apply_batch(batch))
#local_labelled_images = labelled_images.collect()
#print(local_labelled_images)
print labelled_images

