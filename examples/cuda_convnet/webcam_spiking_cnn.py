"""
Classifier for the ImageNet ILSVRC-2012 dataset.
"""
import nengo
import numpy as np

from nengo_extras.camera import Camera
from nengo_extras.data import load_ilsvrc2012_metadata, spasafe_names
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle
from nengo_extras.gui import image_display_function

data_mean, label_names = load_ilsvrc2012_metadata()
data_mean = data_mean[:, 16:-16, 16:-16]
image_shape = data_mean.shape

# retrieve from https://figshare.com/s/f343c68df647e675af28
cc_model = load_model_pickle('ilsvrc2012-lif-48.pkl')

# --- Run model in Nengo
with nengo.Network() as model:
    u = nengo.Node(Camera(
        device='/dev/video0',
        height=image_shape[1], width=image_shape[2], offset=-data_mean))

    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.001))
    nengo.Connection(u, ccnet.input, synapse=None)

    # --- image display
    display_f = image_display_function(image_shape, scale=1., offset=data_mean)
    display_node = nengo.Node(display_f, size_in=u.size_out)
    nengo.Connection(u, display_node, synapse=None)

    # --- output spa display
    vocab_names = spasafe_names(label_names)
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output = nengo.spa.State(
            len(vocab_names), subdimensions=10, vocab=vocab)
    nengo.Connection(ccnet.output, output.input)
