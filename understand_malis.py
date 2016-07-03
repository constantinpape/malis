import vigra
import numpy as np


# FIXME malis 2d seems to be broken due to 3d hardcoded in nodelist_like
def malis_2d(aff_path):
    # import all the malis functionality we need
    from malis import mknhood2d, affgraph_to_edgelist, malis_loss_weights

    # load affinity graph ( only the zeroth slice and the x and y weights )
    aff = vigra.readHDF5(aff_path, "data")[:,:,0,:2].transpose( (2,1,0,3) )

    # make the 2d neighborhood
    nhood = mknhood2d()

    # get the node connectors and weights from the affinitygraph
    connectors1, connectors2, edge_weights = affgraph_to_edgelist(aff, nhood)

    print connectors1.shape, connectors2.shape, edge_weights.shape


def malis_3d(aff_path, gt_path):
    # import all the malis functionality we need
    from malis import mknhood3d, affgraph_to_edgelist, malis_loss_weights

    # load affinity graph, needs to be transposed to match malis convention
    aff = vigra.readHDF5(aff_path, "data")[:256,:256,:10].transpose( (3,2,1,0) )
    gt  = vigra.readHDF5(gt_path, "volumes/labels/neuron_ids")[:256,:256,:10].transpose( (2,1,0) ).astype(np.int32)

    # need to ravel the gt
    gt = gt.ravel()

    print "Loaded affinity graph"

    # make the 3d neighborhood
    nhood = mknhood3d()

    # get the node connectors and weights from the affinitygraph
    # extracts the node connectors and weights from the affinity graph for the given neighborhood
    # TODO why do we have 2 node connectors ? <-> bidirectional ?!
    connectors1, connectors2, edge_weights = affgraph_to_edgelist(aff, nhood)

    print "nodes and weights from affinity graph"
    print "connectors shape:", connectors1.shape
    print "edge weights shape:", edge_weights.shape
    print "This should have equal lengths..."

    # get the malis los
    # parameters:
    # gt : groundtruth (raveled)
    # connectors1 : uvIds(0)
    # connectors2 : uvIds(1) -> the two nodeconnectors tell the nodes that are connected by the edge
    # edge_weights : raveled edge weights from the affinity graph
    # pos : magic pos parameter TODO what is this ?
    pos = 0
    malis_loss_correct_merges = malis_loss_weights(gt, connectors1, connectors2,
            edge_weights, pos)
    pos = 1
    malis_loss_false_merges = malis_loss_weights(gt, connectors1, connectors2,
            edge_weights, pos)

    print "malis loss"
    print malis_loss_correct_merges.shape
    print malis_loss_correct_merges.dtype


if __name__ == '__main__':

    # path to the affinity graph
    # expected shape: (x,y,z,c) - where c is the channel axis for the x,y and z weights
    aff_path = "/home/consti/cremi_wsgraph/cremi_sampleA_affinities_googleV4.h5"
    gt_path  = "/home/consti/cremi_wsgraph/sample_A_20160501.hdf"

    malis_3d(aff_path, gt_path)
