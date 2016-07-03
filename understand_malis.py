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


def malis_3d(aff, gt):

    assert aff.ndim == 4, "affinitygraph needs to be 4 dimensional"
    assert aff.shape[0] == 3, "affinitygraph channel 0 needs z,y and x affinities"
    assert aff.shape[1:] == gt.shape, "Spatial shapes of affinity graph and gt need to be the same"

    # import all the malis functionality we need
    from malis import mknhood3d, affgraph_to_edgelist, malis_loss_weights

    # need to ravel the gt
    gt = gt.ravel()

    # make the 3d neighborhood
    nhood = mknhood3d()

    # get the node connectors and weights from the affinitygraph
    # extracts the uvIds and weights from the affinity graph for the given neighborhood
    uvs1, uvs2, edge_weights = affgraph_to_edgelist(aff, nhood)

    print "nodes and weights from affinity graph"
    print "connectors shape:", uvs1.shape
    print "edge weights shape:", edge_weights.shape

    # malis loss:
    # calculates number of correct / false merges caused by edge

    # parameters:
    # gt : groundtruth (raveled)
    # connectors1 : uvIds(0)
    # connectors2 : uvIds(1) -> the two nodeconnectors tell the nodes that are connected by the edge
    # edge_weights : raveled edge weights from the affinity graph
    # pos : pseudo bool, that determnies whether we count correct (pos = 1) or false (pos = 0) merges per edge
    pos = 0
    malis_loss_false_merges = malis_loss_weights(gt, uvs1, uvs22,
            edge_weights, pos)

    print "Calculated false merges per edge"

    pos = 1
    malis_loss_correct_merges = malis_loss_weights(gt, uvs1, uvs2,
            edge_weights, pos)

    print "Calculated correct merges per edge"

    return malis_loss_false_merges, malis_loss_correct_merges



if __name__ == '__main__':

    # path to the affinity graph
    # expected shape: (x,y,z,c) - where c is the channel axis for the x,y and z weights
    aff_path = "/home/consti/cremi_wsgraph/cremi_sampleA_affinities_googleV4.h5"
    gt_path  = "/home/consti/cremi_wsgraph/sample_A_20160501.hdf"

    # need to transpose the data to axisorder expected by malis:
    # (e,z,y,x) for affinity graph (e = 3: z,y,x affinities)
    # (z,y,x) for groundtruth
    aff = vigra.readHDF5(aff_path, "data")[:256,:256,:10].transpose( (3,2,1,0) )
    gt  = vigra.readHDF5(gt_path, "volumes/labels/neuron_ids")[:256,:256,:10].transpose( (2,1,0) ).astype(np.int32)

    n_false_merges, n_correct_merges = malis_3d(aff, gt)
