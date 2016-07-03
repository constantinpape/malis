#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	    const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const int& ind1, const int& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};

/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */

// Trying to understand this...
// Parameters: nVert: Number of nodes (= number of pixels)
// seg : pointer to the groundtruth segmentation
// nEdge : number of edges (= number of pixels * neighborhood)
// node1: pointer to uvids[0]
// node2: pointer to uvids[1] 
// edgeWeight: pointer to the edge weights TODO what shape is expected here?
// pos: magic parameter TODO wtf is this ?, it is used as a bool!
// nPairEdge: pointer to returnval
void malis_loss_weights_cpp(const int nVert, const int* seg,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               const int pos,
               int* nPairPerEdge){


    /* Disjoint sets and sparse overlap vectors */
    // for every node, store with how many nodes of a given gt segment it was merged
    vector<map<int,int> > overlap(nVert);
    
    // this is boost's UDF
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i){
        dsets.make_set(i);
        if (0!=seg[i]) {
            // initialize with map from gt_id -> 1
            overlap[i].insert(pair<int,int>(seg[i],1));
        }
    }

    /* Sort all the edges in increasing order of weight */
    std::vector< int > pqueue( nEdge );
    // j counts the number of valud edges (which connect nodes with a valid id)
    int j = 0;
    for ( int i = 0; i < nEdge; i++ ){
        if ((node1[i]>=0) && (node1[i]<nVert) && (node2[i]>=0) && (node2[i]<nVert))
	        pqueue[ j++ ] = i;
    }
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    // sort the edge queue according to the edge weights
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edgeWeight ) );


    /* Start MST */
    int e;
    int set1, set2;
    int nPair = 0;
    map<int,int>::iterator it1, it2;

    /* Start Kruskal's */

    // iterate over the edges
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
        // get current edge
        e = pqueue[i];

        // get representatives of nodes connected by the edge
        set1 = dsets.find_set(node1[e]);
        set2 = dsets.find_set(node2[e]);

        // only do stuff, if the representatives are different (note that in the beginning all are different!)
        if (set1!=set2){
            
            // merge the representatives
            dsets.link(set1, set2);

            /* compute the number of pairs merged by this MST edge */

            // iterate over the overlaps (TODO what are these) of the two nodes representatives
            for (it1 = overlap[set1].begin();
                    it1 != overlap[set1].end(); ++it1) {
                for (it2 = overlap[set2].begin();
                        it2 != overlap[set2].end(); ++it2) {

                    nPair = it1->second * it2->second;

                    // here we have our pos int abused as bool
                    // this counts the number of correct merge (it->first = gt id)
                    if (pos && (it1->first == it2->first)) {
                        nPairPerEdge[e] += nPair;
                    } 
                    // this counts the number of false merges
                    else if ((!pos) && (it1->first != it2->first)) {
                        nPairPerEdge[e] += nPair;
                    }
                }
            }

            /* move the pixel bags of the non-representative to the representative */

            // swap references if the merged set has representative of set2
            if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
                swap(set1,set2);

            // loop over the overlaps of set2 and move them to set1
            it2 = overlap[set2].begin();
            while (it2 != overlap[set2].end()) {
                // look if this gt segment has already overlap with set2
                it1 = overlap[set1].find(it2->first);
                if (it1 == overlap[set1].end()) {
                    // no: insert overlap
                    overlap[set1].insert(pair<int,int>(it2->first,it2->second));
                } else {
                    // yes: add overlap
                    it1->second += it2->second;
                }
                // remove overlap of set2
                overlap[set2].erase(it2++);
            }
        } // end link

    } // end while
}


void connected_components_cpp(const int nVert,
               const int nEdge, const int* node1, const int* node2, const int* edgeWeight,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i)
        dsets.make_set(i);

    /* union */
    for (int i = 0; i < nEdge; ++i )
         // check bounds to make sure the nodes are valid
        if ((edgeWeight[i]!=0) && (node1[i]>=0) && (node1[i]<nVert) && (node2[i]>=0) && (node2[i]<nVert))
            dsets.union_set(node1[i],node2[i]);

    /* find */
    for (int i = 0; i < nVert; ++i)
        seg[i] = dsets.find_set(i);
}


void marker_watershed_cpp(const int nVert, const int* marker,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i)
        dsets.make_set(i);

    /* initialize output array and find representatives of each class */
    std::map<int,int> components;
    for (int i=0; i<nVert; ++i){
        seg[i] = marker[i];
        if (seg[i] > 0)
            components[seg[i]] = i;
    }

    // merge vertices labeled with the same marker
    for (int i=0; i<nVert; ++i)
        if (seg[i] > 0)
            dsets.union_set(components[seg[i]],i);

    /* Sort all the edges in decreasing order of weight */
    std::vector<int> pqueue( nEdge );
    int j = 0;
    for (int i = 0; i < nEdge; ++i)
        if ((edgeWeight[i]!=0) &&
            (node1[i]>=0) && (node1[i]<nVert) &&
            (node2[i]>=0) && (node2[i]<nVert) &&
            (marker[node1[i]]>=0) && (marker[node2[i]]>=0))
                pqueue[ j++ ] = i;
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edgeWeight ) );

    /* Start MST */
	int e;
    int set1, set2, label_of_set1, label_of_set2;
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
		e = pqueue[i];
        set1=dsets.find_set(node1[e]);
        set2=dsets.find_set(node2[e]);
        label_of_set1 = seg[set1];
        label_of_set2 = seg[set2];

        if ((set1!=set2) &&
            ( ((label_of_set1==0) && (marker[set1]==0)) ||
             ((label_of_set2==0) && (marker[set1]==0))) ){

            dsets.link(set1, set2);
            // either label_of_set1 is 0 or label_of_set2 is 0.
            seg[dsets.find_set(set1)] = std::max(label_of_set1,label_of_set2);
            
        }

    }

    // write out the final coloring
    for (int i=0; i<nVert; i++)
        seg[i] = seg[dsets.find_set(i)];

}
