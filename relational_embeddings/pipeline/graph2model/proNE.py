# encoding=utf8
import time

import numpy as np
from omegaconf import OmegaConf
from scipy import linalg
import scipy.sparse as sp
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

from relational_embeddings.lib.graph_utils import read_graph
from relational_embeddings.lib.utils import make_symlink

def proNE_graph2model(indir, outdir, cfg):
    infile = indir / "edgelist" 
    outfile_sparse = outdir / "model_sparse"
    outfile_spectral = outdir / "model_spectral"

    nx_G = read_graph(infile, cfg.weighted)

    model = ProNE(nx_G, cfg.dimensions)
    features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
    embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, cfg.step, cfg.mu, cfg.theta)

    save_embedding(outfile_sparse, features_matrix)
    save_embedding(outfile_spectral, embeddings_matrix)

    make_symlink(indir / "node_dict", outdir / "node_dict")

    model_cnf = OmegaConf.create({"model_type": "ProNE"})
    node_types = OmegaConf.load(indir / "node_types")
    model_cnf = OmegaConf.merge(model_cnf, node_types)
    OmegaConf.save(model_cnf, outdir / "model_cnf")


class ProNE():
    def __init__(self, G, dimension):
        self.dimension = dimension
        self.G = G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        matrix0 = sp.lil_matrix((self.node_number, self.node_number))

        for e in self.G.edges():
            if e[0] != e[1]:
                matrix0[e[0], e[1]] = self.G[e[0]][e[1]]["weight"]
                matrix0[e[1], e[0]] = self.G[e[1]][e[0]]["weight"]
        self.matrix0 = sp.csr_matrix(matrix0)

    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = sp.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(
            smat, n_components=self.dimension, n_iter=3, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False,
                              check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U

    def pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = sp.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        print('Chebyshev Series -----------------')
        t1 = time.time()

        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA

        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #		 Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
            print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        emb = self.get_embedding_dense(mm, self.dimension)
        return emb


def save_embedding(emb_file, features):
    # save node embedding into emb_file with word2vec format
    f_emb = open(emb_file, 'w')
    f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
    for i in range(len(features)):
        s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
        f_emb.write(s + "\n")
    f_emb.close()
