#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.spatial import cKDTree as KDTree

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)

  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself. Avg 5 nearest samples as high dimension estimate is unstable
  r = xtree.query(x, k=5, eps=.01, p=2)[0][:,1:].mean(axis =1)
  # select 2 nearest and average
  s = ytree.query(x, k=5, eps=.01, p=2)[0].mean(axis = 1)

  # stop numerical instability when nearing close to 0
  r += 1e-4
  s += 1e-4

  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))