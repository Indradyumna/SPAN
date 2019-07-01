import numpy as np
import random
from itertools import combinations
from abc import ABC, abstractmethod

class DataGenerator(ABC):
    """
    Abstract base class for Data Generators
    """
    @abstractmethod
    def getData(self, size):
        """
        Dataset generating method. Returns numpy arrays x and y corresponding to sets and set-labels
        Params:
            size: Size of dataset to be generated
        """
        pass

class MaxKAryDistanceDG(DataGenerator):
    """
    Data Generator for Max k-ary distance Task
    """
    def __init__(self, k, n, dim, maxN, std):
        """
        Params:
            k: k-ary distance parameter (e.g. 2, 3)
            n: Number of elements in a set
            dim: Dimensionality of each element in set
            maxN: Upper bound on the size of each number in set
            std: Standard deviation around multi-variate Gaussian from which data is generated
        """
        self.k = k
        self.n = n
        self.dim = dim
        self.maxN = maxN
        self.cov = std * np.eye(dim)
        
    def getMaxKAryDistance(self, x):
        """
        Computes the Max k-ary distance within the set x
        Params:
            x: Input set of shape (n, dim)
        """
        assert x.ndim == 2
        assert x.shape[0] == self.n
        assert x.shape[1] == self.dim
        ret = -1
        indexList = list(combinations(np.arange(self.n), self.k))
        for indices in indexList:
            sm = 0
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sm += np.linalg.norm(x[indices[i], :] - x[indices[j], :])
            ret = max(ret, sm)
        return ret
    
    def getData(self, size):
        """
        Dataset generating method. Returns numpy arrays x and y corresponding to sets and set-labels
        Params:
            size: Size of dataset to be generated
        """
        retX = []
        retY = []
        clusterSizes = [int(self.n / self.k) for _ in range(self.k)]
        clusterSizes[-1] += (self.n - np.sum(clusterSizes))
        for _ in range(size):
            x = None
            for clusterSize in clusterSizes:
                mu = np.random.randint(1, np.random.randint(2, self.maxN), (self.dim)).tolist()
                _x = np.random.multivariate_normal(mu, self.cov, clusterSize)
                if x is None:
                    x = _x
                else:
                    x = np.concatenate([x, _x], axis=0)
            np.random.shuffle(x)
            retX.append(x)
            retY.append(self.getMaxKAryDistance(x))
        retX = np.reshape(retX, (-1, self.n, self.dim))
        retY = np.reshape(np.array(retY), [-1, 1])
        return retX, retY
    
class RthPercentileEstimationDG(DataGenerator):
    """
    Data Generator for r^th Percentile Estimation Task
    """
    def __init__(self, r, n, dim, maxN):
        """
        Params:
            r: r^th Percentile to be estimated (e.g. 50, 70)
            n: Number of elements in a set
            dim: Dimensionality of each element in set
            maxN: Upper bound on the size of each number in set
        """
        self.r = r
        self.n = n
        self.dim = dim
        assert self.dim == 1
        self.maxN = maxN
    
    def getData(self, size):
        """
        Dataset generating method. Returns numpy arrays x and y corresponding to sets and set-labels
        Params:
            size: Size of dataset to be generated
        """
        x = []
        _range = int(self.maxN / 2)
        for i in range(size):
            tp = np.random.randint(1, np.random.randint(self.maxN - _range, self.maxN + _range), (self.n, self.dim)).tolist()
            x.append(tp)
        x = np.array(x)
        x = np.reshape(x, [-1, self.n])
        y = []
        idx = int(self.n * (self.r / 100))
        for i in range(len(x)):
            sortedRow = sorted(x[i, :].tolist())
            label = sortedRow[idx]
            y.append(label)
            
        x = np.reshape(x, [-1, self.n, self.dim])
        y = np.reshape(y, [-1, 1])
        return x, y
        
class MultiSourceMaxFlowDG(DataGenerator):
    """
    Data Generator for Multiple Source Maximum Flow Task
    """
    def __init__(self, filePath, n, sinkInFile=False):
        """
        Params:
            filePath: Path to the file with the input graph in standard format
                      The first line contains V, E - number of nodes, number of edges
                      The second line optionally contains the sink
                      Third line onwards, we have E (u, v, c) triplets, which specify an edge
                      between node u and node v with capacity c
            n: Number of elements in a set
            sinkInFile: Boolean flag to indicate if file contains sink node
        """
        self.sinkInFile = sinkInFile
        self.sink = None
        self.formGraph(filePath)
        assert self.numComponents(self.u_g) == 1
        self.n = n
                
    def formGraph(self, filePath):
        """
        Reads the file containing the graph and populates the adjacency list
        Params:
            filePath: Path to the file containing the graph input
        """
        f = open(filePath, "r")
        self.numNodes, self.numEdges = map(int, f.readline().split())
        if self.sinkInFile:
            self.sink = map(int, f.readline().split())
        self.u_g = []
        self.g = []
        for _ in range(self.numNodes):
            self.u_g.append([])
            self.g.append([]) 
        while(True):
            line = f.readline()
            if not line:
                break
            u, v, c = map(int, line.split())
            self.g[u].append((v, c))
            self.u_g[u].append((v, c))
            self.u_g[v].append((u, c))
            
    def dfs(self, graph, cur, vis):
        """
        Performs depth-first search on the graph and returns the connected component containing cur
        Params:
            graph: Adjacency list of the graph
            cur: Current node in depth-first search
            vis: Visited array
        """
        vis[cur] = 1
        component = [cur]
        for node, cost in graph[cur]:
            if vis[node] == 0:
                child_component = self.dfs(graph, node, vis)
                component.extend(child_component)
        return component
    
    def numComponents(self, graph):
        """
        Calculates number of connected components in graph, uses dfs() as a helper
        Params:
            graph: Adjacency list of the graph
        """
        _numNodes = len(graph)
        vis = [0 for _ in range(_numNodes)]
        cnt = 0
        for i in range(_numNodes):
            if vis[i] == 0:
                component = self.dfs(graph, i, vis)
                cnt += 1
        return cnt
    
    def fordDfs(self, cur, sink, vis, flow):
        """
        Computes a path from cur to sink, if it exists
        Params:
            cur: Current node in the depth-first search
            sink: Sink in the flow network
            vis: Visited array
            flow: Flow matrix consisting of edge capacities
        """
        _numNodes = len(flow)
        path = [cur]
        vis[cur] = 1
        if cur == sink:
            return True, path
        pathExists = False
        for node in range(_numNodes):
            if vis[node] == 0 and flow[cur][node] > 0:
                pathExists, childPath = self.fordDfs(node, sink, vis, flow)
                if pathExists:
                    path.extend(childPath)
                    break
        return pathExists, path
    
    def ford(self, source, sink, graph):
        """
        Implements Ford-Fulkerson's algorithm to compute max flow from source to sink
        Uses fordDfs() as a helper
        Params:
            source: Source in the flow network
            sink: Sink in the flow network
            graph: Adjacency list of the graph
        """
        _numNodes = len(graph)
        flow = []
        for i in range(_numNodes):
            flow.append([0 for _ in range(_numNodes)])
        for u in range(_numNodes):
            for v, c in graph[u]:
                flow[u][v] = c
        maxFlow = 0
        while(True):
            vis = [0 for _ in range(_numNodes)]
            pathExists, path = self.fordDfs(source, sink, vis, flow)
            if not pathExists:
                break
            minFlow = 10000000000
            for i in range(len(path) - 1):
                minFlow = min(minFlow, flow[path[i]][path[i+1]])
            maxFlow += minFlow
            for i in range(len(path) - 1):
                flow[path[i]][path[i+1]] -= minFlow
                flow[path[i+1]][path[i]] += minFlow
        return maxFlow

    def oneHot(self, subset):
        """
        One-hot encodes the nodes in subset
        Params:
            subset: Set of nodes to one-hot encode
        """
        ret = []
        for node in subset:
            gg = [0 for _ in range(self.numNodes)]
            gg[node] = 1
            ret.append(gg)
        return ret
        
    def getData(self, size):
        """
        Dataset generating method. Returns numpy arrays x and y corresponding to sets and set-labels
        Params:
            size: Size of dataset to be generated
        """
        if self.sink is None:
            self.sink = self.numNodes - 1
        nodeList = list(range(self.numNodes))
        nodeList.remove(self.sink)      
        x = []
        y = []
        for _ in range(size):
            subset = random.sample(nodeList, self.n)
            oh = self.oneHot(subset)
            dummySource = self.numNodes
            self.g.append([])
            for node in subset:
                self.g[dummySource].append((node, 100000000000))
            x.append(oh)
            y.append(self.ford(dummySource, self.sink, self.g))
            self.g = self.g[:self.numNodes]
        x = np.reshape(np.array(x), [-1, self.n, self.numNodes])
        y = np.reshape(np.array(y), [-1, 1])
        
        return x, y   

class TopEigenVectorSpikedCovDG(DataGenerator):
    """
    Data Generator for Top Eigenvector in Spiked Covariance Model Task
    """    
    def __init__(self, spike, n, dim):
        """
        Params:
            spike: A measure of magnitude of noise with which the covariance matrix is spiked
            n: Number of elements in a set
            dim: Dimensionality of each element in set
        """
        self.spike = spike
        self.n = n
        self.dim = dim
        self.cov = np.eye(self.dim)
        self.mu = np.zeros(self.dim)
        self.covH = self.spike * np.eye(self.dim)
         
    def getData(self, size):
        """
        Dataset generating method. Returns numpy arrays x and y corresponding to sets and set-labels
        Params:
            size: Size of dataset to be generated
        """
        x = []
        y = []
        for _ in range(size):
            v = np.random.multivariate_normal(self.mu, self.cov)
            v = v / np.linalg.norm(v)
            y.append(v)
            x_list = []
            for _ in range(self.n):
                z = np.random.normal(0, 1)
                h = np.random.multivariate_normal(self.mu, self.covH)
                x_list.append(z * v + h)
            x.append(x_list)
        x = np.array(x)
        y = np.array(y)
        x = np.reshape(x, (-1, self.n, self.dim))
        y = np.reshape(y, (-1, self.dim))
        return x, y
    