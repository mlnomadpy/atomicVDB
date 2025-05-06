/**
 * atomicVDB - A high-performance vector storage and clustering library
 * 
 * Features:
 * - Efficient vector storage with customizable similarity metrics
 * - Dynamic clustering with configurable parameters
 * - Optimized nearest neighbor search
 * - Vector indexing and metadata support
 * - TypeScript-friendly JSDoc annotations
 */

/**
 * @typedef {Object} VectorEntry
 * @property {string} id - Unique identifier
 * @property {number[]} vector - The vector data
 * @property {Object|null} metadata - Optional metadata
 */

/**
 * @typedef {Object} Cluster
 * @property {string} id - Unique cluster identifier
 * @property {number[]} center - Vector representing the center of the cluster
 * @property {VectorEntry[]} members - Vectors belonging to this cluster
 */

/**
 * @typedef {Object} ClusterSummary
 * @property {string} id - Unique cluster identifier
 * @property {number[]} center - Vector representing the center of the cluster
 * @property {number} size - Number of vectors in the cluster
 * @property {number} [radius] - Maximum distance from center to any member
 */

/**
 * @typedef {Object} SearchResult
 * @property {VectorEntry} entry - The vector entry
 * @property {number} similarity - Similarity score
 * @property {string} clusterId - ID of the cluster containing this entry
 */

/**
 * @typedef {Object} atomicVDBOptions
 * @property {Function} [similarityFn] - Function to calculate similarity between vectors
 * @property {number} [clusterThreshold=0.85] - Similarity threshold for joining existing clusters
 * @property {boolean} [dynamicClustering=true] - Whether to create new clusters automatically
 * @property {boolean} [recalculateCenters=true] - Whether to recalculate cluster centers on insert
 * @property {number} [maxClusters=100] - Maximum number of clusters
 */

/**
 * Calculate cosine similarity between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} Similarity score between -1 and 1
 */
function cosineSimilarity(a, b) {
    if (a.length !== b.length) {
      throw new Error(`Vector dimensions don't match: ${a.length} vs ${b.length}`);
    }
    
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    
    // Handle zero vectors
    if (magA === 0 || magB === 0) return 0;
    
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
  }
  
  /**
   * Calculate Euclidean distance between two vectors
   * @param {number[]} a - First vector
   * @param {number[]} b - Second vector
   * @returns {number} Distance (lower is more similar)
   */
  function euclideanDistance(a, b) {
    if (a.length !== b.length) {
      throw new Error(`Vector dimensions don't match: ${a.length} vs ${b.length}`);
    }
    
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
  
  /**
   * Convert Euclidean distance to a similarity score
   * @param {number} distance - Euclidean distance
   * @returns {number} Similarity score between 0 and 1
   */
  function euclideanSimilarity(a, b) {
    const distance = euclideanDistance(a, b);
    // Convert distance to similarity (1 when identical, approaching 0 as distance increases)
    return 1 / (1 + distance);
  }
  
  /**
   * Generate a UUID
   * @returns {string} A UUID string
   */
  function uuid() {
    return typeof crypto !== 'undefined' && crypto.randomUUID 
      ? crypto.randomUUID()
      : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
          const r = Math.random() * 16 | 0;
          return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
  }
  
  /**
   * Enhanced vector store with clustering capabilities
   */
  class atomicVDB {
    /**
     * Create a new clustered vector store
     * @param {atomicVDBOptions} [options] - Configuration options
     */
    constructor(options = {}) {
      this.options = {
        similarityFn: options.similarityFn || cosineSimilarity,
        clusterThreshold: options.clusterThreshold ?? 0.85,
        dynamicClustering: options.dynamicClustering ?? true,
        recalculateCenters: options.recalculateCenters ?? true,
        maxClusters: options.maxClusters ?? 100
      };
      
      /** @type {Cluster[]} */
      this.clusters = [];
      
      /** @type {Object.<string, VectorEntry>} */
      this.vectorIndex = {};
      
      /** @type {Object.<string, string>} */
      this.vectorToCluster = {};
      
      /** @type {number|null} */
      this.dimensions = null;
    }
  
    /**
     * Add a new cluster with the given vector as its center
     * @param {number[]} initialVector - Vector to use as the cluster center
     * @param {Object} [metadata] - Optional metadata for the initial vector
     * @returns {string} ID of the new cluster
     */
    addCluster(initialVector, metadata = null) {
      // Set dimensions if this is the first vector
      if (this.dimensions === null) {
        this.dimensions = initialVector.length;
      } else if (initialVector.length !== this.dimensions) {
        throw new Error(`Vector dimension mismatch: expected ${this.dimensions}, got ${initialVector.length}`);
      }
      
      // Check if we've exceeded the maximum cluster limit
      if (this.clusters.length >= this.options.maxClusters) {
        throw new Error(`Maximum number of clusters (${this.options.maxClusters}) reached`);
      }
      
      const clusterId = uuid();
      const entryId = uuid();
      
      const entry = {
        id: entryId,
        vector: [...initialVector], // Clone the vector to prevent mutations
        metadata
      };
      
      this.vectorIndex[entryId] = entry;
      this.vectorToCluster[entryId] = clusterId;
      
      this.clusters.push({
        id: clusterId,
        center: [...initialVector],
        members: [entry],
        radius: 0 // Initialize radius to 0
      });
      
      return clusterId;
    }
  
    /**
     * Insert a vector into the store
     * @param {number[]} vector - The vector to insert
     * @param {Object} [metadata] - Optional metadata to associate with the vector
     * @returns {string} ID of the vector entry
     */
    insert(vector, metadata = null) {
      // Validate vector
      if (!Array.isArray(vector) || vector.length === 0) {
        throw new Error('Vector must be a non-empty array of numbers');
      }
      
      for (let i = 0; i < vector.length; i++) {
        if (typeof vector[i] !== 'number' || isNaN(vector[i])) {
          throw new Error(`Invalid vector: element at index ${i} is not a number`);
        }
      }
      
      // Set dimensions if this is the first vector
      if (this.dimensions === null) {
        this.dimensions = vector.length;
      } else if (vector.length !== this.dimensions) {
        throw new Error(`Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`);
      }
      
      const entry = {
        id: uuid(),
        vector: [...vector], // Clone the vector to prevent mutations
        metadata
      };
      
      // Store the vector in the index
      this.vectorIndex[entry.id] = entry;
      
      // If there are no clusters yet, create the first one
      if (this.clusters.length === 0) {
        const clusterId = this.addCluster(vector);
        this.vectorToCluster[entry.id] = clusterId;
        return entry.id;
      }
      
      // Find closest cluster center
      let bestCluster = null;
      let bestSimilarity = -Infinity;
      
      for (const cluster of this.clusters) {
        const similarity = this.options.similarityFn(vector, cluster.center);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestCluster = cluster;
        }
      }
      
      // Decide whether to add to an existing cluster or create a new one
      if (bestSimilarity >= this.options.clusterThreshold) {
        // Add to existing cluster
        bestCluster.members.push(entry);
        this.vectorToCluster[entry.id] = bestCluster.id;
        
        // Update the cluster center if configured to do so
        if (this.options.recalculateCenters) {
          bestCluster.center = this._recalculateCenter(bestCluster);
        }
        
        // Update cluster radius
        this._updateClusterRadius(bestCluster);
      } else if (this.options.dynamicClustering) {
        // Create a new cluster
        const clusterId = this.addCluster(vector, metadata);
        this.vectorToCluster[entry.id] = clusterId;
      } else {
        // Force into best cluster even though it's below threshold
        bestCluster.members.push(entry);
        this.vectorToCluster[entry.id] = bestCluster.id;
        
        // Update the cluster center if configured to do so
        if (this.options.recalculateCenters) {
          bestCluster.center = this._recalculateCenter(bestCluster);
        }
        
        // Update cluster radius
        this._updateClusterRadius(bestCluster);
      }
      
      return entry.id;
    }
  
    /**
     * Update the radius of a cluster
     * @param {Cluster} cluster - The cluster to update
     * @private
     */
    _updateClusterRadius(cluster) {
      let maxDistance = 0;
      
      for (const member of cluster.members) {
        // Calculate distance from center
        let distance;
        if (this.options.similarityFn === cosineSimilarity) {
          // For cosine similarity, distance = 1 - similarity
          const similarity = cosineSimilarity(member.vector, cluster.center);
          distance = 1 - similarity;
        } else {
          // For other similarity functions, use Euclidean distance
          distance = euclideanDistance(member.vector, cluster.center);
        }
        
        maxDistance = Math.max(maxDistance, distance);
      }
      
      cluster.radius = maxDistance;
    }
  
    /**
     * Recalculate the center of a cluster
     * @param {Cluster} cluster - The cluster to recalculate
     * @returns {number[]} The new center vector
     * @private
     */
    _recalculateCenter(cluster) {
      // If there's only one member, use its vector as the center
      if (cluster.members.length === 1) {
        return [...cluster.members[0].vector];
      }
      
      const dim = this.dimensions;
      const sum = new Array(dim).fill(0);
      
      // Sum all vectors
      for (const { vector } of cluster.members) {
        for (let i = 0; i < dim; i++) {
          sum[i] += vector[i];
        }
      }
      
      // Divide by count to get average
      return sum.map(val => val / cluster.members.length);
    }
  
    /**
     * Get all clusters in the store
     * @param {boolean} [includeRadius=true] - Whether to include radius information
     * @returns {ClusterSummary[]} Summary of all clusters
     */
    getClusters(includeRadius = true) {
      return this.clusters.map(c => {
        const summary = {
          id: c.id,
          center: c.center,
          size: c.members.length
        };
        
        if (includeRadius) {
          summary.radius = c.radius;
        }
        
        return summary;
      });
    }
  
    /**
     * Get a specific cluster by ID
     * @param {string} id - The ID of the cluster to retrieve
     * @returns {Cluster|null} The cluster, or null if not found
     */
    getClusterById(id) {
      return this.clusters.find(c => c.id === id) || null;
    }
  
    /**
     * Get all vectors in the store
     * @returns {VectorEntry[]} All vector entries
     */
    getAllVectors() {
      return Object.values(this.vectorIndex);
    }
  
    /**
     * Find the most similar vectors to the query vector
     * @param {number[]} queryVector - The vector to compare against
     * @param {Object} [options] - Search options
     * @param {number} [options.limit=10] - Maximum number of results
     * @param {number} [options.minSimilarity=0] - Minimum similarity threshold
     * @param {boolean} [options.searchAllClusters=false] - Whether to search in all clusters
     * @returns {SearchResult[]} The most similar vectors with their similarity scores
     */
    search(queryVector, options = {}) {
      const {
        limit = 10,
        minSimilarity = 0,
        searchAllClusters = false
      } = options;
      
      if (this.clusters.length === 0) {
        return [];
      }
      
      // Validate vector dimensions
      if (queryVector.length !== this.dimensions) {
        throw new Error(`Vector dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`);
      }
      
      const results = [];
      
      if (searchAllClusters) {
        // Search all vectors regardless of clusters
        for (const entry of Object.values(this.vectorIndex)) {
          const similarity = this.options.similarityFn(queryVector, entry.vector);
          
          if (similarity >= minSimilarity) {
            results.push({
              entry,
              similarity,
              clusterId: this.vectorToCluster[entry.id]
            });
          }
        }
      } else {
        // First, find clusters that might contain similar vectors
        const clusterSimilarities = this.clusters.map(cluster => ({
          cluster,
          similarity: this.options.similarityFn(queryVector, cluster.center)
        }));
        
        // Sort clusters by similarity to query
        clusterSimilarities.sort((a, b) => b.similarity - a.similarity);
        
        // Search within clusters, starting with the most similar
        for (const { cluster, similarity } of clusterSimilarities) {
          // Skip clusters that are too dissimilar
          if (similarity < minSimilarity) continue;
          
          // Search within this cluster
          for (const entry of cluster.members) {
            const similarity = this.options.similarityFn(queryVector, entry.vector);
            
            if (similarity >= minSimilarity) {
              results.push({
                entry,
                similarity,
                clusterId: cluster.id
              });
            }
          }
          
          // If we have enough results, stop searching
          if (results.length >= limit) break;
        }
      }
      
      // Sort results by similarity (highest first)
      results.sort((a, b) => b.similarity - a.similarity);
      
      // Limit the number of results
      return results.slice(0, limit);
    }
  
    /**
     * Get a vector by its ID
     * @param {string} id - The ID of the vector to retrieve
     * @returns {VectorEntry|null} The vector entry, or null if not found
     */
    getVectorById(id) {
      return this.vectorIndex[id] || null;
    }
  
    /**
     * Remove a vector from the store
     * @param {string} id - The ID of the vector to remove
     * @returns {boolean} Whether the vector was successfully removed
     */
    removeVector(id) {
      const entry = this.vectorIndex[id];
      if (!entry) return false;
      
      const clusterId = this.vectorToCluster[id];
      const cluster = this.getClusterById(clusterId);
      
      if (cluster) {
        // Remove from cluster members
        cluster.members = cluster.members.filter(m => m.id !== id);
        
        // Delete from indexes
        delete this.vectorIndex[id];
        delete this.vectorToCluster[id];
        
        // Recalculate cluster center if necessary
        if (cluster.members.length > 0) {
          if (this.options.recalculateCenters) {
            cluster.center = this._recalculateCenter(cluster);
          }
          this._updateClusterRadius(cluster);
        } else {
          // Remove empty cluster
          this.clusters = this.clusters.filter(c => c.id !== clusterId);
        }
        
        return true;
      }
      
      return false;
    }
  
    /**
     * Update a vector's metadata
     * @param {string} id - The ID of the vector to update
     * @param {Object} metadata - The new metadata
     * @returns {boolean} Whether the vector was successfully updated
     */
    updateMetadata(id, metadata) {
      const entry = this.vectorIndex[id];
      if (!entry) return false;
      
      entry.metadata = metadata;
      return true;
    }
  
    /**
     * Merge two clusters
     * @param {string} clusterId1 - ID of the first cluster
     * @param {string} clusterId2 - ID of the second cluster
     * @returns {string} ID of the merged cluster
     */
    mergeClusters(clusterId1, clusterId2) {
      const cluster1 = this.getClusterById(clusterId1);
      const cluster2 = this.getClusterById(clusterId2);
      
      if (!cluster1 || !cluster2) {
        throw new Error('One or both cluster IDs are invalid');
      }
      
      // Merge members from cluster2 into cluster1
      cluster1.members = [...cluster1.members, ...cluster2.members];
      
      // Update vectorToCluster index
      for (const member of cluster2.members) {
        this.vectorToCluster[member.id] = clusterId1;
      }
      
      // Remove cluster2
      this.clusters = this.clusters.filter(c => c.id !== clusterId2);
      
      // Recalculate center of merged cluster
      if (this.options.recalculateCenters) {
        cluster1.center = this._recalculateCenter(cluster1);
      }
      
      // Update cluster radius
      this._updateClusterRadius(cluster1);
      
      return clusterId1;
    }
  
    /**
     * Split a cluster into two using k-means
     * @param {string} clusterId - ID of the cluster to split
     * @returns {string[]} IDs of the resulting clusters
     */
    splitCluster(clusterId) {
      const cluster = this.getClusterById(clusterId);
      
      if (!cluster) {
        throw new Error('Invalid cluster ID');
      }
      
      if (cluster.members.length < 2) {
        throw new Error('Cannot split a cluster with fewer than 2 members');
      }
      
      // Simple k-means implementation for splitting
      // 1. Choose two initial centers
      const dim = this.dimensions;
      
      // Find the two most distant vectors in the cluster
      let maxDistance = -1;
      let center1Index = 0;
      let center2Index = 0;
      
      for (let i = 0; i < cluster.members.length; i++) {
        for (let j = i + 1; j < cluster.members.length; j++) {
          const dist = 1 - this.options.similarityFn(
            cluster.members[i].vector,
            cluster.members[j].vector
          );
          
          if (dist > maxDistance) {
            maxDistance = dist;
            center1Index = i;
            center2Index = j;
          }
        }
      }
      
      // Create two new clusters
      const cluster1Id = uuid();
      const cluster2Id = uuid();
      
      const cluster1 = {
        id: cluster1Id,
        center: [...cluster.members[center1Index].vector],
        members: [cluster.members[center1Index]],
        radius: 0
      };
      
      const cluster2 = {
        id: cluster2Id,
        center: [...cluster.members[center2Index].vector],
        members: [cluster.members[center2Index]],
        radius: 0
      };
      
      // Update index for initial members
      this.vectorToCluster[cluster.members[center1Index].id] = cluster1Id;
      this.vectorToCluster[cluster.members[center2Index].id] = cluster2Id;
      
      // Assign remaining vectors to the closest center
      for (let i = 0; i < cluster.members.length; i++) {
        if (i === center1Index || i === center2Index) continue;
        
        const member = cluster.members[i];
        const sim1 = this.options.similarityFn(member.vector, cluster1.center);
        const sim2 = this.options.similarityFn(member.vector, cluster2.center);
        
        if (sim1 > sim2) {
          cluster1.members.push(member);
          this.vectorToCluster[member.id] = cluster1Id;
        } else {
          cluster2.members.push(member);
          this.vectorToCluster[member.id] = cluster2Id;
        }
      }
      
      // Recalculate centers
      if (this.options.recalculateCenters) {
        cluster1.center = this._recalculateCenter(cluster1);
        cluster2.center = this._recalculateCenter(cluster2);
      }
      
      // Update radii
      this._updateClusterRadius(cluster1);
      this._updateClusterRadius(cluster2);
      
      // Replace the old cluster with the two new ones
      this.clusters = this.clusters.filter(c => c.id !== clusterId);
      this.clusters.push(cluster1, cluster2);
      
      return [cluster1Id, cluster2Id];
    }
  
    /**
     * Get statistics about the vector store
     * @returns {Object} Statistics about the store
     */
    getStats() {
      const numClusters = this.clusters.length;
      const numVectors = Object.keys(this.vectorIndex).length;
      const dimensions = this.dimensions;
      
      let minClusterSize = Infinity;
      let maxClusterSize = 0;
      let avgClusterSize = 0;
      
      for (const cluster of this.clusters) {
        const size = cluster.members.length;
        minClusterSize = Math.min(minClusterSize, size);
        maxClusterSize = Math.max(maxClusterSize, size);
        avgClusterSize += size;
      }
      
      if (numClusters > 0) {
        avgClusterSize /= numClusters;
      } else {
        minClusterSize = 0;
      }
      
      return {
        numVectors,
        numClusters,
        dimensions,
        clusterStats: {
          min: minClusterSize,
          max: maxClusterSize,
          avg: avgClusterSize
        }
      };
    }
  
    /**
     * Export the store for serialization
     * @returns {Object} Serializable representation of the store
     */
    export() {
      return {
        dimensions: this.dimensions,
        options: this.options,
        clusters: this.clusters,
        vectorToCluster: this.vectorToCluster
      };
    }
  
    /**
     * Import data into the store
     * @param {Object} data - Data exported from another store
     * @returns {atomicVDB} The updated store instance
     */
    static import(data) {
      const store = new atomicVDB(data.options);
      
      store.dimensions = data.dimensions;
      store.clusters = data.clusters;
      store.vectorToCluster = data.vectorToCluster;
      
      // Rebuild the vector index
      store.vectorIndex = {};
      for (const cluster of store.clusters) {
        for (const entry of cluster.members) {
          store.vectorIndex[entry.id] = entry;
        }
      }
      
      return store;
    }
  }
  
  // Export similarity functions
  atomicVDB.similarities = {
    cosine: cosineSimilarity,
    euclidean: euclideanSimilarity
  };
  
  // Export for browser or Node
  if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = atomicVDB;
  } else {
    window.atomicVDB = atomicVDB;
  }