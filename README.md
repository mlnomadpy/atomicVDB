# atomicVDB

## Usage Examples

### Basic Usage

```javascript
// Create a vector store with default options (cosine similarity)
const store = new atomicVDB();

// Insert vectors with metadata
store.insert([1, 0, 0], { label: 'x-axis' });
store.insert([0, 1, 0], { label: 'y-axis' });
store.insert([0, 0, 1], { label: 'z-axis' });

// Search for similar vectors
const results = store.search([0.9, 0.1, 0], { limit: 5 });
console.log(results);
```

### Custom Configuration

```javascript
// Create with custom options
const store = new atomicVDB({
  similarityFn: atomicVDB.similarities.euclidean,
  clusterThreshold: 0.7,
  dynamicClustering: true,
  recalculateCenters: true,
  maxClusters: 50
});

// Work with clusters directly
const clusterId = store.addCluster([1, 1, 1]);
console.log(store.getClusters());
```

### Advanced Operations

```javascript
// Get statistics
const stats = store.getStats();
console.log(`Total vectors: ${stats.numVectors}, Clusters: ${stats.numClusters}`);

// Manipulate clusters
const [cluster1Id, cluster2Id] = store.splitCluster(existingClusterId);
store.mergeClusters(cluster1Id, cluster2Id);

// Export/import
const data = store.export();
localStorage.setItem('vectorStore', JSON.stringify(data));

// Later...
const loaded = atomicVDB.import(
  JSON.parse(localStorage.getItem('vectorStore'))
);
```
