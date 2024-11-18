[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lng4wb3ya85nV8aI7ctEpEgFyYelZXka?usp=sharing)

# Log Anomaly Detection and Event Correlation

This repository provides a step-by-step pipeline for detecting anomalies and correlating events in log data. It uses Semantic Embeddings, Autoencoders, and Graph-Based Clustering to analyze log data and produce meaningful insights.

---

## **Overview**

Logs are critical for monitoring system behavior, diagnosing errors, and identifying issues. This project uses machine learning and graph algorithms to:
- Detect anomalous logs.
- Correlate related logs into clusters for incident investigation.

By combining semantic embeddings with anomaly detection and graph analysis, this approach enhances system observability and improves incident response.

---

## **Step-by-Step Explanation**

### **1. Loading and Preprocessing the Data**
```python
file_path = '/content/mapped_logs_2.csv'
log_data = pd.read_csv(file_path)
```
- **What this does**: Reads the log dataset (`mapped_logs_2.csv`) containing log messages and timestamps.
- **Why we use this**: Prepares the data for further processing like embedding generation and anomaly detection.

---

### **2. Semantic Embedding of Log Messages**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
log_data['embedding'] = list(model.encode(log_data['message'].fillna("").astype(str)))
```
- **What this does**: Uses a pre-trained Sentence Transformer (`all-MiniLM-L6-v2`) to convert log messages into semantic embeddings, which are numerical vectors capturing the meaning of the text.
- **Why we use this**: Enables similarity measurement between messages. For example:
  - Log A: "User login failed."
  - Log B: "Login attempt unsuccessful."
  - These logs have similar meanings, and their embeddings will be close in the vector space.

---

### **3. Autoencoder for Anomaly Detection**
```python
embeddings = np.array(log_data['embedding'].tolist())
input_dim = embeddings.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(embeddings, embeddings, epochs=100, batch_size=32, shuffle=True)
```
- **What this does**: Constructs an autoencoder to learn compressed representations of embeddings and reconstruct them.
  - **Input Layer**: Accepts embeddings.
  - **Encoded Layer**: Compresses embeddings to a smaller dimensional space (`64 dimensions`).
  - **Decoded Layer**: Reconstructs embeddings from the compressed space.
- **Why we use this**: High reconstruction errors indicate anomalies, as they deviate from the patterns learned by the autoencoder.

---

### **4. Reconstruction Error for Anomaly Detection**
```python
reconstructed = autoencoder.predict(embeddings)
reconstruction_error = np.mean((embeddings - reconstructed) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 95)
log_data['is_anomaly'] = reconstruction_error > threshold
```
- **What this does**: Calculates reconstruction errors for each log and flags anomalies exceeding the 95th percentile.
- **Why we use this**: Logs with high reconstruction errors are likely to be outliers, indicating potential issues.

---

### **5. Event Correlation Using Graphs**
```python
G = nx.Graph()

for index, row in log_data.iterrows():
    G.add_node(index, message=row['message'], timestamp=row['@timestamp'], anomaly=row['is_anomaly'])

for i in range(len(log_data)):
    for j in range(i + 1, len(log_data)):
        if pd.isnull(log_data.loc[i, '@timestamp']) or pd.isnull(log_data.loc[j, '@timestamp']):
            continue
        time_diff = abs((log_data.loc[i, '@timestamp'] - log_data.loc[j, '@timestamp']).total_seconds())
        if time_diff < 300:
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if sim > 0.8:
                G.add_edge(i, j, weight=sim)
```
- **What this does**:
  1. Creates a graph (`G`) where each log is a node with attributes like message, timestamp, and anomaly flag.
  2. Adds edges between nodes based on:
     - **Temporal Proximity**: Logs occur within 5 minutes (`300 seconds`).
     - **Semantic Similarity**: Cosine similarity exceeds `0.8`.
- **Why we use this**: Helps uncover relationships between logs, often clustering related events or cascading failures.

---

### **6. Cluster Detection**
```python
clusters = list(nx.connected_components(G))
print(f"Detected {len(clusters)} clusters of anomalies!")
```
- **What this does**: Finds connected components (clusters) in the graph.
- **Why we use this**: Clusters represent related logs. For example, several logs related to an authentication issue might form a cluster.

---

### **7. Incident Reporting**
```python
incidents = []
for cluster_id, cluster in enumerate(clusters):
    leading_edge = min(cluster, key=lambda node: G.nodes[node].get('timestamp', pd.Timestamp.max))
    related_logs = [G.nodes[node]['message'] for node in cluster]
    incidents.append({
        "incident_id": cluster_id,
        "leading_edge": G.nodes[leading_edge]['message'],
        "timestamp": G.nodes[leading_edge]['timestamp'],
        "related_logs": related_logs
    })

incident_df = pd.DataFrame(incidents)
incident_df.to_csv('/content/incidents.csv', index=False)
```
- **What this does**:
  - Identifies the earliest log in each cluster as the "leading edge."
  - Collects related logs in the cluster.
  - Saves the results as an incident report CSV.
- **Why we use this**: Provides actionable insights into incidents, allowing for better monitoring and debugging.

---

## **Example**

### Input Data (`log_data.csv`):
| message                  | @timestamp           |
|--------------------------|----------------------|
| "User login failed."     | 2024-11-18 10:00:00 |
| "Login unsuccessful."    | 2024-11-18 10:01:00 |
| "Service unavailable."   | 2024-11-18 10:05:00 |

### Output (`incidents.csv`):
| incident_id | leading_edge         | timestamp           | related_logs                                |
|-------------|----------------------|---------------------|---------------------------------------------|
| 0           | "User login failed." | 2024-11-18 10:00:00 | ["User login failed.", "Login unsuccessful."] |

---

## **Why This Approach?**

1. **Semantic Understanding**: Embeddings capture the meaning of log messages, enabling similarity-based analysis.
2. **Anomaly Detection**: Reconstruction errors effectively identify outliers in log patterns.
3. **Event Correlation**: Graph-based clustering highlights relationships between events, simplifying root cause analysis.
4. **Actionable Insights**: Generates detailed incident reports for system observability.

---

## **Improvements and Customizations**
1. **Threshold Tuning**:
   - Adjust the percentile for anomaly detection.
   - Modify time and similarity thresholds for event correlation.
2. **Enhanced Features**:
   - Add domain-specific log parsing.
   - Integrate with real-time systems like Kafka.
3. **Visualization**:
   - Visualize clusters using tools like Gephi or Matplotlib.

---

## **Outputs**
- **Anomaly Detection Results**: Adds an `is_anomaly` column to the original dataset.
- **Incident Report**: Outputs a CSV file with cluster details.
