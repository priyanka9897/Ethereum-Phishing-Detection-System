# HT-GNN Training Phase: Detailed Input-to-Output Flow

## Overview
This document explains exactly what happens when **one sample** passes through the HT-GNN model during training.

---

## 1. Input: What Does the Model Receive?

### 1.1 Raw Input (Single Sample from Training Batch)
```
data = HeteroData object  (PyTorch Geometric heterogeneous graph)
label = 0 or 1           (0=benign, 1=phishing)
address = "0x1234..."    (for tracking)
```

### 1.2 HeteroData Structure
```python
HeteroData {
  # Node types with features
  node_types: ['eoa', 'contract']
  
  eoa: {
    x: torch.Tensor(num_eoa_nodes, 10)          # 10 features per EOA
    target_mask: torch.Tensor(num_eoa_nodes)    # 1 if this is the target address
  }
  
  contract: {
    x: torch.Tensor(num_contract_nodes, 8)     # 8 features per contract
  }
  
  # Edge connectivity
  ('eoa', 'sends', 'eoa'): {
    edge_index: torch.Tensor(2, num_edges)     # source_idx, target_idx
    edge_attr: torch.Tensor(num_edges, 6)      # [fee, value, t_sin, t_cos, ...]
  }
  
  ('eoa', 'calls', 'contract'): {
    edge_index: torch.Tensor(2, num_edges)
    edge_attr: torch.Tensor(num_edges, 6)
  }
  
  ('contract', 'returns', 'eoa'): {
    edge_index: torch.Tensor(2, num_edges)
    edge_attr: torch.Tensor(num_edges, 6)
  }
  
  ('eoa', 'approves', 'contract'): {
    edge_index: torch.Tensor(2, num_edges)
    edge_attr: torch.Tensor(num_edges, 6)
  }
}
```

### 1.3 Example Sizes
```
Typical transaction subgraph:
├─ EOA nodes:       8-15 addresses
├─ Contract nodes:  2-4 contracts
├─ Edges:           20-40 interactions
└─ Target:         1 (the address being analyzed)
```

---

## 2. Forward Pass: Step-by-Step

### Step 0: Move to Device
```python
data = data.to(device)  # GPU or CPU
# Now all tensors on same device
```

### Step 1: Input Projection
**Goal**: Convert raw features to latent space

```python
# Current state:
# EOA:      x_eoa = (num_eoas, 10)
# Contract: x_contract = (num_contracts, 8)

# Projection layer (Linear)
x_dict = {}
x_dict['eoa']      = F.relu(self.proj['eoa'](x_eoa))
                     # (num_eoas, 10) → Linear(10→128) → (num_eoas, 128)
                     
x_dict['contract'] = F.relu(self.proj['contract'](x_contract))
                     # (num_contracts, 8) → Linear(8→128) → (num_contracts, 128)

# Output:
# x_dict['eoa']:      (num_eoas, 128)
# x_dict['contract']: (num_contracts, 128)
```

**Visualization**:
```
[f1, f2, ..., f10] (10 raw features per EOA)
        ↓ Linear(10→128)
[h1, h2, ..., h128] (128 hidden features per EOA)
```

---

### Step 2: Heterogeneous Message Passing (Layer 1)

Now the graph itself becomes important. We use **HGTConv** (Heterogeneous Graph Transformer Convolution).

#### 2.1 HGTConv Forward: Multi-Head Attention
```python
# Input to HGTConv:
#   x_dict: embeddings per node type
#   edge_index_dict: connectivity
#   metadata: tells HGTConv which edges exist

output = HGTConv(x_dict, edge_index_dict)

# What HGTConv does internally (high level):
#
# For each edge type (e.g., 'eoa' --sends--> 'eoa'):
#   1. Get source nodes: h_src = x_dict['eoa'][src_idx]
#   2. Get target nodes: h_tgt = x_dict['eoa'][tgt_idx]
#   3. Multi-head attention:
#        Q = W_q @ h_tgt           (query from target)
#        K = W_k @ h_src           (key from source)
#        V = W_v @ h_src           (value from source)
#        attn_weight = softmax(Q·K^T / sqrt(d))
#        message = attn_weight @ V  (weighted value)
#   4. Aggregate messages to target node:
#        h_tgt_new += message (sum over all incoming edges of this type)
#
# Output: updated embeddings for each node

# Shapes after HGTConv:
# output['eoa']:      (num_eoas, 128)
# output['contract']: (num_contracts, 128)
```

**Why this works for phishing detection:**
- Attention learns which neighbors matter (e.g., high-frequency traders, flagged contracts)
- Multi-head captures different interaction types (value, frequency, timing)
- Heterogeneous = respects edge types (sends ≠ calls ≠ approves)

#### 2.2 Temporal Attention Gate
```python
# After HGTConv, apply temporal gating to incorporate time signals

# First, aggregate time features per node
for each node_type in ['eoa', 'contract']:
    time_signal = aggregate_time_from_edges(data, node_type)
    # (num_nodes, 4)  [time2vec features]
    
    h_old = x_dict[node_type]                    # (num_nodes, 128)
    h_new_from_hgt = output[node_type]           # (num_nodes, 128)
    
    # Temporal attention (gating mechanism):
    t_emb = self.W_time(time_signal)             # (num_nodes, 128)
    gate = sigmoid(self.W_gate([h_new, t_emb]))  # (num_nodes, 128)
    
    h_gated = gate * h_new_from_hgt + (1 - gate) * t_emb
    # Element-wise: if gate ≈ 1, keep HGT output
    #               if gate ≈ 0, prefer time signal
    
    # Apply residual + layer norm
    h_final = self.norm[node_type](h_gated + h_old)
    h_final = self.dropout(h_final)
    
    x_dict[node_type] = h_final  # (num_nodes, 128)
```

**Why temporal gating?**
- Captures time-based patterns (e.g., "this address sends at 3 AM every Friday" = suspicious)
- Balances graph structure (neighbors) with timing (when interactions happen)
- Phishing often has unusual temporal patterns

#### 2.3 Output after Layer 1
```
x_dict['eoa']:      (num_eoas, 128)      [updated with attention + temporal info]
x_dict['contract']: (num_contracts, 128) [updated with attention + temporal info]
```

---

### Step 3: Heterogeneous Message Passing (Layer 2)

Repeat Step 2 again:
```python
# Same process: HGTConv → TemporalAttention → Residual + LayerNorm + Dropout
# 
# Now the model can integrate information from 2-hop neighborhoods
# (address → neighbors → their neighbors)

x_dict = apply_layer_2(x_dict, data)
# 
# x_dict['eoa']:      (num_eoas, 128)      [aggregated from 2-hops]
# x_dict['contract']: (num_contracts, 128)
```

**What changed?**
- First layer: direct neighbors (who did this address directly interact with?)
- Second layer: indirect patterns (who do their neighbors interact with?)

---

### Step 4: Extract Target Embedding

Find the embedding of the **target address** (the one we're analyzing):
```python
# The target EOA node is marked with:
# data['eoa'].target_mask = [0, 0, 1, 0, ...]  (1 at target position)

mask = data['eoa'].target_mask          # (num_eoas,)
target_emb = x_dict['eoa'][mask]        # (1, 128)  ← Only the target row

# Shape: (1, 128)
# Contains: aggregated info about the target address from its transaction graph
```

**What does this embedding represent?**
- Node features (balance, frequency, etc.)
- Neighborhood patterns (who do they interact with?)
- Temporal behavior (when do interactions happen?)
- 2-layer aggregation (multi-hop structure)

---

### Step 5: Classification Head

Convert embedding to phishing probability:
```python
# Input: target_emb = (1, 128)

# Layer 1: Hidden layer
hidden = self.classifier[0](target_emb)  # Linear(128→64)
# hidden = (1, 64)

# Activation
hidden = F.relu(hidden)
# hidden = (1, 64)  [ReLU = max(0, x)]

# Dropout (regularization during training)
hidden = self.dropout(hidden)
# hidden = (1, 64)  [some units zeroed randomly]

# Layer 2: Output logits
logits = self.classifier[2](hidden)      # Linear(64→2)
# logits = (1, 2)  = [score_benign, score_phishing]

# Softmax to probabilities
proba = F.softmax(logits, dim=-1)
# proba = (1, 2)  = [P(benign), P(phishing)]
#         where P(benign) + P(phishing) = 1

# Extract phishing probability
p_phishing = proba[0, 1].item()
# p_phishing ∈ [0, 1]  ← What we predict
```

**Example output:**
```
logits = [-0.5, 0.8]
proba = softmax(logits) = [0.27, 0.73]
p_phishing = 0.73  ← 73% chance this is phishing
```

---

## 3. Loss Computation

### 3.1 Binary Cross-Entropy (CE) Component

```python
target = torch.tensor([label], device=device)  # label = 0 or 1
logits = model(data)['logits']                  # (1, 2)

ce_loss = F.cross_entropy(logits, target)

# Internally, PyTorch does:
# 1. logits = [-0.5, 0.8]
# 2. softmax = [0.27, 0.73]
# 3. If label=1 (phishing):
#    ce_loss = -log(0.73) ≈ 0.315
# 4. If label=0 (benign):
#    ce_loss = -log(0.27) ≈ 1.31

# Low loss = high confidence in correct class
# High loss = low confidence (penalizes being wrong)
```

### 3.2 Contrastive Loss Component

```python
embeddings = model(data)['embeddings']['eoa'][0:1]  # (1, 128) target embedding
target_label = torch.tensor([label], device=device)  # 0 or 1

contra_loss = contrastive_loss(embeddings, target_label, margin=1.0)

# Contrastive loss encourages:
# - If label=1 (phishing): embedding stays unique
# - If label=0 (benign): embedding pulls toward benign cluster
# - Deepens the separation in embedding space
```

### 3.3 Combined Loss

```python
# Combine both (80% CE, 20% contrastive for regularization)
total_loss = 0.8 * ce_loss + 0.2 * contra_loss

# Example:
# ce_loss = 0.315
# contra_loss = 0.05
# total_loss = 0.8*0.315 + 0.2*0.05 = 0.262

# This is the ONE sample loss
# Later, average over all N samples in the batch for epoch loss
```

---

## 4. Backward Pass (Gradient Computation)

### 4.1 Backpropagation
```python
total_loss.backward()

# PyTorch traces backward through:
# Classification head → HGT layers → Input projection
# Computing ∂loss/∂w for every weight w

# Gradient shapes match parameter shapes:
# Linear weights:     (out_dim, in_dim)
# HGTConv weights:    Various (attention matrices, etc.)
# LayerNorm params:   (dim,)
```

**Computational graph:**
```
target_emb (1,128)
    ↓ Linear(128→64)
hidden (1,64)
    ↓ ReLU
hidden_activated (1,64)
    ↓ Dropout
hidden_dropped (1,64)
    ↓ Linear(64→2)
logits (1,2)
    ↓ Cross-entropy loss
loss = scalar
    ↓ .backward()
    ∇w₁, ∇w₂, ∇w₃, ... (gradients flow up)
```

### 4.2 Gradient Clipping (Numerical Stability)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Prevents exploding gradients
# If ||∇|| > 1.0, scale all gradients down: ∇ ← ∇ / (||∇|| / 1.0)
```

---

## 5. Parameter Update (Optimizer Step)

### 5.1 AdamW Optimizer
```python
optimizer.step()

# For each parameter w with gradient ∇w:
# - Compute exponential moving average of gradients (momentum)
# - Compute exponential moving average of squared gradients (adaptive LR)
# - Update: w ← w - lr * m_hat / (sqrt(v_hat) + ε)
#
# Where:
#   lr = learning rate (5e-4)
#   m_hat = bias-corrected momentum
#   v_hat = bias-corrected squared gradient average
#   ε = small constant for numerical stability

# AdamW also includes weight decay (L2 regularization)
# Prevents overfitting by penalizing large weights
```

---

## 6. Validation & Early Stopping

After each epoch:
```python
# Evaluate on validation set (100 samples)
val_auc, val_f1, val_probs, val_preds = evaluate(model, val_set, device)

# If validation AUC improves:
if val_auc > best_auc:
    best_auc = val_auc
    torch.save(model.state_dict(), model_path)
    # Save checkpoint
```

---

## 7. Complete Training Loop (One Epoch)

```python
def train_epoch(model, dataset, optimizer, device):
    model.train()
    random.shuffle(dataset)
    total_loss = 0.0

    for data, label, address in dataset:  # Each sample one at a time
        # 1. Move to device
        data = data.to(device)
        
        # 2. Forward: compute loss
        out = model(data)
        logits = out["logits"]                    # (1, 2)
        embeddings = out["embeddings"]["eoa"][0:1]
        target = torch.tensor([label], device=device)
        
        ce_loss = F.cross_entropy(logits, target)
        contra_loss = contrastive_loss(embeddings, target, margin=1.0)
        total_task_loss = 0.8 * ce_loss + 0.2 * contra_loss
        
        # 3. Zero gradients
        optimizer.zero_grad()
        
        # 4. Backward pass
        total_task_loss.backward()
        
        # 5. Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 6. Update weights
        optimizer.step()
        
        # 7. Accumulate loss
        total_loss += total_task_loss.item()
    
    # Return average epoch loss
    return total_loss / len(dataset)
```

---

## 8. Example: One Training Step Visualized

Let's trace ONE sample (address 0x1234) as it goes through training:

### Input
```
address: 0x1234...
label: 1 (phishing)
graph: 12 EOA nodes, 3 contracts, 28 edges
target_mask: [0,0,1,0,0,0,0,0,0,0,0,0]  (3rd EOA is target)
```

### Forward Pass

**Projection:**
```
EOA nodes: (12, 10) → Linear(10→128) → (12, 128)
Contracts: (3, 8) → Linear(8→128) → (3, 128)
```

**Layer 1 HGTConv + Temporal:**
```
For each node, aggregate from neighbors:
  Target node (position 2):
    - Receives attention from 5 neighbors (who contacted this address)
    - Weight by their importance (high-value senders matter more)
    - Merge with temporal signal (interaction timing)
  Result: refined embedding capturing local neighborhood
```

**Layer 2 HGTConv + Temporal:**
```
For target node, aggregate from neighbors' neighbors:
  - Neighbors' interactions reveal indirect patterns
  - E.g., "my neighbors only talk to Chainalysis honeypots"
  Result: global context embedding
```

**Extract Target:**
```
target_emb = embeddings[2] = (128,) vector containing:
  ✓ Direct neighbor patterns
  ✓ Indirect interaction chains
  ✓ Temporal behavior
  ✓ Aggregated across 2-hops
```

**Classification:**
```
target_emb (128,) → FC layers → logits [-0.3, 1.2] → softmax → [0.21, 0.79]
⟹ P(phishing) = 0.79
```

### Loss Computation
```
CE loss:     -log(0.79) = 0.236
Contrastive: 0.08 (embeddings being nicely separated in cluster)
Total loss:  0.8*0.236 + 0.2*0.08 = 0.205
```

### Backward & Update
```
Gradient computed through all layers
All weights updated slightly toward reducing this loss
Learning rate: 5e-4 (small steps to avoid overshooting)
```

### Result
```
After this one step:
- Model now slightly more confident phishing = 0.79 (was random before)
- Embeddings pushed slightly apart (phishing cluster vs benign)
- Weights adjusted to reduce loss on THIS sample
```

### Next Sample
```
Process repeats for 842 more training samples in epoch
Average of all losses = epoch loss ≈ 0.56
```

---

## 9. Why This Architecture Works for Phishing Detection

| Component | Why It Matters |
|-----------|----------------|
| **Heterogeneous Graph** | Different edge types have different meanings (sends ≠ calls ≠ approves) |
| **Multi-Head Attention** | Different "heads" learn different phishing patterns simultaneously |
| **Temporal Gating** | Phishers often have suspicious timing (exploit windows, specific hours) |
| **2 Message Passing Layers** | 1 layer = direct neighbors, 2 layers = indirect patterns (mules, funnels) |
| **Contrastive Loss** | Pulls phishing embeddings away from benign in latent space |
| **80-20 CE/Contrastive** | Forces both discriminative power AND robust feature space |

---

## 10. Loss During Training (Your Model)

```
Epoch 1:   loss = 0.5881  (random initialization, high uncertainty)
Epoch 10:  loss = 0.6085  (learning patterns, loss stable)
Epoch 30:  loss = 0.5784  (good convergence)
Epoch 50:  loss = 0.5605  (final, very confident predictions)

Final on validation: AUC = 0.8426 → Best model saved
Final on test:       loss = 0.7083 (generalization loss)
```

---

## Summary

**One Training Step:**
1. Graph (local transactions) → HT-GNN (2 layers with attention + temporal)
2. Extract target embedding (128-dim)
3. Classify (2 logits) → softmax probability
4. Compute loss (80% CE + 20% contrastive)
5. Backpropagate gradients
6. Update all weights toward reducing loss
7. Repeat for 843 samples/epoch, 50 epochs

**Result:** Model learns to embed phishing addresses far from benign in latent space, enabling fast, accurate inference.
