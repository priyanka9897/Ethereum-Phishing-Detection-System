# ChainGuard: Complete Project Workflow

## Overview
**ChainGuard** is an Ethereum phishing detection system using a Heterogeneous Temporal Graph Neural Network (HT-GNN). It analyzes on-chain transaction graphs to identify malicious addresses with 79% accuracy.

---

## 1. Data Collection Pipeline

### 1.1 Raw Data Sources
- **Etherscan API**: Transaction data, contract interactions
- **Alchemy API**: Transfer events, approval logs
- **Labels**: CSV dataset with phishing/benign flags

### 1.2 Data Fetching (`Backend/pipeline/fetcher.py`)
```
For each address:
  1. Fetch normal transactions (send/receive)
  2. Fetch token transfers (ERC-20)
  3. Fetch contract approvals
  4. Fetch balance history
  5. Collect full subgraph (address + neighbors)
```

**Methods in EthereumFetcher:**
- `get_balance_eth()` → Account balance in ETH
- `get_normal_txs()` → EOA-to-EOA transfers
- `get_token_transfers()` → ERC-20 token movements
- `get_erc20_approvals()` → Approval transactions
- `collect_full_subgraph_data()` → Complete transaction subgraph

**Cache Strategy:**
- All raw data cached in `./data/graphs/0xABCD1234.pt`
- Contains 1,947 pre-fetched addresses
- Enables fast training without API rate limits

---

## 2. Graph Construction (`Backend/pipeline/graph_builder.py`)

### 2.1 Graph Structure
Convert raw transaction data into heterogeneous graphs:

**Node Types:**
- `eoa`: Externally Owned Accounts (10-dim features)
- `contract`: Smart Contracts (8-dim features)

**Edge Types:**
- `eoa → sends → eoa`: Direct ETH transfers
- `eoa → calls → contract`: Function calls
- `contract → returns → eoa`: Contract returns
- `eoa → approves → contract`: Token approvals

### 2.2 Node Features (per address)
```
[balance_normalized, freq_in, freq_out, 
 avg_value_in, avg_value_out, 
 block_age, contract_created, unique_partners,
 token_transfers, approval_count]
```

### 2.3 Edge Features (per transaction)
```
[timestamp, value_normalized, 
 time2vec_encoded(4-dims)]
```

### 2.4 Time2Vec Encoding
Temporal encoding using sine/cosine:
```
t'_sin = sin(ω·t + φ)
t'_cos = cos(ω·t + φ)
```
Captures periodic patterns (blocks, hours, days)

---

## 3. Dataset Preparation

### 3.1 Label Distribution Challenge
```
Original Dataset (9,841 addresses):
  Phishing:  2,179 (22.1%)
  Benign:    7,662 (77.9%)
  → Imbalanced → Model bias to benign class
```

### 3.2 Balancing Strategy
```
Step 1: Identify cached addresses (addresses with pre-fetched graphs)
  → 1,000 total (500 phishing, 500 benign)

Step 2: Balance by sampling equally
  → Take min(500,500) = 5500 from each
  → Create 1,000-row dataset (perfect 50-50 split)

Step 3: Train/Val/Test Split (80-10-10 stratified)
  → Train:      799 samples
  → Validation: 100 samples
  → Test:       100 samples
```

### 3.3 Dataset Files
- `transaction_dataset.csv` – Original 9,841 rows
- `transaction_dataset_cached_balanced_1000.csv` – Final 1,000-row balanced (used for training)
- `transaction_dataset_cached_balanced_2000.csv` – Attempted 1,056-row (cache-limited)

---

## 4. Model Architecture (`Backend/model/htgnn.py`)

### 4.1 HT-GNN (Heterogeneous Temporal Graph Neural Network)

```
Input: HeteroData graph (heterogeneous transaction subgraph)
       ↓
[Layer 1] Projection
  - EOA: 10 dims → 128 hidden dims
  - Contract: 8 dims → 128 hidden dims
       ↓
[Layers 2-3] Message Passing (HGT Convolution)
  For each layer:
    1. HGTConv: Multi-head heterogeneous attention
    2. TemporalAttention: Gate network with time encoding
    3. Residual connection + LayerNorm + Dropout
       ↓
[Classification Head]
  1. Extract target address embedding (EOA node marked as target)
  2. FC: 128 → 64 dims (ReLU)
  3. FC: 64 → 2 dims (logits for [benign, phishing])
  4. Softmax → probability p(phishing)
       ↓
Output: P(phishing) ∈ [0,1]
```

### 4.2 Key Components

**TemporalAttention Module:**
```python
t_emb = W_time(edge_time_features)       # (N, hidden)
gate = sigmoid(W_gate([h, t_emb]))       # (N, hidden)
h_new = gate * h + (1-gate) * t_emb      # Gated fusion
```

**HGTConv (PyTorch Geometric):**
```python
Attention-based message passing:
  message = Attention(Q=h_i, K=h_j, V=h_j, E=edge_type)
  h_i^new = Aggregate(messages from neighbors)
```

---

## 5. Training Pipeline (`Backend/model/train.py`)

### 5.1 Loss Function (Combined)

```
L_total = 0.8 * L_CE + 0.2 * L_contrastive

L_CE = -1/N Σ[y_i·log(p_i) + (1-y_i)·log(1-p_i)]
       ↑
       Standard Binary Cross-Entropy

L_contrastive = Average pairwise distances
       ↑
       Push phishing embeddings away from benign
```

### 5.2 Training Configuration (Tuned)
```
Hyperparameters:
  hidden_dim:  128        (model capacity)
  num_heads:   8          (attention heads)
  num_layers:  2          (message-passing layers)
  dropout:     0.3        (regularization)
  learning_rate: 5e-4     (AdamW with weight decay 1e-4)
  scheduler:   CosineAnnealing (T_max=50 epochs, eta_min=1e-5)
  
Epochs:      50           (best convergence)
Batch size:  1            (per-sample gradient updates)
```

### 5.3 Training Loop Pseudocode
```
for epoch in 1..50:
  for each sample (graph, label) in train_set:
    1. Forward pass: graph → model → logits + embeddings
    2. Compute CE loss:   F.cross_entropy(logits, label)
    3. Compute contrastive loss: distance-based regularization
    4. Combine: total_loss = 0.8*ce_loss + 0.2*contra_loss
    5. Backward: compute gradients
    6. Clip gradients: ||∇|| ≤ 1.0 (stability)
    7. Optimizer step: update weights with AdamW
  
  Evaluate on validation_set:
    - Compute AUC, F1, loss
    - Save model if AUC > best_auc
  
  Scheduler step: reduce learning rate (cosine annealing)
```

### 5.4 Final Metrics (Test Set, 106 samples)
```
Accuracy:         79.0% (Error: 21%)
AUC:              0.8208
F1 Score:         0.7789

Per-Class Performance:
  Benign    (53 samples):
    Precision: 0.76
    Recall:    0.84  ✓ Low false negatives (catch benign)
  
  Phishing  (53 samples):
    Precision: 0.82
    Recall:    0.74  ✓ Good detection rate

Binary Cross-Entropy Loss (averages):
  Train:  0.7429
  Val:    1.0275  (slight overfitting signal)
  Test:   0.7083  ← Generalization loss
```

---

## 6. Inference Engine (`Backend/model/inference.py`)

### 6.1 Inference Pipeline
```
Input: Ethereum address (string)
       ↓
Step 1: Check if cached graph exists
  - Yes → Load from disk (fast)
  - No  → Fetch on-chain data (slow, ~18s)
       ↓
Step 2: Run HT-GNN forward pass
  graph → model → phishing_probability
       ↓
Step 3: Extract heuristic signals
  - Balance age
  - Transaction frequency
  - Interaction diversity
  - Contract participation
       ↓
Step 4: Blend scores (70% model, 30% heuristics)
  final_score = 0.7 * model_prob + 0.3 * heuristic_score
       ↓
Step 5: Assign risk tier
  < 0.3: Benign
  0.3-0.6: Low Risk
  0.6-0.85: Medium Risk
  > 0.85: High Risk (Phishing)
       ↓
Output: PhishingResult
  {
    score: 0.0-1.0,
    verdict: "benign" | "medium_risk" | "phishing",
    risk_tier: 1-4,
    evidence: [list of indicators],
    feature_importance: {model_score, heuristic_score}
  }
```

### 6.2 Heuristic Features
```
Signals that detect phishing without ML:
  1. Fresh account (created < 7 days ago)
  2. High transaction frequency (> 100/day)
  3. Multiple interactions with flagged contracts
  4. Unusual fund movement patterns
  5. Approval of unknown tokens
```

---

## 7. API Backend (`Backend/main.py`)

### 7.1 Endpoints

```
GET /health
  → Returns: {"status": "ok", "model": "./model/htgnn_phishing.pt"}
  
POST /api/v1/analyze
  Body: {"address": "0x1234..."}
  Returns: PhishingResult (score, verdict, evidence)
  
GET /api/v1/marketplace
  Returns: Global marketplace data (top phishing patterns)
  
POST /api/v1/fetch-and-save
  Body: {"address": "0x1234..."}
  Side effect: Fetches + caches graph for address
  
POST /api/v1/fetch-raw
  Body: {"address": "0x1234..."}
  Returns: Raw subgraph data (for debugging)
```

### 7.2 Technology Stack
```
Framework:   FastAPI (Python async web framework)
Model:       PyTorch + PyTorch Geometric
Cache:       Redis (with in-memory fallback)
API Sources: Etherscan, Alchemy
Port:        8000
```

---

## 8. Frontend Applications

### 8.1 Static HTML Dashboard (`Frontend/index.html`)
```
URL: http://localhost:3000

Features:
  1. Address input field
  2. Risk score visualization (gauge chart)
  3. Traffic light indicator (green/yellow/red)
  4. Evidence panel (heuristic signals)
  5. Marketplace section (trending patterns)
  
Calls: POST http://localhost:8000/api/v1/analyze
```

### 8.2 Streamlit Dashboard (`Backend/dashboard_app.py`)
```
URL: http://localhost:8501

Features:
  1. Interactive address lookup
  2. Detailed model explainability
  3. Historical analysis graphs
  4. Model performance metrics
  5. Real-time risk feed
  
Technology: Streamlit (Python)
Calls: Direct model inference (no API)
```

---

## 9. Complete End-to-End Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                           │
│            (Ethereum address to check)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │ Check Graph Cache     │
         └───┬─────────────┬─────┘
             │ Hit         │ Miss
             ↓             ↓
        (Load from     (Fetch from API)
         disk - fast)   (slow ~18s)
             │             │
             └──────┬──────┘
                    ↓
      ┌─────────────────────────┐
      │  HeteroData Graph Ready │
      └────────────┬────────────┘
                   ↓
      ┌─────────────────────────┐
      │   HT-GNN Inference      │
      │  (Forward pass on GPU)  │
      │    → Phishing Score     │
      └────────────┬────────────┘
                   ↓
      ┌─────────────────────────┐
      │  Extract Heuristics     │
      │  (Balance, frequency,   │
      │   interaction patterns) │
      └────────────┬────────────┘
                   ↓
      ┌─────────────────────────┐
      │  Blend Scores           │
      │  70% model + 30% rules  │
      └────────────┬────────────┘
                   ↓
      ┌─────────────────────────┐
      │  Assign Risk Tier       │
      │  & Generate Evidence    │
      └────────────┬────────────┘
                   ↓
      ┌─────────────────────────┐
      │  Return PhishingResult  │
      │  JSON Response         │
      └────────────┬────────────┘
                   ↓
         ┌─────────────────┐
         │ Render Frontend │
         └─────────────────┘
```

---

## 10. Training Workflow (Development)

### 10.1 Create Balanced Dataset
```bash
# Identify which addresses have cached graphs
python3 << EOF
import csv, os, random
# Filter transaction_dataset.csv to only cached addresses
# Balance to 500 phishing + 500 benign = 1000 total
EOF
```

### 10.2 Train Model
```bash
source venv/bin/activate
cd Backend
python model/train.py \
  --dataset ./data/transaction_dataset_cached_balanced_1000.csv \
  --epochs 50 \
  --skip-fetch \
  --hidden 128 \
  --heads 8 \
  --layers 2
```

### 10.3 Evaluate
```
Training produces:
  ✓ Best model checkpoint → ./model/htgnn_phishing.pt
  ✓ Test metrics (AUC, F1, confusion matrix)
  ✓ Per-class recall, precision, support
  ✓ Classification report
```

### 10.4 Loss Calculation (Per Epoch)
```
For each sample in training batch:
  1. Forward pass → model output probability p_i
  2. Target label y_i from CSV
  3. CE component: -(y_i·log(p_i) + (1-y_i)·log(1-p_i))
  4. Contrastive component: ||embedding_phishing - embedding_benign||
  5. Combined: 0.8*CE + 0.2*contrastive
  6. Average over all N samples in epoch

Result: training loss reported each epoch
Best model selected based on validation AUC
```

---

## 11. Deployment Architecture

```
┌─────────────────────────────────────────┐
│         End User (Browser)              │
│  http://localhost:3000 (Frontend)       │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│      FastAPI Backend (Port 8000)        │
│                                         │
│  - /api/v1/analyze endpoint             │
│  - Model loading + inference            │
│  - Heuristic scoring                    │
│  - Redis cache layer                    │
└────────────┬────────────────────────────┘
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
  ┌──────┐ ┌──────┐ ┌──────────┐
  │Cached │ │Graph │ │Etherscan/│
  │Model  │ │Data  │ │Alchemy   │
  │(GPU)  │ │(Disk)│ │APIs      │
  └──────┘ └──────┘ └──────────┘
```

---

## 12. Key Metrics & Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Test Accuracy | 79.0% | Model correct on 4/5 predictions |
| Test AUC | 0.8208 | Good discrimination between classes |
| Benign Recall | 84% | Catches benign addresses reliably |
| Phishing Recall | 74% | Detects 3/4 of actual phishing |
| Test BCE Loss | 0.7083 | Per-sample log-loss |
| Inference Speed | <200ms (cached) | Real-time suitable |
| Model Size | ~5 MB | Deployable |

---

## 13. Known Limitations & Future Work

### Current Limitations
1. **Data bias**: Only Ethereum data (no cross-chain)
2. **Temporal**: Snapshots don't capture time dynamics fully
3. **Label quality**: Phishing labels from Etherscan may have false positives
4. **Cold start**: New addresses without transaction history hard to classify

### Future Improvements
1. Multi-chain support (Polygon, Arbitrum, Optimism)
2. Temporal graph attention (TGN, DyRep)
3. Active learning on uncertain predictions
4. Real-time transaction monitoring (streaming)
5. Fine-tuning on domain-specific patterns (MEV, sandwich attacks)

---

## 14. Quick Start (Developer)

```bash
# 1. Activate environment
source venv/bin/activate
cd /path/to/Backend

# 2. Start backend (API)
python main.py
# Listens on http://localhost:8000

# 3. Start frontend (in another terminal)
cd ../Frontend
python3 -m http.server 3000
# Browse http://localhost:3000

# 4. Analyze an address
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"address": "0x1234567890abcdef"}'

# 5. View Streamlit dashboard (optional)
streamlit run dashboard_app.py
# Browse http://localhost:8501
```

---

## Summary

**ChainGuard** is a full-stack phishing detection platform that:
1. **Fetches** on-chain transaction data
2. **Builds** heterogeneous temporal graphs
3. **Trains** HT-GNN with balanced data & combined loss
4. **Infers** via FastAPI + hybrid ML+heuristics
5. **Visualizes** via HTML/Streamlit dashboards

**Target**: 79% accuracy, <200ms latency, production-ready.
