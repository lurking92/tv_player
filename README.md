# Knowledge Graph for Elderly Safety Monitoring System

**Important:** Use **Git LFS (Large File Storage)** to manage large model files (e.g., K-BERT model weights). To successfully download all files and perform training or prediction, ensure your environment is configured with Git LFS before cloning the repository.

This is a neuro-symbolic AI system combining **Visual Perception** and **Knowledge Reasoning**. It analyzes home surveillance videos to detect potential dangerous behaviors in the elderly (such as falling, wandering, climbing, or running) in real-time.

---

## ☁️ System Architecture and GCP Configuration

This project is deployed on Google Cloud Platform (GCP) and relies on the following services:

- **Google Cloud Storage (GCS):** Stores raw videos and analysis results.
- **Vertex AI (Custom Jobs):** Executes GPU-accelerated AI model computations.
- **Cloud Functions (Gen 2):** Acts as event triggers to orchestrate the automated pipeline.
- **Artifact Registry:** Stores Docker images.
- **Pub/Sub:** Handles asynchronous message delivery.
- **Secret Manager:** Securely manages service accounts and API keys.

### 1\. GCS Buckets

You need to create two buckets:

- **Raw Video Bucket:** `kg_tv-subtitle-videos-20251016` (Upload videos here)
- **Analysis Results Bucket:** `kg_tv-subtitle-videos-20251016-results` (Stores .ttl, .json, .zip)

### 2\. Pub/Sub Topics

- `gcs-video-uploads`: Listens for raw video upload events.
- `kbert-job-trigger-topic`: Listens for `.ttl` creation events in the results bucket.

### 3\. Secret Manager

Create the following secrets to protect sensitive information:

- `vertex-sa-email`: Your Vertex AI Service Account Email (e.g., `...-compute@developer.gserviceaccount.com`).
- `gemini-api-key`: Gemini API Key used for frontend real-time analysis.

---

## 🚀 Deployment Guide

### A. Job 1: Visual Analysis Pipeline

Responsible for executing DINO/BLIP and generating the knowledge graph.

1.  **Build Image:**
    ```bash
    cd docker_gpu_code
    gcloud builds submit --region=asia-east1 --tag asia-east1-docker.pkg.dev/[PROJECT_ID]/kg-pipeline-repo/kg-pipeline-gpu:latest .
    ```
2.  **Deploy Trigger (Cloud Function):**
    ```bash
    cd vertex_trigger_function
    gcloud functions deploy trigger-vertex-job --gen2 \
      --region=asia-east1 --runtime=python310 \
      --source=./ --entry-point=trigger_vertex_job \
      --trigger-topic=gcs-video-uploads --timeout=540s
    ```

### B. Job 2: K-BERT Risk Inference Pipeline

Responsible for reading TTL files, converting them to action sequences, and predicting risks.

1.  **Build Image:**
    ```bash
    cd k-bert_training
    gcloud builds submit --region=asia-east1 --tag asia-east1-docker.pkg.dev/[PROJECT_ID]/kg-pipeline-repo/kg-pipeline-kbert:latest .
    ```
2.  **Deploy Trigger (Cloud Function):**
    _Note: Must set `--memory=512MiB` to avoid OOM errors._
    ```bash
    cd vertex_trigger_function_job2
    gcloud functions deploy trigger-kbert-job-2 --gen2 \
      --region=asia-east1 --runtime=python310 \
      --source=./ --entry-point=trigger_kbert_job \
      --trigger-topic=kbert-job-trigger-topic \
      --timeout=540s --memory=512MiB
    ```

### C. Frontend Server (Web UI)

Provides the user interface for file uploads and result display.

1.  **Install Dependencies:**
    ```bash
    cd tv_server_full
    npm install
    ```
2.  **Start Service:**
    _This command automatically executes `build_manifest.js` to update risk data for built-in videos._
    ```bash
    npm start
    ```
3.  **Access:** Open a browser and go to `http://localhost:3000`.

---

## 🧠 K-BERT Dangerous Event Detection

This section details the training and usage of the risk detection model. It is capable of simultaneously detecting the following four high-risk events:

1.  **Fall**
2.  **Walk_with_memory_loss**
3.  **Fall_with_climb**
4.  **Run_with_disorientation**

### Stage 1: Dataset Preparation

This stage converts raw action sequences into the JSONL training format required by K-BERT and performs multi-label annotation.

| File                      | Output                              | Notes                                                                                                     |
| :------------------------ | :---------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `1.1_labeling_tool.py`    | `action_sequences_with_labels.json` | Performs interactive labeling; outputs **4 independent 0/1 label lists** per sequence.                    |
| `2_dataset_generation.py` | `kbert_train_data.jsonl`            | Integrates labeling results with knowledge triplets. **The final `label` field is a 4-dimensional list.** |

**Steps:**

1.  **Labeling:** Run `python 1.1_labeling_tool.py`
2.  **Generate Dataset:** Run `python 2_dataset_generation.py`

---

### Stage 2: Model Training and Configuration

This stage uses `kbert_train_data.jsonl` to train the multi-label classification model.

| File                         | Configuration & Output                                                                             | Output Example                                                                                        |
| :--------------------------- | :------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| `kbert_custom_dataloader.py` | Converts labels to `torch.float`.                                                                  | No direct output; used by `3_kbert_event_detector.py`.                                                |
| `3_kbert_event_detector.py`  | Sets `NUM_LABELS = 4`; uses `BCEWithLogitsLoss`. Model outputs to `kbert_model_output_multilabel`. | **Training complete, average loss decreases:**<br>Epoch 1 Loss: 0.4402<br>...<br>Epoch 5 Loss: 0.0764 |

**Steps:**

1.  **Execute Training:** Run `python 3_kbert_event_detector.py`

---

### Stage 3: Model Evaluation (Multi-label Metrics)

This stage evaluates the model's performance on the validation set using metrics suitable for multi-label classification.

| File                  | Prediction / Metric Calculation                                                             | Output Results                                                                                                                                                                                                                                                 |
| :-------------------- | :------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `4_evaluate_model.py` | Uses **Sigmoid & 0.5 Threshold** for prediction; uses **Micro/Macro F1-Score** for metrics. | **Individual Label F1-Score:**<br>Fall: 1.0000<br>Walk_with_memory_loss: 1.0000<br>Fall_with_climb: 0.0000 **(Data Insufficiency Warning)**<br>Run_with_disorientation: 0.6667<br><br>**Overall Metrics:**<br>Micro F1-Score: 0.9908<br>Macro F1-Score: 0.6667 |

**Steps:**

1.  **Execute Evaluation:** Run `python 4_evaluate_model.py`

---

### Stage 4: Actual Prediction (Batch Detection)

This stage uses the trained multi-label model to perform batch risk detection on new JSON action sequences.

| File                | Logic                                                                                     | Output                                                                      |
| :------------------ | :---------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| `5_predict_risk.py` | Loads the model and outputs 4 probabilities and 0/1 judgments for input action sequences. | Outputs risk judgments and 4-dimensional probability vectors for each file. |

**Steps:**

1.  **Execute Prediction:** Run `python 5_predict_risk.py`

---

### Stage 5: Real-time Detection Pipeline Integration (TTL Conversion & K-BERT Prediction)

This stage converts the low-level TTL observations output by the visual pipeline into action sequence JSON for K-BERT input, and performs real-time risk detection.

| File                         | Logic                                                                                                                                                  | Output                                                                         |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| `ttl_to_json_converter_6.py` | **TTL Converter:** Reads TTL files, executes logic rules (temporal/keyword), and converts low-level observations into high-level action sequence JSON. | Outputs JSON action sequences (e.g., `*_inferred_actions.json`).               |
| `json_risk_detector_7.py`    | **Real-time Predictor:** Loads the trained model, reads a single JSON action sequence, and outputs 4 probabilities and 0/1 judgments.                  | Outputs risk judgment and 4-dimensional probability vectors for a single file. |

**Steps:**

1.  **Convert JSON:** Run `python ttl_to_json_converter_6.py`
2.  **Detect Risk:** Run `python json_risk_detector_7.py`
