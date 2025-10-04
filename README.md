# RefineRAG: Multimodal Noise Reduction for LLM-Based Question Answering

<img width="1397" height="366" alt="image" src="https://github.com/user-attachments/assets/f24591e0-5b12-4076-b1d0-2e35b1bc5044" />

<img width="1397" height="411" alt="image" src="https://github.com/user-attachments/assets/6fd237cd-e51f-4201-ae91-45f1a23fc0ef" />

A modular **retrieval-augmented generation (RAG)** pipeline that filters noisy multimodal retrievals (image + text) to improve LLM reasoning accuracy.

## Highlights
- **Boosted MultimodalQA accuracy by 63%** and F1 by 56% vs baseline.
- Built with **LLaVA, LlamaIndex, and Qdrant**, with plug-and-play document filtering agents.
- Demonstrates scalable **multimodal information retrieval** for LLM-based QA systems.

## Setup

Install Package
```Shell
virtualenv vllava
source vllava\bin\activate
pip install -e .
```

Download `WebQA_test.json` and `WebQA_train_val.json` from [here](https://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/)
and place them to `datasets/WebQA/annotations`.

## Usage

```Shell
python final_project.py --use_mmqa --use_km
```

## Results

### MultimodalQA 
| Approach | Retr. | Read | Text | Image | Overall |
| -- | -- | -- | -- | -- | -- |
| LLaVA-1.5 | - | - | 11.99 | 6.31 | 9.15 |
| LLaVA-1.5 + RAG | 22.81 | 22.81 | 31.67 | 21.95 | 26.81 |
| LLaVA-1.5 + RAG+RM | 22.81 | 16.26 | 23.24 | 44.63 | 33.94 |
| LLaVA-1.5 + RAG+IRM+KM | 22.81 | **40.51** | **33.92** | **50.13** | **42.02** |

### WebQA
| Approach | Retr. | Read | Text | YesNo | Number | Color | Shape | Overall |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LLaVA-1.5 | - | - | 18.22 | 33.49 | 26.37 | 23.10 | 11.71 | 22.58 |
| LLaVA-1.5 + RAG | 26.43 | 26.43 | 33.70 | 43.68 | **36.90** | 47.80 | 19.14 | 36.25 |
| LLaVA-1.5 + RAG+RM | 26.43 | 24.80 | 25.01 | 42.63 | 30.03 | **51.69** | **21.85** | 34.24 |
| LLaVA-1.5 + RAG+IRM+KM | 26.43 | **29.43** | **34.88** | **43.92** | 36.40 | 51.19 | 21.40 | **37.56** |
