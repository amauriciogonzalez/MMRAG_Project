from PIL import Image
import matplotlib.pyplot as plt
import os

from llama_index.core import SimpleDirectoryReader, StorageContext  # pip install llama-index-core
from llama_index.readers.web import SimpleWebPageReader # pip install llama-index-readers-web
import qdrant_client    # pip install qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore # pip install llama-index-vector-stores-qdrant llama-index-readers-file llama-index-embeddings-fastembed llama-index-llms-openai
#from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.indices import MultiModalVectorStoreIndex
                                                                # pip install llama-index-embeddings-clip
                                                                # pip install git+https://github.com/openai/CLIP.git
                                                                # pip install llama-index-embeddings-huggingface

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode

                                                                # pip install ipython

import shutil
import os
import requests

import json

from pprint import pprint



#image_documents = SimpleDirectoryReader(input_files=["./cars/o1.jpg","./cars/t1.jpg","./cars/v1.jpg"]).load_data()
#image_documents = SimpleWebPageReader().load_data(["https://llava-vl.github.io/static/images/view.jpg"])
#print(image_documents[0])


""""
# NOTE: the html_to_text=True option requires html2text to be installed
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)
documents[0]
index = SummaryIndex.from_documents(documents)
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""


# https://medium.aiplanet.com/advanced-rag-using-llama-index-e06b00dc0ed8
""""
dict(nodes[10]) # since no indexing performed so embeding is None

#### RESPONSE
{'id_': '8418b939-dc08-42a6-8ee1-821e46f7a2a1',
 'embedding': None,
 'metadata': {'window': 'Many big companies are adopting recognition systems for their security and authentication\npurposes.\n Use Cases of Recognition Systems\nFace recognition systems are widely used in the modern era, and many new innovative systems are built on\ntop of recognition systems.\n There are a few used cases :\nFinding Missing Person\nIdentifying accounts on social media\nRecognizing Drivers in Cars\nSchool Attendance System\nSeveral methods and algorithms implement facial recognition systems depending on the performance and\naccuracy.\n Traditional Face Recognition Algorithm\nTraditional face recognition algorithms don’t meet modern-day’s facial recognition standards.  They were\ndesigned to recognize faces using old conventional algorithms.\n OpenCV provides some traditional facial Recognition Algorithms.\n',
  'original_text': 'Traditional Face Recognition Algorithm\nTraditional face recognition algorithms don’t meet modern-day’s facial recognition standards. ',
  'page_label': '1',
  'file_name': 'face-recognition-system-using-python.pdf',
  'file_path': 'Data/face-recognition-system-using-python.pdf',
  'file_type': 'application/pdf',
  'file_size': 465666,
  'creation_date': '2023-12-21',
  'last_modified_date': '2023-12-21',
  'last_accessed_date': '2023-12-21'},
 'excluded_embed_metadata_keys': ['file_name',
  'file_type',
  'file_size',
  'creation_date',
  'last_modified_date',
  'last_accessed_date',
  'window',
  'original_text'],
 'excluded_llm_metadata_keys': ['file_name',
  'file_type',
  'file_size',
  'creation_date',
  'last_modified_date',
  'last_accessed_date',
  'window',
  'original_text'],
 'relationships': {<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='4538122a-90cd-4af5-b6e9-84b8399da135', node_type=<ObjectType.DOCUMENT: '4'>, metadata={'page_label': '1', 'file_name': 'face-recognition-system-using-python.pdf', 'file_path': 'Data/face-recognition-system-using-python.pdf', 'file_type': 'application/pdf', 'file_size': 465666, 'creation_date': '2023-12-21', 'last_modified_date': '2023-12-21', 'last_accessed_date': '2023-12-21'}, hash='a819d446ca212183fcf6284e3cc010747bf04a25aa39040bff8877fe5e35734d'),
  <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='942980f2-6328-45a8-94ce-2afe8fba03ba', node_type=<ObjectType.TEXT: '1'>, metadata={'window': 'Many applications can be built on top of recognition\nsystems.  Many big companies are adopting recognition systems for their security and authentication\npurposes.\n Use Cases of Recognition Systems\nFace recognition systems are widely used in the modern era, and many new innovative systems are built on\ntop of recognition systems.\n There are a few used cases :\nFinding Missing Person\nIdentifying accounts on social media\nRecognizing Drivers in Cars\nSchool Attendance System\nSeveral methods and algorithms implement facial recognition systems depending on the performance and\naccuracy.\n Traditional Face Recognition Algorithm\nTraditional face recognition algorithms don’t meet modern-day’s facial recognition standards.  They were\ndesigned to recognize faces using old conventional algorithms.\n', 'original_text': 'There are a few used cases :\nFinding Missing Person\nIdentifying accounts on social media\nRecognizing Drivers in Cars\nSchool Attendance System\nSeveral methods and algorithms implement facial recognition systems depending on the performance and\naccuracy.\n', 'page_label': '1', 'file_name': 'face-recognition-system-using-python.pdf', 'file_path': 'Data/face-recognition-system-using-python.pdf', 'file_type': 'application/pdf', 'file_size': 465666, 'creation_date': '2023-12-21', 'last_modified_date': '2023-12-21', 'last_accessed_date': '2023-12-21'}, hash='c7993150674462ffbf83c9cfbb786980c1f1d5ea27b2313954b4901f55448f59'),
  <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='18f77456-c93d-478d-a162-e4d01ad8135b', node_type=<ObjectType.TEXT: '1'>, metadata={'window': 'Use Cases of Recognition Systems\nFace recognition systems are widely used in the modern era, and many new innovative systems are built on\ntop of recognition systems.\n There are a few used cases :\nFinding Missing Person\nIdentifying accounts on social media\nRecognizing Drivers in Cars\nSchool Attendance System\nSeveral methods and algorithms implement facial recognition systems depending on the performance and\naccuracy.\n Traditional Face Recognition Algorithm\nTraditional face recognition algorithms don’t meet modern-day’s facial recognition standards.  They were\ndesigned to recognize faces using old conventional algorithms.\n OpenCV provides some traditional facial Recognition Algorithms.\n Eigenfaces\nScale Invariant Feature Transform (SIFT)\nFisher faces\nLocal Binary Patterns Histograms (LBPH)\nCOMPUTER VISION\nIMAGE ANALYSIS\nINTERMEDIATE\nPYTHON', 'original_text': 'They were\ndesigned to recognize faces using old conventional algorithms.\n'}, hash='4c57a3fcceaebf806622383637926ea4e27e153542a4d2ce7a4a82d3df8a72de')},
 'hash': '5f83425f868962e1066c20252056676e299fc51a8af0dbab4bb3c8bcc9130e2f',
 'text': 'Traditional Face Recognition Algorithm\nTraditional face recognition algorithms don’t meet modern-day’s facial recognition standards. ',
 'start_char_idx': 1166,
 'end_char_idx': 1299,
 'text_template': '{metadata_str}\n\n{content}',
 'metadata_template': '{key}: {value}',
 'metadata_seperator': '\n'}
"""

def demo():

    # Specify the path to the folder you want to delete
    database_path = './qdrant_database'
    qdrant_path = database_path + "/qdrant_mm_db"

    # Check if the folder exists before attempting to delete it
    if os.path.exists(database_path):
        # Delete the folder and its contents
        shutil.rmtree(database_path)
        os.makedirs(database_path, exist_ok=True)


    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path=qdrant_path)

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

    # Create the MultiModal index
    documents = SimpleDirectoryReader("./testing_data").load_data()

    #documents = SimpleWebPageReader().load_data(["https://llava-vl.github.io/static/images/view.jpg"])

    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model="local",
    )

    # generate Text retrieval results
    retriever_engine = index.as_retriever(
        similarity_top_k=3, image_similarity_top_k=3
    )
    retrieval_results = retriever_engine.retrieve("Black Image")

    print("")

    retrieved_image = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            #res_node.node.metadata["file_path"]   <-- File path of the image
            print("An image node:")
            print(res_node)
        else:
            #display_source_node(res_node, source_length=200)
            print("Not an image node:")
            print(res_node)




# Function to save text snippets as .txt files
def save_text_snippets(text_facts, folder_path):
    for fact in text_facts:
        # Extract relevant information
        snippet_id = fact['snippet_id']
        content = f"Title: {fact['title']}. Fact: {fact['fact']}"
        content = content.replace('\ufeff', '')
        # Save text snippet as .txt file
        #file_path = os.path.join(folder_path, f"{snippet_id}.txt")
        file_path = f"{folder_path}/{snippet_id}.txt"
        with open(file_path, 'w', encoding='utf-8') as file:
            #print("Saving", file_path)
            file.write(content)

# Function to download images from URLs
def download_images(image_facts, folder_path):
    for fact in image_facts:
        # Extract relevant information
        image_url = fact['imgUrl']
        image_id = fact['image_id']
        # Send HTTP request to download the image
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        response = requests.get(image_url, stream=True, headers=headers)
        if response.status_code == 200:
            # Save the image
            file_path = os.path.join(folder_path, f"{image_id}.jpg")
            with open(file_path, 'wb') as file:
                file.write(response.content)
        else:
            print(f"--- ERROR: UNABLE TO FETCH IMAGE {image_id}, RESPONSE STATUS CODE: {response.status_code}, URL: {image_url} ---")
            


def index_store_retrieve_webqa(sample_guid, query, data_folder_path, text_similarity_top_k=3, image_similarity_top_k=3):
    database_path = './qdrant_database'
    #qdrant_path = database_path + f"/qdrant_mm_db-{index}"
    qdrant_path = database_path + f"/qdrant_mm_db-{sample_guid}"

    # Check if the folder exists before attempting to delete it
    if os.path.exists(qdrant_path):
        # Delete the folder and its contents
        shutil.rmtree(qdrant_path)
        os.makedirs(qdrant_path, exist_ok=True)

    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path=qdrant_path)

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

    # LOADING
    documents = SimpleDirectoryReader(data_folder_path).load_data()

    #documents = SimpleWebPageReader().load_data(["https://llava-vl.github.io/static/images/view.jpg"])

    # INDEXING
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model="local",
    )

    # RETRIEVING
    retriever_engine = index.as_retriever(
        similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k
    )

    # CLIP context length maximum: 77 tokens
    # env/lib/sitepackages/clip/clip.py
    # truncate: bool = True
    print('query to retrieve from:', query)
    retrieval_results = retriever_engine.retrieve(query)

    return retrieval_results


def display_retrieved_results(retrieval_results, positive_ids, sample_data):
    retrieved_image_path = ""
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path = res_node.node.metadata.get("file_path", "")
            image_id = os.path.splitext(os.path.basename(retrieved_image_path))[0]
            print("image_id:", image_id)
            source_set = "Positive" if image_id in positive_ids else "Negative"
            print(f"An image node from {source_set} set:")
            print("Image Name:", retrieved_image_path.split("\\")[-1])

            # Get the caption of the image with the corresponding image_id
            image_caption = ""
            for img_info in sample_data.get('img_posFacts' if image_id in positive_ids else 'img_negFacts', []):
                if str(img_info.get('image_id')) == image_id:
                    image_caption = img_info.get('caption', "")
                    break

            res_node.node.text = image_caption
            print(res_node)
        else:
            retrieved_text_path = res_node.node.metadata.get("file_path", "")
            text_id = os.path.splitext(os.path.basename(retrieved_text_path))[0]
            print("text_id:", text_id)
            source_set = "Positive" if text_id in positive_ids else "Negative"
            print(f"A text node from {source_set} set:")
            print(res_node)

    if not retrieved_image_path:
        #print("BLACK IMAGE RETRIEVED")
        retrieved_image_path = "./black_image.png"
    
    return retrieved_image_path


def download_dataset(download_data, example_retrieval, clean_data=False, text_similarity_top_k=3, image_similarity_top_k=1):

    data_folder_path = "./source_data"

    # Create the folder if it doesn't exist
    os.makedirs(data_folder_path, exist_ok=True)

    index = 0
    # Iterate over each sample in the dataset
    for sample_guid, sample_data in dataset.items():

        # Extract text and image facts
        txt_posFacts = sample_data['txt_posFacts']
        txt_negFacts = sample_data['txt_negFacts']
        img_posFacts = sample_data['img_posFacts']
        img_negFacts = sample_data['img_negFacts']

        [dataset[k]['split'] for k in dataset]

        positive_ids = [str(sample['snippet_id']) for sample in txt_posFacts] + [str(sample['image_id']) for sample in img_posFacts]
        
        #print("\n\n")
        #pprint(sample_data)
        #print("\n\n")

        #sample_data_folder_path = f"/data-{index}"      # CONSIDER CHANGING THIS FROM INDEX TO GUID FOR VARIABLE DATSET SIZES
        sample_data_folder_path = f"/data-{sample_guid}" 
        sample_data_folder_path = data_folder_path + sample_data_folder_path
        os.makedirs(sample_data_folder_path, exist_ok=True)

        if download_data:
            # Save text snippets as .txt files
            save_text_snippets(txt_posFacts + txt_negFacts, sample_data_folder_path)
            
            # Download images from URLs
            download_images(img_posFacts + img_negFacts, sample_data_folder_path)

        if example_retrieval:
            query = sample_data['Q']

            print(f"\n{index}")
            print("\nPositive ids:", positive_ids)
            print(f'\nQuestion: {query}\n')

            retrieval_results = index_store_retrieve_webqa(sample_guid=sample_guid, query=query, data_folder_path=sample_data_folder_path, text_similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
            
            display_retrieved_results(retrieval_results, positive_ids, sample_data)

            print("=" * 25)


        # Delete the folder and its contents
        if clean_data:
            if os.path.exists(data_folder_path):
                shutil.rmtree(data_folder_path)
                os.makedirs(data_folder_path, exist_ok=True)
        
        index += 1


# === Practical RAG implementation ===

# Output positive id's separately
# Output retrieved nodes

def set_webqa_node_captions(retrieval_results, sample_data, positive_ids):
    retrieved_ids = []
    # Inserts caption as the text portion of the image node and obtains retrieved ids for positive id comparison later
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path = res_node.node.metadata.get("file_path", "")
            image_id = os.path.splitext(os.path.basename(retrieved_image_path))[0]
            image_caption = ""
            for img_info in sample_data.get('img_posFacts' if image_id in positive_ids else 'img_negFacts', []):
                if str(img_info.get('image_id')) == image_id:
                    image_caption = img_info.get('caption', "")
                    break
            res_node.node.text = image_caption
            retrieved_ids.append(str(image_id))
        else:
            retrieved_text_path = res_node.node.metadata.get("file_path", "")
            text_id = os.path.splitext(os.path.basename(retrieved_text_path))[0]
            retrieved_ids.append(str(text_id))

    return retrieval_results, retrieved_ids

def construct_augmented_query(retrieval_results, query):
    obtained_facts = 1
    augmented_query = "Consider the following facts: "
    retrieved_image_path = ""       # ONLY EXPECTS ONE RETRIEVED IMAGE.
    image_caption = ''
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path = res_node.node.metadata.get("file_path", "")
            image_caption = res_node.node.text
        else:
            augmented_query += f'\n\n{obtained_facts}. {res_node.node.text}'
            obtained_facts += 1
    
    if retrieved_image_path:
        augmented_query += f'\n\nAlso, the image has the following caption: {image_caption}.'
    else:
        augmented_query += f'\n\nIgnore the image.'
    
    augmented_query += f"\n\nAnswer the following question in a complete sentence: {query}"

    if not retrieved_image_path:
        #print("BLACK IMAGE RETRIEVED")
        retrieved_image_path = "./black_image.png"
    
    return augmented_query, retrieved_image_path

def calculate_retrieved_id_metrics(positive_ids, retrieved_ids):
    """
    Calculate precision, recall, and F1 score based on positive IDs and retrieved IDs.

    Args:
    positive_ids (list): List of positive IDs.
    retrieved_ids (list): List of retrieved IDs.

    Returns:
    tuple: A tuple containing precision, recall, and F1 score.
    """
    # Calculate true positives (intersection of positive_ids and retrieved_ids)
    true_positives = len(set(positive_ids) & set(retrieved_ids))

    # Calculate false positives (IDs in retrieved_ids but not in positive_ids)
    false_positives = len(set(retrieved_ids) - set(positive_ids))

    # Calculate false negatives (IDs in positive_ids but not in retrieved_ids)
    false_negatives = len(set(positive_ids) - set(retrieved_ids))

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score

def simulated_webqa_prcoess(dataset, data_folder_path="./source_data", text_similarity_top_k=3, image_similarity_top_k=1):

    index = 0
    # Iterate over each sample in the dataset
    for sample_guid, sample_data in dataset.items():

        # Extract text and image facts
        txt_posFacts = sample_data['txt_posFacts']
        txt_negFacts = sample_data['txt_negFacts']
        img_posFacts = sample_data['img_posFacts']
        img_negFacts = sample_data['img_negFacts']

        positive_ids = [str(sample['snippet_id']) for sample in txt_posFacts] + [str(sample['image_id']) for sample in img_posFacts]
        
        sample_data_folder_path = f"/data-{sample_guid}" 
        sample_data_folder_path = data_folder_path + sample_data_folder_path

        query = sample_data['Q']

        retrieval_results = index_store_retrieve_webqa(sample_guid=sample_guid, query=query, data_folder_path=sample_data_folder_path, text_similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
        retrieval_results, retrieved_ids = set_webqa_node_captions(retrieval_results, sample_data, positive_ids)

        augmented_query, retrieved_image_path = construct_augmented_query(retrieval_results, query)

        precision, recall, f1_score = calculate_retrieved_id_metrics(positive_ids, retrieved_ids)


        print(f"\n{index}")
        print("\nPositive ids:", positive_ids)
        print("\nRetrieved ids:", retrieved_ids)
        print('\nPrecision:', precision)
        print('\nRecall:', recall)
        print('\nF1-Score:', f1_score)
        print('\nQuery:', query)
        print("\nAugmented Query:")
        print(augmented_query)
        print("\nRetrieved image file path:")
        print(retrieved_image_path)

        print("=" * 25)
        
        index += 1





# ====================================



#demo()


# n: Number of samples to use from WebQA (4966 max)
# example_retrieval: Displays query and retrieval results
# download_data: Downloads n datasamples to use
# clean_data: deletes data after execution. Good to use for example retrievals.

dataset = json.load(open("./datasets/WebQA/annotations/WebQA_train_val.json", "r"))

n = 10

# Ensures val selection
val_samples = {}
for key, value in dataset.items():
    if value['split'] == 'val':
        val_samples[key] = value
dataset = val_samples # len = 4966

#n = len(dataset)
#dataset = dataset[:n]
selected_samples = {}
count = 0
for key, value in dataset.items():
    selected_samples[key] = value
    count += 1
    if count >= n:
        break
dataset = selected_samples

#download_dataset(download_data=False, example_retrieval=True, clean_data=False, text_similarity_top_k=3, image_similarity_top_k=1)

simulated_webqa_prcoess(dataset, data_folder_path="./source_data", text_similarity_top_k=3, image_similarity_top_k=1)