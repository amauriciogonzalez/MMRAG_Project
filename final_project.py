from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse
import json
from PIL import Image
from collections import Counter
import numpy as np

from word2number import w2n
import string, re
from pprint import pprint
import shutil
import os
import requests
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# By default, LlamaIndex will use text-embedding-ada-002
#Settings.embed_model = OpenAIEmbedding()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


from sklearn.feature_extraction.text import TfidfVectorizer 
from collections import Counter
import spacy
import numpy as np
# pip install keyword-spacy
# python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")

from nltk.corpus import stopwords
import nltk


# ORIGINAL IMPLEMENTATION


def detectNum(l):
    result = []
    for w in l:
        try: result.append(str(int(w)))
        except: pass
    return result
def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word
def norm_text_to_nums(norm_text):
    nums = []
    for word in norm_text:
        if word == 'twice':
            nums.append('2')
        if word == 'no':
            nums.append('0')
        else:
            nums.append(toNum(word))
    return nums

"""
def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))
"""


# IDEAL IMPLEMENTATION

"""
def normalize_text(text):
    # Define functions for text normalization
    def remove_articles(text):
        # Remove articles (a, an, the) from the text
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def remove_punctuation(text):
        # Remove punctuation from the text
        return text.translate(str.maketrans("", "", string.punctuation))

    def lowercase(text):
        # Convert text to lowercase
        return text.lower()

    def lemmatize(text):
        # Lemmatize the text using spaCy
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    # Apply text normalization functions sequentially
    normalized_text = remove_articles(text)
    normalized_text = remove_punctuation(normalized_text)
    normalized_text = lowercase(normalized_text)
    normalized_text = lemmatize(normalized_text)

    return normalized_text
"""


def normalize_text(text):
    # Define functions for text normalization
    def remove_articles(text):
        # Remove articles (a, an, the) from the text
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def remove_punctuation(text):
        # Remove punctuation from the text
        return text.translate(str.maketrans("", "", string.punctuation))

    def lowercase(text):
        # Convert text to lowercase
        return text.lower()

    # Changes "has" to "ha"
    """
    def lemmatize(text):
        # Lemmatize the text using a simple method
        lemmatized_text = []
        for word in text.split():
            # You can add more sophisticated lemmatization rules if needed
            lemmatized_word = word
            if word.endswith('s'):
                lemmatized_word = word[:-1]  # Remove the 's' at the end
            lemmatized_text.append(lemmatized_word)
        return " ".join(lemmatized_text)
    """

    # Apply text normalization functions sequentially
    normalized_text = remove_articles(text)
    normalized_text = remove_punctuation(normalized_text)
    normalized_text = lowercase(normalized_text)
    #normalized_text = lemmatize(normalized_text)

    return normalized_text





def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
    print("Normalized Prediction:", bow_pred)
    print("Normalized Target:", bow_target)
    if domain == {"NUMBER"}:
        bow_pred = norm_text_to_nums(bow_pred)
        bow_target = norm_text_to_nums(bow_target)
        bow_pred = detectNum(bow_pred)
        bow_target = detectNum(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))
    else:
        # TODO: fine-grained evaluation (e.g., content words) for text question types
        bow_pred = bow_pred
        bow_target = bow_target

    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, recall, precision


def webqa_metrics_approx(prediction, ground_truth, Qcate="text"):
    color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
    shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
    yesno_set = {'yes', 'no'}

    f1, recall, precision = _webqa_acc_approx(
        prediction,
        ground_truth,
        domain={
            "color": color_set,
            "shape": shape_set,
            "YesNo": yesno_set,
            "number": {"NUMBER"},
            "text": None,
            "Others": None,
            "choose": None,
            'image': None,
        }[Qcate],
    )

    """
    if Qcate in ["color", "shape", "number", "YesNo"]:
        accuracy = f1
    else:
        accuracy = recall
    return {"acc_approx": accuracy}
    """

    return {"precision": precision, "recall": recall, "F1-score": f1}

def llava_example():
    model_path = "liuhaotian/llava-v1.5-7b"
    prompt = "What are the things I should be cautious about when I visit here?"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    answer = eval_model(args)
    print("Query:", prompt)
    print("")
    print("Answer:", answer)

def webqa_val_extraction(dataset, n):
    # Ensures val selection
    
    val_samples = {}
    for key, value in dataset.items():
        if value['split'] == 'val':
            val_samples[key] = value
    dataset = val_samples # len = 4966

    
    #dataset = dataset[:n]
    selected_samples = {}
    count = 0
    for key, value in dataset.items():
        selected_samples[key] = value
        count += 1
        if count >= n:
            break
    dataset = selected_samples
    return dataset


def keyword_extraction(sentence, top_n):
    doc = nlp(sentence) 

    # Use the noun_chunks property of the document to identify the noun phrases in the text 
    noun_phrases = [chunk.text for chunk in doc.noun_chunks] 

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words_without_stopwords = [word for word in doc.text.split() if word.lower() not in stop_words]

    # Reconstruct the sentence without stop words
    sentence_without_stopwords = ' '.join(words_without_stopwords)

    # Use term frequency-inverse document frequency (TF-IDF) analysis to rank the noun phrases 
    vectorizer = TfidfVectorizer() 
    tfidf = vectorizer.fit_transform([sentence_without_stopwords]) 

    # Get the top n most important words 
    feature_names = vectorizer.get_feature_names_out()
    top_words = [feature_names[i] for i in tfidf[0].indices[np.argsort(tfidf[0].data)][::-1][:top_n]]

    return top_words

def count_common_elements(list1, list2):
    # Convert lists to sets for efficient intersection
    set1 = set(list1)
    set2 = set(list2)

    # Find the intersection of the two sets
    intersection = set1.intersection(set2)

    # Return the count of common elements
    return len(intersection)


# Iteration 1
"""
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
"""

# Iteration 2, global retr and keyword ids
"""
def set_node_captions_keyword_module(retrieval_results, sample_data, positive_ids, query, use_km, top_n):
    retrieved_ids = []
    updated_retrieval_results = []

    query_keywords = []
    keyword_ids = []
    keyword_count_threshold = top_n // 4
    if use_km:
        query_keywords = keyword_extraction(query, top_n)

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
            if use_km:
                keyword_ids.append(str(image_id))
            updated_retrieval_results.append(res_node)
        else:
            retrieved_text_path = res_node.node.metadata.get("file_path", "")
            text_id = os.path.splitext(os.path.basename(retrieved_text_path))[0]
            
            if use_km:
                # Keyword Module
                text = res_node.node.text
                text_keywords = keyword_extraction(text, top_n)
                common_keyword_count = count_common_elements(query_keywords, text_keywords)
                if common_keyword_count >= keyword_count_threshold:
                    keyword_ids.append(str(text_id))
                    updated_retrieval_results.append(res_node)
            else:
                updated_retrieval_results.append(res_node)
            retrieved_ids.append(str(text_id))
                

    return updated_retrieval_results, retrieved_ids, keyword_ids
"""

def set_node_captions_keyword_module(retrieval_results, sample_data, positive_ids, query, use_km, top_n):
    retrieved_ids = []
    updated_retrieval_results = []

    query_keywords = []
    keyword_ids = []
    keyword_count_threshold = top_n // 4
    if use_km:
        query_keywords = keyword_extraction(query, top_n)

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
            retrieved_ids.append((str(image_id), 'i'))
            if use_km:
                keyword_ids.append((str(image_id), 'i'))
            updated_retrieval_results.append(res_node)
        else:
            retrieved_text_path = res_node.node.metadata.get("file_path", "")
            text_id = os.path.splitext(os.path.basename(retrieved_text_path))[0]
            
            if use_km:
                # Keyword Module
                text = res_node.node.text
                text_keywords = keyword_extraction(text, top_n)
                common_keyword_count = count_common_elements(query_keywords, text_keywords)
                if common_keyword_count >= keyword_count_threshold:
                    keyword_ids.append(((str(text_id)), 't'))
                    updated_retrieval_results.append(res_node)
            else:
                updated_retrieval_results.append(res_node)
            retrieved_ids.append((str(text_id), 't'))
                

    return updated_retrieval_results, retrieved_ids, keyword_ids


# Expecting only one retrieved image
"""
def construct_augmented_query(retrieval_results, query):
    obtained_facts = 1
    # If there exists a text node
    if any(not isinstance(res_node.node, ImageNode) for res_node in retrieval_results):
        augmented_query = "Consider the following facts: "
    else:
        augmented_query = ""
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
        augmented_query += f'\n\nThe image has the following caption: {image_caption}.'
    else:
        augmented_query += f'\n\nIgnore the image.'
    
    augmented_query += f"\n\nAnswer the following question: '{query}' Answer in one complete sentence."

    if not retrieved_image_path:
        #print("BLACK IMAGE RETRIEVED")
        retrieved_image_path = "./black_image.png"
    
    return augmented_query, retrieved_image_path
"""


def construct_augmented_query(retrieval_results, query):
    obtained_facts = 1
    # If there exists a text node
    if any(not isinstance(res_node.node, ImageNode) for res_node in retrieval_results):
        txt_nodes_exists = True
        augmented_query = "Consider the following facts: "
    else:
        txt_nodes_exists = False
        augmented_query = ""

    retrieved_image_paths = []
    image_captions = []

    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path = res_node.node.metadata.get("file_path", "")
            retrieved_image_paths.append(retrieved_image_path)
            image_captions.append(res_node.node.text)
        else:
            augmented_query += f'\n\n{obtained_facts}. {res_node.node.text}'
            obtained_facts += 1
    
    if retrieved_image_paths:
        if txt_nodes_exists:
            augmented_query += f'\n\n'
        if at_least_one_non_empty_caption(image_captions):
            augmented_query += f'The images have the following captions: '
            for image_caption in image_captions:
                augmented_query += (image_caption + " ")
    else:
        retrieved_image_paths.append("./black_image.png")
        augmented_query += f'\n\nIgnore the image.'
    
    augmented_query += f"\n\nAnswer the following question: '{query}' Answer in one complete sentence."
        
    return augmented_query, retrieved_image_paths


def at_least_one_non_empty_caption(image_captions):
    for caption in image_captions:
        if caption != '':
            return True
    return False


def construct_perfect_augmented_query(txt_posFacts, img_posFacts, sample_guid, query, data_folder_path):
    read_ids = []

    obtained_facts = 1
    if len(txt_posFacts) != 0:
        augmented_query = "Consider the following facts: "
    else:
        augmented_query = ""

    retrieved_image_paths = []
    image_captions = []

    for txt_posFact in txt_posFacts:
        augmented_query += f'\n\n{obtained_facts}. Title: {txt_posFact["title"]}. Fact: {txt_posFact["fact"]}'
        obtained_facts += 1
        read_ids.append(str(txt_posFact['snippet_id']))

    for img_posFact in img_posFacts:
        image_path = data_folder_path + f"/data-{sample_guid}/{img_posFact['image_id']}.jpg"
        if os.path.exists(image_path):
            retrieved_image_paths.append(image_path)
            image_captions.append(img_posFact['caption'])
            read_ids.append(str(img_posFact['image_id']))

    if retrieved_image_paths:
        if len(txt_posFacts) != 0:
            augmented_query += "\n\n"
        if at_least_one_non_empty_caption(image_captions):
            augmented_query += f'The images have the following captions: '
        for image_caption in image_captions:
            augmented_query += (image_caption + " ")
    else:
        augmented_query += f'\n\nIgnore the image.'
        retrieved_image_paths.append("./black_image.png")
    
    #augmented_query += f"\n\nAnswer the following question in a complete sentence: {query}"
    augmented_query += f"\n\nAnswer the following question: '{query}' Answer in one complete sentence."

    return augmented_query, retrieved_image_paths, read_ids


    
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
    #client = qdrant_client.QdrantClient(location=':memory:')
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
        embed_model=Settings.embed_model,
        #embed_model="local",
        #model_name='CLIP',
    )

    # RETRIEVING
    retriever_engine = index.as_retriever(
        similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k
    )

    # CLIP context length maximum: 77 tokens
    #query = query[-337:]
    retrieval_results = retriever_engine.retrieve(query)

    return retrieval_results


def rejecter_module(model_args, sample_query, retrieval_results, image_only=True, bu_image=True, bu_text=False):
    read_ids = []
    read_results = []

    print(('=' * 50) + "REJECTER MODULE START" + ('=' * 50))

    backup_image_node = None
    backup_image_id = None
    backup_text_node = None
    backup_text_id = None

    num_seen_images = 0
    num_seen_text = 0

    accepted_images = 0
    accepted_text = 0
    
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path = res_node.node.metadata.get("file_path", "")
            image_id = os.path.splitext(os.path.basename(retrieved_image_path))[0]
            image_caption = res_node.node.text

            if num_seen_images == 0:
                backup_image_node = res_node
                backup_image_id = image_id

            # Get response from llava, asking if the image can be used to answer the query.

            question = f"Can the image, with caption '{res_node.node.text}' be used to answer the query '{sample_query}'? Answer 'yes' or 'no' and provide a small explanation of your reasoning."

            model_args.query = question
            model_args.image_file = retrieved_image_path

            response = eval_model(model_args)

            print("Rejecter Query:", question)
            print("Response:", response)

            if 'yes' in response.lower():
                # If yes, keep the response and add node the node in read_results.
                read_results.append(res_node)
                read_ids.append((str(image_id), 'i'))
                accepted_images += 1
                print("Accepted source.")
            else:
                # If no, discard the response
                print("Rejected source.")
                pass

            num_seen_images += 1
        
        else:
            retrieved_text_path = res_node.node.metadata.get("file_path", "")
            text_id = os.path.splitext(os.path.basename(retrieved_text_path))[0]

            if image_only:
                read_results.append(res_node)
                read_ids.append((str(text_id), 't'))
                continue

            if num_seen_text == 0:
                backup_text_node = res_node
                backup_text_id = text_id

            # Get response from llava, asking if the text snippet can be used to answer the query.
            # Use the black image as input

            context = "Ignore the image and answer the following question:\n"
            question = f"Can the following text snippet be used to answer the query '{sample_query}'? Answer 'yes' or 'no' and provide a small explanation of your reasoning:\n{res_node.node.text}"
            query = context + question
            model_args.query = query
            model_args.image_file = './black_image.png'

            response = eval_model(model_args)

            print("Rejecter Query:", question)
            print("Response:", response)

            if 'yes' in response.lower():
                # If yes, keep the response and add node the node in read_results.
                read_results.append(res_node)
                read_ids.append((str(text_id), 't'))
                accepted_text += 1
                print("Accepted source.")
            else:
                # If no, discard the response
                print("Rejected source.")
                pass

            num_seen_text += 1

    # Backup image
    if accepted_images == 0 and backup_image_node is not None and bu_image:
        read_results.append(backup_image_node)
        read_ids.append((str(backup_image_id), 'i'))
    # Backup text
    if accepted_text == 0 and backup_text_node is not None and not image_only and bu_text:
        read_results.append(backup_text_node)
        read_ids.append((str(backup_text_id), 't'))

    print(('=' * 50) + "REJECTER MODULE END" + ('=' * 50))

    return read_results, read_ids

"""
def count_facts_in_ids(positive_ids, ids):
    # Initialize counts for each type of fact
    txt_posFacts_count = 0
    txt_negFacts_count = 0
    img_posFacts_count = 0
    img_negFacts_count = 0
    
    # Count occurrences of each type of fact in ids
    for fact_id in ids:
        if fact_id in positive_ids:
            if fact_id.startswith('d'):
                txt_posFacts_count += 1
            else:
                img_posFacts_count += 1
        else:
            if fact_id.startswith('d'):
                txt_negFacts_count += 1
            else:
                img_negFacts_count += 1
    
    return {
        'txt_posFacts': txt_posFacts_count,
        'txt_negFacts': txt_negFacts_count,
        'img_posFacts': img_posFacts_count,
        'img_negFacts': img_negFacts_count
    }
"""

def count_facts_in_ids(positive_text_ids, positive_image_ids, negative_text_ids, negative_image_ids, ids):
    # Initialize counts for each type of fact
    txt_posFacts_count = 0
    txt_negFacts_count = 0
    img_posFacts_count = 0
    img_negFacts_count = 0
    
    # Count occurrences of each type of fact in ids
    for fact_id in ids:
        if fact_id in positive_text_ids:
            txt_posFacts_count += 1
        elif fact_id in positive_image_ids:
            img_posFacts_count += 1
        elif fact_id in negative_text_ids:
            txt_negFacts_count += 1
        elif fact_id in negative_image_ids:
            img_negFacts_count += 1
                
    return {
        'txt_posFacts': txt_posFacts_count,
        'txt_negFacts': txt_negFacts_count,
        'img_posFacts': img_posFacts_count,
        'img_negFacts': img_negFacts_count
    }

def input_metrics_category(category, metrics_by_category, particular_positive_ids, particular_retrieved_ids):
    precision, recall, f1_score = calculate_retrieved_id_metrics(particular_positive_ids, particular_retrieved_ids)
    metrics_by_category[category]['precision'].append(precision)
    metrics_by_category[category]['recall'].append(recall)
    metrics_by_category[category]['F1-score'].append(f1_score)
    return metrics_by_category


def benchmark(dataset, mode, text_similarity_top_k=3, image_similarity_top_k=1, use_mmqa=False, use_km=False, top_n=10, bu_image=True, bu_text=False):
    # Define the LLAVA model path and other parameters
    """
    model_path = "liuhaotian/llava-v1.5-7b"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),  # Extract model name from path
        "conv_mode": None,
        "image_file": 'black_image.png',
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })
    """

    model_path = "liuhaotian/llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),  # Extract model name from path
        "conv_mode": None,
        "image_file": 'black_image.png',
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len": context_len,
    })


    if use_mmqa:
        categories = ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read'] + ['text', 'image']
        data_folder_path = './source_data_MMQA'
    else:
        categories = ['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape']
        if mode != 2:
            categories = ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read'] + categories
        data_folder_path = './source_data'

    # Initialize dictionaries to accumulate evaluation metrics for each question category
    metrics_by_category = {category: {'precision': [], 'recall': [], 'F1-score': []} for category in categories}

    # Iterate over each sample in the validation set
    i = 0
    for sample_guid, sample_data in dataset.items():
        question = sample_data['Q']
        positive_ids = []
        retrieved_ids = []
        read_ids = []

        # Query construction
        if mode == 1:
            # Assuming perfect retrieval of document objects
            txt_posFacts = sample_data['txt_posFacts']
            txt_negFacts = sample_data['txt_negFacts']
            img_posFacts = sample_data['img_posFacts']
            img_negFacts = sample_data['img_negFacts']

            positive_text_ids = [str(sample['snippet_id']) for sample in txt_posFacts]
            positive_image_ids = [str(sample['image_id']) for sample in img_posFacts]
            positive_ids = positive_text_ids + positive_image_ids
            negative_text_ids = [str(sample['snippet_id']) for sample in txt_negFacts]
            negative_image_ids = [str(sample['image_id']) for sample in img_negFacts]
            
            sample_data_folder_path = f"/data-{sample_guid}" 
            sample_data_folder_path = data_folder_path + sample_data_folder_path

            query = sample_data['Q']

            augmented_query, retrieved_image_paths, read_ids = construct_perfect_augmented_query(txt_posFacts, img_posFacts, sample_guid, query, data_folder_path)

            retrieved_ids = positive_ids

            precision, recall, f1_score = calculate_retrieved_id_metrics(positive_ids, retrieved_ids)
            metrics_by_category['retr']['precision'].append(precision)
            metrics_by_category['retr']['recall'].append(recall)
            metrics_by_category['retr']['F1-score'].append(f1_score)
            precision, recall, f1_score = calculate_retrieved_id_metrics(positive_ids, read_ids)
            metrics_by_category['read']['precision'].append(precision)
            metrics_by_category['read']['recall'].append(recall)
            metrics_by_category['read']['F1-score'].append(f1_score)

            query = augmented_query

            args.query = query
            args.image_file = ",".join(retrieved_image_paths)
        if mode == 2:
            # Question only
            context = "Ignore the image and answer the following question: "
            query = context + question
            args.query = query
            args.image_file = './black_image.png'
        elif mode == 3:
            # Naive RAG (may have keyword module)
            txt_posFacts = sample_data['txt_posFacts']
            txt_negFacts = sample_data['txt_negFacts']
            img_posFacts = sample_data['img_posFacts']
            img_negFacts = sample_data['img_negFacts']

            positive_text_ids = [str(sample['snippet_id']) for sample in txt_posFacts]
            positive_image_ids = [str(sample['image_id']) for sample in img_posFacts]
            positive_ids = positive_text_ids + positive_image_ids
            negative_text_ids = [str(sample['snippet_id']) for sample in txt_negFacts]
            negative_image_ids = [str(sample['image_id']) for sample in img_negFacts]
            
            sample_data_folder_path = f"/data-{sample_guid}" 
            sample_data_folder_path = data_folder_path + sample_data_folder_path

            query = sample_data['Q']

            retrieval_results = index_store_retrieve_webqa(sample_guid=sample_guid, query=query, data_folder_path=sample_data_folder_path, text_similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
            retrieval_results, retrieved_ids, keyword_ids = set_node_captions_keyword_module(retrieval_results, sample_data, positive_ids, query, use_km, top_n)

            retrieved_text_ids = [ID for (ID, modality) in retrieved_ids if modality == 't']
            retrieved_image_ids = [ID for (ID, modality) in retrieved_ids if modality == 'i']
            retrieved_ids = retrieved_text_ids + retrieved_image_ids
            keyword_text_ids = [ID for (ID, modality) in keyword_ids if modality == 't']
            keyword_image_ids = [ID for (ID, modality) in keyword_ids if modality == 'i']
            keyword_ids = keyword_text_ids + keyword_image_ids

            if use_km:
                read_text_ids = keyword_text_ids
                read_image_ids = keyword_image_ids
                read_ids = keyword_ids
            else:
                read_text_ids = retrieved_text_ids
                read_image_ids = retrieved_image_ids
                read_ids = retrieved_ids

            #print("Positive_text_ids:", positive_text_ids)
            #print("Retrieved_text_ids:", retrieved_text_ids)
            #print("Read_text_ids:", read_text_ids)

            #print("Positive_image_ids:", positive_image_ids)
            #print("Retrieved_image_ids:", retrieved_image_ids)
            #print("Read_image_ids:", read_image_ids)


            metrics_by_category = input_metrics_category('retr_t', metrics_by_category, positive_text_ids, retrieved_text_ids)
            metrics_by_category = input_metrics_category('retr_i', metrics_by_category, positive_image_ids, retrieved_image_ids)
            metrics_by_category = input_metrics_category('retr', metrics_by_category, positive_ids, retrieved_ids)
            metrics_by_category = input_metrics_category('read_t', metrics_by_category, positive_text_ids, read_text_ids)
            metrics_by_category = input_metrics_category('read_i', metrics_by_category, positive_image_ids, read_image_ids)
            metrics_by_category = input_metrics_category('read', metrics_by_category, positive_ids, read_ids)


            augmented_query, read_image_paths = construct_augmented_query(retrieval_results, query)

            query = augmented_query

            args.query = query
            args.image_file = ",".join(read_image_paths)

            

        elif mode == 4 or mode == 5:
            # RAG with rejecter module (may have keyword module)
            txt_posFacts = sample_data['txt_posFacts']
            txt_negFacts = sample_data['txt_negFacts']
            img_posFacts = sample_data['img_posFacts']
            img_negFacts = sample_data['img_negFacts']

            positive_text_ids = [str(sample['snippet_id']) for sample in txt_posFacts]
            positive_image_ids = [str(sample['image_id']) for sample in img_posFacts]
            positive_ids = positive_text_ids + positive_image_ids
            negative_text_ids = [str(sample['snippet_id']) for sample in txt_negFacts]
            negative_image_ids = [str(sample['image_id']) for sample in img_negFacts]
    
            sample_data_folder_path = f"/data-{sample_guid}" 
            sample_data_folder_path = data_folder_path + sample_data_folder_path

            query = sample_data['Q']

            retrieval_results = index_store_retrieve_webqa(sample_guid=sample_guid, query=query, data_folder_path=sample_data_folder_path, text_similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
            retrieval_results, retrieved_ids, keyword_ids = set_node_captions_keyword_module(retrieval_results, sample_data, positive_ids, query, use_km, top_n)            

            if mode == 4:
                image_only = False
            elif mode == 5:
                image_only = True
            
            read_results, read_ids = rejecter_module(args, query, retrieval_results, image_only, bu_image, bu_text)

            retrieved_text_ids = [ID for (ID, modality) in retrieved_ids if modality == 't']
            retrieved_image_ids = [ID for (ID, modality) in retrieved_ids if modality == 'i']
            retrieved_ids = retrieved_text_ids + retrieved_image_ids
            read_text_ids = [ID for (ID, modality) in read_ids if modality == 't']
            read_image_ids = [ID for (ID, modality) in read_ids if modality == 'i']
            read_ids = read_text_ids + read_image_ids

            metrics_by_category = input_metrics_category('retr_t', metrics_by_category, positive_text_ids, retrieved_text_ids)
            metrics_by_category = input_metrics_category('retr_i', metrics_by_category, positive_image_ids, retrieved_image_ids)
            metrics_by_category = input_metrics_category('retr', metrics_by_category, positive_ids, retrieved_ids)
            metrics_by_category = input_metrics_category('read_t', metrics_by_category, positive_text_ids, read_text_ids)
            metrics_by_category = input_metrics_category('read_i', metrics_by_category, positive_image_ids, read_image_ids)
            metrics_by_category = input_metrics_category('read', metrics_by_category, positive_ids, read_ids)

            augmented_query, read_image_paths = construct_augmented_query(read_results, query)

            query = augmented_query

            args.query = query
            args.image_file = ",".join(read_image_paths)



        prediction = eval_model(args)
        ground_truth = sample_data['A']
        ground_truth = ground_truth[0]
        Qcate = sample_data['Qcate']

        print({i})
        print(sample_guid)
        print("Question:", question)
        print("Query:", query)
        print("Qcate:", Qcate)
        print("Prediction:", prediction)
        print("Answer:", ground_truth)
        
        # Compute evaluation metrics for the current sample
        metrics = webqa_metrics_approx(prediction, ground_truth, Qcate)
        
        # Accumulate metrics for the current question category
        metrics_by_category[Qcate]['precision'].append(metrics['precision'])
        metrics_by_category[Qcate]['recall'].append(metrics['recall'])
        metrics_by_category[Qcate]['F1-score'].append(metrics['F1-score'])

        if mode != 2:
            print("Positive source ids:", positive_ids)
            print("Retrieved source ids:", retrieved_ids)
            print("Read source ids:", read_ids)
            num_retr_ids = count_facts_in_ids(positive_text_ids, positive_image_ids, negative_text_ids, negative_image_ids, retrieved_ids)
            num_read_ids = count_facts_in_ids(positive_text_ids, positive_image_ids, negative_text_ids, negative_image_ids, read_ids)
            for category, count in num_retr_ids.items():
                print(f"Number of retrieved {category}: {count}")
                print(f"Number of read {category}: {num_read_ids[category]}")

        print("=" * 100)

        i+=1

    # Compute overall metrics for each question category
    overall_metrics_by_category = {}
    for category, category_metrics in metrics_by_category.items():
        precision = np.mean(category_metrics['precision'])
        recall = np.mean(category_metrics['recall'])
        F1_score = np.mean(category_metrics['F1-score'])
        overall_metrics_by_category[category] = {'precision': precision, 'recall': recall, 'F1-score': F1_score}

    # Computing average overall categories:

    if use_mmqa:
        overall_precision = np.mean([metrics['precision'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])
        overall_recall = np.mean([metrics['recall'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])
        overall_f1_score = np.mean([metrics['F1-score'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])
        overall_metrics_by_category['overall'] = {'precision': overall_precision, 'recall': overall_recall, 'F1-score': overall_f1_score}
        categories = categories + ['overall']
    else:
        # Define the categories to exclude in overall score (included all except 'retr' in overall_t)
        categories_to_exclude = ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read'] + ['Others', 'choose']

        # Compute overall metrics excluding the specified categories
        overall_precision = np.mean([metrics['precision'] for category, metrics in overall_metrics_by_category.items() if category not in categories_to_exclude])
        overall_recall = np.mean([metrics['recall'] for category, metrics in overall_metrics_by_category.items() if category not in categories_to_exclude])
        overall_f1_score = np.mean([metrics['F1-score'] for category, metrics in overall_metrics_by_category.items() if category not in categories_to_exclude])

        overall_precision_t = np.mean([metrics['precision'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])
        overall_recall_t = np.mean([metrics['recall'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])
        overall_f1_score_t = np.mean([metrics['F1-score'] for category, metrics in overall_metrics_by_category.items() if category not in ['retr_t', 'retr_i', 'retr', 'read_t', 'read_i', 'read']])

        # Add the overall metrics to the dictionary
        overall_metrics_by_category['overall'] = {'precision': overall_precision, 'recall': overall_recall, 'F1-score': overall_f1_score}
        overall_metrics_by_category['overall_t'] = {'precision': overall_precision_t, 'recall': overall_recall_t, 'F1-score': overall_f1_score_t}

        categories = categories + ['overall', 'overall_t']


    print("")
    print(("="*25) + " Final Results " + ('='*25))

    #print(categories)
    #pprint(metrics_by_category)
    #pprint(overall_metrics_by_category)

    

    for category in categories:
        print('')
        print("Category:", category)
        print(f"Precision: {round(overall_metrics_by_category[category]['precision']*100, 2)}")
        if category in ["color", "shape", "number", "YesNo"]:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)} <---- Primary Metric")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)}")
        elif category in ['choose', 'Others', 'text']:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)}")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)} <---- Primary Metric")
        else:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)}")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)}")
    

def main(FLAGS):
    # setup device to use
    #device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    #print("Device:", device)

    """
    model_path = "liuhaotian/llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    """
    if FLAGS.use_mmqa:
        dataset = json.load(open("./datasets/MMQA/annotations/MMQA.json", "r")) # len = 951
        missing_guids = ['725cff35b53790928967f609bb24985f', '9c4edd8d739fd174c3e3def12de3e7d8', '074182b0a52f0fd0e740b4f64b30113c', '68e8165c0eed4cd49ae2ed8947dbf41d', '47800dd3d7edae3069eb4e683d1889ef', 'a6cb73d18b79117f01e6d814c07a60f7', '4f183e594eda0238cc1d8f06a8ac4cae', '819e3160f22972d6db17a64a3ba79bb2', '08fac605c314afb1b2121b278dbfc41b', 'ced237f8af0c8f6b8b1989a48f0b0969', 'c45775178d1457f3250348da7f2763a9', '28bbe6fa4a8f5277ba4354085ec4663f', '2576221a02209dfcc4ce4e847c231445', '7664500bdee6462bcdd353d855b077ee', '38027a2b6b1a3d30030572e199d0b885', '4ffd86699e830254993604a40ee1f807', '3635238b6da69b7d89e0eecc3cbfd62b', 'c7b0f4a0bce4148f29dbb78be8e8efbd', '7200b883e81509f5486d3fc8cee35417', 'de4a4b0fdd903eadbf8bc1e60b083467', 'f39db465d59d6a5b2146cb4cf9c8a5e9', '20c33d307142256cebf255fd3b767753', 'f02a2e164a60e8b3a66946b558aab288', '59e62161a30fc6eaee3dc916f66eca0e', '8a39ecdb53fa36aa6c65f4042f54c45f', '9e7649460743d4fb42f1b30558c6ee30', '2494ebf69cf93104339896d73448af5d', '3710f699d71011766460064b5a00ec06', '4840be576c674b4c2f76d567932f8990', 'ccab4dc57019d3e03961f845e7e6cea3', 'b07b21de209a6dbb3c69f42a1f2044e5', 'ff4e4666b70c68522ee408cd874369d3', 'ecbbb1949b144a35fd9f916cb4270a72', '27daa3331e30a8f02c7ede6991fe84a5', '9778cc053c0a0f168309d3d4f40d7230', '9133577d0fbf19f3fc096ce431f7d2f9', '18ff961c71fb70bf1b9888caf790ca08', 'd31bb6e3704b9022a0ed4accdf8ec5d2', 'f70bae73f07c5fe18055500fda9e21d0', '55650fcaa3b2911df0940e46a5721b07']
        for guid in missing_guids:
            if guid in dataset:
                dataset.pop(guid)
        if FLAGS.n == 0:
            n = len(dataset)
        else:
            n = FLAGS.n
        selected_samples = {}
        count = 0
        for key, value in dataset.items():
            selected_samples[key] = value
            count += 1
            if count >= n:
                break
        dataset = selected_samples # len = 911
    else:
        dataset = json.load(open("datasets/WebQA/annotations/WebQA_train_val.json", "r"))
        if FLAGS.n == 0:
            n = len(dataset)
        else:
            n = FLAGS.n
        dataset = webqa_val_extraction(dataset, n)

    text_similarity_top_k=FLAGS.text_ret
    image_similarity_top_k=FLAGS.image_ret

    print(f"{len(dataset)} samples loaded.")
    print("Question Categories: ", Counter([dataset[k]['Qcate'] for k in dataset]))
    print("FLAGS.mode ==", FLAGS.mode)

    if FLAGS.use_mmqa:
        dataset_name = "MultimodalQA"
    else:
        dataset_name = "WebQA"

    if FLAGS.mode == 1:
        # Assuming perfect retrieval

        print(f"LLaVA-{dataset_name} Assuming Perfect Retrieval")
        #benchmark(dataset, FLAGS.mode, use_mmqa=FLAGS.use_mmqa)
    elif FLAGS.mode == 2:
        # Question only

        print(f"LLaVA-{dataset_name} Question Only")

        #benchmark(dataset, FLAGS.mode, use_mmqa=FLAGS.use_mmqa)
    elif FLAGS.mode == 3:
        # Naive RAG

        print(f"LLaVA-{dataset_name} Naive-RAG ({text_similarity_top_k}, {image_similarity_top_k})")

        #benchmark(dataset, FLAGS.mode, text_similarity_top_k, image_similarity_top_k, use_mmqa=FLAGS.use_mmqa)
    elif FLAGS.mode == 4:
        # RAG, Rejector Module

        print(f"LLaVA-{dataset_name} RAG with Rejecter Module ({text_similarity_top_k}, {image_similarity_top_k})")
        print(f"Image Backup: {FLAGS.bu_image}, Text Backup: {FLAGS.bu_text}")
        #benchmark(dataset, FLAGS.mode, text_similarity_top_k, image_similarity_top_k, use_mmqa=FLAGS.use_mmqa)
    elif FLAGS.mode == 5:
        # RAG, Image-only Rejector Module

        print(f"LLaVA-{dataset_name} RAG with Image-only Rejecter Module ({text_similarity_top_k}, {image_similarity_top_k})")
        print(f"Image Backup: {FLAGS.bu_image}, Text Backup: {FLAGS.bu_text}")
    
    if FLAGS.use_km:
        print(f"Using Keyword Module, top {FLAGS.top_n} keywords.")
    print("")
    benchmark(dataset, FLAGS.mode, text_similarity_top_k, image_similarity_top_k, use_mmqa=FLAGS.use_mmqa, use_km=FLAGS.use_km, top_n=FLAGS.top_n, bu_image=FLAGS.bu_image, bu_text=FLAGS.bu_text)






if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal RAG')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode to run.')
    parser.add_argument('--use_mmqa',
                        action="store_true",
                        help='Use MMQA instead of WebQA as a benchmark')
    parser.add_argument('--text_ret',
                        type=int, default=3,
                        help='text_similarity_top_k')
    parser.add_argument('--image_ret',
                        type=int, default=1,
                        help='image_similarity_top_k')
    parser.add_argument('--n',
                        type=int, default=0,
                        help='Number of samples to use. Leave blank for full set.')
    parser.add_argument('--use_km',
                        action="store_true",
                        help='Use keyword module')
    parser.add_argument('--top_n',
                        type=int, default=10,
                        help='Top n keywords used by the keyword module')
    parser.add_argument('--bu_image',
                        action="store_true",
                        help='Read at least one image from the rejecter module')
    parser.add_argument('--bu_text',
                        action="store_true",
                        help='Read at least one text_snippet from the rejecter module')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)