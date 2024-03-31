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


#import spacy 
#import en_core_web_sm   

#pip install -U pip setuptools wheel
#pip install -U spacy
#python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser, NER, and word vectors
#nlp = spacy.load("en_core_web_sm")


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


def llava_question_only_webqa_test(dataset, save_output):
    # Define the LLAVA model path and other parameters
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

    # Initialize the output dictionary
    output_dict = {}

    # Iterate over each sample in the dataset
    for index, key in enumerate(dataset):
        context = "Ignore the image and answer the following question: "

        # Extract question from the sample
        question = dataset[key]['Q']
        
        # Set the query parameter for LLAVA
        args.query = context + question
        
        # Perform evaluation using LLAVA
        answer = eval_model(args)
        
        # Store the answer in the output dictionary
        output_dict[key] = {'sources': [], 'answer': answer}
        
        print("=" * 100)
        print({index})
        print(key)
        print("Query:", context + question)
        print("Prediction:", answer)
        print("=" * 100)

    print("Total answered questions:", len(output_dict))

    # Print the output dictionary
    print("Output dictionary:", output_dict)

    if save_output:
        file_path = "json-output/llava_question_only_test.json"

        with open(file_path, "w") as json_file:
            json.dump(output_dict, json_file)

        print("JSON file saved successfully.")

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
    
    augmented_query += f"\n\nAnswer the following question: {query}"

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


def llava_webqa_val(dataset, mode, text_similarity_top_k=3, image_similarity_top_k=1):
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



    categories = ['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape']
    if mode == 3:
        categories = ['retr'] + categories

    # Initialize dictionaries to accumulate evaluation metrics for each question category
    metrics_by_category = {category: {'precision': [], 'recall': [], 'F1-score': []} for category in categories}

    # Iterate over each sample in the validation set
    i = 0
    for sample_guid, sample_data in dataset.items():
        question = sample_data['Q']

        # Query construction
        if mode == 2:
            # Question only
            context = "Ignore the image and answer the following question: "
            query = context + question
            args.query = query
            args.image_file = './black_image.png'
        elif mode == 3:
            # Naive RAG
            txt_posFacts = sample_data['txt_posFacts']
            txt_negFacts = sample_data['txt_negFacts']
            img_posFacts = sample_data['img_posFacts']
            img_negFacts = sample_data['img_negFacts']

            positive_ids = [str(sample['snippet_id']) for sample in txt_posFacts] + [str(sample['image_id']) for sample in img_posFacts]

            data_folder_path = './source_data'
            
            sample_data_folder_path = f"/data-{sample_guid}" 
            sample_data_folder_path = data_folder_path + sample_data_folder_path

            query = sample_data['Q']

            retrieval_results = index_store_retrieve_webqa(sample_guid=sample_guid, query=query, data_folder_path=sample_data_folder_path, text_similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
            retrieval_results, retrieved_ids = set_webqa_node_captions(retrieval_results, sample_data, positive_ids)

            precision, recall, f1_score = calculate_retrieved_id_metrics(positive_ids, retrieved_ids)
            metrics_by_category['retr']['precision'].append(precision)
            metrics_by_category['retr']['recall'].append(recall)
            metrics_by_category['retr']['F1-score'].append(f1_score)

            augmented_query, retrieved_image_path = construct_augmented_query(retrieval_results, query)
            query = augmented_query


            args.query = query
            args.image_file = retrieved_image_path



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

        print("=" * 100)

        i+=1

    # Compute overall metrics for each question category
    overall_metrics_by_category = {}
    for category, category_metrics in metrics_by_category.items():
        precision = np.mean(category_metrics['precision'])
        recall = np.mean(category_metrics['recall'])
        F1_score = np.mean(category_metrics['F1-score'])
        overall_metrics_by_category[category] = {'precision': precision, 'recall': recall, 'F1-score': F1_score}

    print("")
    print(("="*25) + " Final Results " + ('='*25))

    #print(categories)
    #pprint(metrics_by_category)
    #pprint(overall_metrics_by_category)

    for category in categories:
        print('')
        print("Category:", category)
        print(f"Precision: {round(overall_metrics_by_category[category]['precision']*100, 2)}")
        if category in ['retr']:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)}")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)}")
            continue
        if category in ["color", "shape", "number", "YesNo"]:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)} <---- Primary Metric")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)}")
        else:
            print(f"Recall: {round(overall_metrics_by_category[category]['recall']*100, 2)}")
            print(f"F1-score: {round(overall_metrics_by_category[category]['F1-score']*100, 2)} <---- Primary Metric")
    

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


    if FLAGS.mode == 1:
        # Question only, test

        dataset = json.load(open("datasets/WebQA/annotations/WebQA_test.json", "r"))
        llava_question_only_webqa_test(dataset, FLAGS.save_output)
    if FLAGS.mode == 2:
        # Question only, val

        dataset = json.load(open("datasets/WebQA/annotations/WebQA_train_val.json", "r"))
        if FLAGS.n == 0:
            n = len(dataset)
        else:
            n = FLAGS.n
        dataset = webqa_val_extraction(dataset, n)

        print(f"{len(dataset)} samples loaded.")
        print("Question Categories: ", Counter([dataset[k]['Qcate'] for k in dataset]))
        print("LLaVA-WebQA Question Only")
        print("")

        llava_webqa_val(dataset, FLAGS.mode)
    if FLAGS.mode == 3:
        # Naive RAG, val

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
        print(f"LLaVA-WebQA Naive-RAG ({text_similarity_top_k}, {image_similarity_top_k})")
        print("")

        llava_webqa_val(dataset, FLAGS.mode, text_similarity_top_k, image_similarity_top_k)






if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal RAG')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode to run.')
    parser.add_argument('--save_output',
                        action="store_true",
                        help='Saves output as a json file')
    parser.add_argument('--text_ret',
                        type=int, default=3,
                        help='text_similarity_top_k')
    parser.add_argument('--image_ret',
                        type=int, default=1,
                        help='image_similarity_top_k')
    parser.add_argument('--n',
                        type=int, default=0,
                        help='Number of samples to use. Leave blank for full set.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)