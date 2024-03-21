from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse
import json
from PIL import Image
from collections import Counter

from word2number import w2n
import string, re
import spacy 
import en_core_web_sm   

#pip install -U pip setuptools wheel
#pip install -U spacy
#python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser, NER, and word vectors
#nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()


"""
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




def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
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
    if Qcate in ["color", "shape", "number", "YesNo"]:
        accuracy = f1
    else:
        accuracy = recall
    return {"acc_approx": accuracy}

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


def llava_question_only_webqa_val(dataset):
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



    categories = ['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape']

    # Initialize dictionaries to accumulate evaluation metrics for each question category
    metrics_by_category = {category: {'precision': [], 'recall': [], 'F1-score': []} for category in categories}

    # Iterate over each sample in the validation set
    i = 0
    for sample_guid, sample_data in dataset.items():
        # Extract predicted and ground truth answers along with the question category
        context = "Ignore the image and answer the following question: "
        question = sample_data['Q']
        args.query = context + question
        prediction = eval_model(args) # NEED TO EXPAND THIS
        ground_truth = sample_data['A']
        Qcate = sample_data['Qcate']

        print("Question:", question)
        print("Qcate:", Qcate)
        print("prediction:", prediction)
        print("Answer:", ground_truth)
        
        # Compute evaluation metrics for the current sample
        metrics = webqa_metrics_approx(prediction, ground_truth, Qcate)
        
        # Accumulate metrics for the current question category
        metrics_by_category[Qcate]['precision'].append(metrics['precision'])
        metrics_by_category[Qcate]['recall'].append(metrics['recall'])
        metrics_by_category[Qcate]['F1-score'].append(metrics['F1-score'])

        print({index})
        print(sample_guid)
        print("Query:", context + question)
        print("Prediction:", answer)
        print("=" * 100)

        i+=1

    # Compute overall metrics for each question category
    overall_metrics_by_category = {}
    for category, category_metrics in metrics_by_category.items():
        precision = np.mean(category_metrics['precision'])
        recall = np.mean(category_metrics['recall'])
        F1_score = np.mean(category_metrics['F1-score'])
        overall_metrics_by_category[category] = {'precision': precision, 'recall': recall, 'F1-score': F1_score}

    # Print or return the overall metrics for each question category
    print("Final Results:")
    print(overall_metrics_by_category)
    

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

        # Ensures val selection
        val_samples = {}
        for key, value in dataset.items():
            if value['split'] == 'val':
                val_samples[key] = value
        dataset = val_samples # len = 4966

        n = 20
        #dataset = dataset[:n]
        selected_samples = {}
        count = 0
        for key, value in dataset.items():
            selected_samples[key] = value
            count += 1
            if count >= n:
                break
        dataset = selected_samples

        print(f"{len(dataset)} samples loaded.")
        llava_question_only_webqa_val(dataset)





if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal RAG')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode to run.')
    parser.add_argument('--save_output',
                        action="store_true",
                        help='Saves output as a json file')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)