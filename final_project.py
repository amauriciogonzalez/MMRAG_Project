from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse
import json
from PIL import Image
from collections import Counter

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

    # Load the dataset
    dataset = json.load(open("datasets/WebQA/annotations/WebQA_test.json", "r"))

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
    i = 0

    for key in dataset:

        if i == 3:
            break

        context = "Ignore the image and answer the following question: "

        # Extract question from the sample
        question = dataset[key]['Q']
        
        # Set the query parameter for LLAVA
        args.query = question
        
        # Perform evaluation using LLAVA
        answer = eval_model(args)
        
        # Store the answer in the output dictionary
        output_dict[key] = {'sources': [], 'answer': answer}
        
        print("=" * 100)
        print(key)
        print("Query:", context + question)
        print("Prediction:", answer)
        print("=" * 100)

        i += 1


    # Print the output dictionary
    print("Output dictionary:", output_dict)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal RAG')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select example to run.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)