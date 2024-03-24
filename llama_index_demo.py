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


#image_documents = SimpleDirectoryReader(input_files=["./cars/o1.jpg","./cars/t1.jpg","./cars/v1.jpg"]).load_data()
image_documents = SimpleWebPageReader().load_data(["https://llava-vl.github.io/static/images/view.jpg"])
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

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

# Create the MultiModal index
#documents = SimpleDirectoryReader("./cars").load_data()

documents = SimpleWebPageReader().load_data(["https://llava-vl.github.io/static/images/view.jpg"])

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model="local",
)

# generate Text retrieval results
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
retrieval_results = retriever_engine.retrieve("Water")

print("Hi!")

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        print("An image node:")
        print(res_node)
    else:
        #display_source_node(res_node, source_length=200)
        print("Not an image node:")
        print(res_node)


print("=" * 20)
print("Dataset Application")
print("=" * 20)

